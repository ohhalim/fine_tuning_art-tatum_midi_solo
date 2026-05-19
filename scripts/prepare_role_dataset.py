"""
Stage A role-conditioned dataset builder.

Creates:
  data/roles/<role>/<sample_id>/conditioning.mid
  data/roles/<role>/<sample_id>/target.mid
  data/roles/<role>/<sample_id>/meta.json
  data/roles/<role>/tokenized/{train,val}/*.npy
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pretty_midi

# Add music_transformer to path
SCRIPT_DIR = Path(__file__).parent.parent
import sys

sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from midi_processor.processor import RANGE_NOTE_ON, RANGE_NOTE_OFF, RANGE_TIME_SHIFT

try:
    from control_tokens import (
        SEQUENCE_FORMAT_CHOICES,
        SEQUENCE_FORMAT_CONTROL_V1,
        SEQUENCE_FORMAT_LEGACY_SEP,
        build_control_sequence,
        build_legacy_sequence,
        control_prefix_tokens,
        token_names,
    )
    from stage_b_tokens import SEQUENCE_FORMAT_STAGE_B_V1, build_stage_b_sequence
except ModuleNotFoundError:
    from scripts.control_tokens import (
        SEQUENCE_FORMAT_CHOICES,
        SEQUENCE_FORMAT_CONTROL_V1,
        SEQUENCE_FORMAT_LEGACY_SEP,
        build_control_sequence,
        build_legacy_sequence,
        control_prefix_tokens,
        token_names,
    )
    from scripts.stage_b_tokens import SEQUENCE_FORMAT_STAGE_B_V1, build_stage_b_sequence


SEQUENCE_FORMAT_CHOICES_WITH_STAGE_B = SEQUENCE_FORMAT_CHOICES + (SEQUENCE_FORMAT_STAGE_B_V1,)


def find_midi_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []
    files.extend(input_dir.rglob("*.mid"))
    files.extend(input_dir.rglob("*.midi"))
    # deterministic ordering before shuffle/sampling
    return sorted(files)


def read_midi_manifest(manifest_path: Path) -> List[Path]:
    files: List[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        files.append(Path(line))
    return files


def resolve_input_splits(args: argparse.Namespace) -> tuple[dict[str, List[Path]], str]:
    if args.train_manifest or args.val_manifest:
        if not args.train_manifest or not args.val_manifest:
            raise ValueError("--train_manifest and --val_manifest must be provided together")
        if args.max_files is not None:
            raise ValueError("--max_files is not supported with explicit train/val manifests")

        train_manifest = Path(args.train_manifest)
        val_manifest = Path(args.val_manifest)
        train_files = read_midi_manifest(train_manifest)
        val_files = read_midi_manifest(val_manifest)
        if not train_files:
            raise ValueError(f"No MIDI paths found in train manifest: {train_manifest}")
        if not val_files:
            raise ValueError(f"No MIDI paths found in val manifest: {val_manifest}")
        return {"train": train_files, "val": val_files}, "manifest"

    input_dir = Path(args.input_dir)
    midi_files = find_midi_files(input_dir)
    if not midi_files:
        raise ValueError(f"No MIDI files found under: {input_dir}")

    if args.max_files is not None:
        midi_files = midi_files[: max(0, args.max_files)]
    return {"unsplit": midi_files}, "directory"


def clone_note(note: pretty_midi.Note, pitch_shift: int = 0) -> pretty_midi.Note:
    pitch = note.pitch + pitch_shift
    return pretty_midi.Note(
        velocity=int(note.velocity),
        pitch=int(pitch),
        start=float(note.start),
        end=float(note.end),
    )


def transpose_notes_if_valid(
    notes: Sequence[pretty_midi.Note], semitones: int
) -> List[pretty_midi.Note]:
    transposed: List[pretty_midi.Note] = []
    for n in notes:
        shifted = n.pitch + semitones
        if shifted < 0 or shifted > 127:
            return []
        transposed.append(clone_note(n, pitch_shift=semitones))
    return transposed


def build_sparse_conditioning(
    notes: Sequence[pretty_midi.Note], window_sec: float = 0.5
) -> List[pretty_midi.Note]:
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda x: (x.start, x.pitch))
    anchors: List[pretty_midi.Note] = []

    cur_window_start = sorted_notes[0].start
    bucket: List[pretty_midi.Note] = []

    def flush_bucket(b: Sequence[pretty_midi.Note]) -> None:
        if not b:
            return
        anchor = min(b, key=lambda x: x.pitch)
        end = min(anchor.end, anchor.start + 0.8)
        if end <= anchor.start:
            end = anchor.start + 0.1
        anchors.append(
            pretty_midi.Note(
                velocity=max(40, min(100, anchor.velocity)),
                pitch=max(24, anchor.pitch - 12),
                start=float(anchor.start),
                end=float(end),
            )
        )

    for n in sorted_notes:
        while n.start >= cur_window_start + window_sec:
            flush_bucket(bucket)
            bucket = []
            cur_window_start += window_sec
        bucket.append(n)

    flush_bucket(bucket)
    return anchors


def split_conditioning_target(
    notes: Sequence[pretty_midi.Note],
    split_pitch: int,
    min_conditioning_notes: int,
    min_target_notes: int,
) -> Tuple[List[pretty_midi.Note], List[pretty_midi.Note]]:
    conditioning = [n for n in notes if n.pitch <= split_pitch]
    target = [n for n in notes if n.pitch > split_pitch]

    # If right-hand extraction is too sparse, keep all notes for target.
    if len(target) < min_target_notes:
        target = list(notes)

    # If left-hand extraction is too sparse, synthesize sparse guide notes.
    if len(conditioning) < min_conditioning_notes:
        conditioning = build_sparse_conditioning(notes)

    return conditioning, target


def notes_to_midi(notes: Sequence[pretty_midi.Note], tempo_bpm: float) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=max(20.0, float(tempo_bpm)))
    inst = pretty_midi.Instrument(program=0, is_drum=False, name="piano")
    inst.notes = sorted(list(notes), key=lambda x: (x.start, x.pitch))
    midi.instruments.append(inst)
    return midi


def infer_tempo(pm: pretty_midi.PrettyMIDI) -> float:
    try:
        _, tempi = pm.get_tempo_changes()
        if len(tempi) > 0:
            tempo = float(tempi[0])
            if tempo > 0:
                return tempo
    except Exception:
        pass
    return 120.0


def iter_transpositions(transpose_all_keys: bool, transpose_range: int) -> Iterable[int]:
    if transpose_all_keys:
        # 12 keys around original range
        return range(-6, 6)
    if transpose_range <= 0:
        return [0]
    return range(-transpose_range, transpose_range + 1)


def encode_midi_simple(file_path: str) -> List[int]:
    start_idx = {
        "note_on": 0,
        "note_off": RANGE_NOTE_ON,
        "time_shift": RANGE_NOTE_ON + RANGE_NOTE_OFF,
        "velocity": RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT,
    }

    mid = pretty_midi.PrettyMIDI(midi_file=file_path)

    all_notes = []
    for inst in mid.instruments:
        if not inst.is_drum:
            all_notes.extend(inst.notes)

    if not all_notes:
        return []

    events = []
    cur_time = 0.0
    cur_vel = 0

    note_events = []
    for note in all_notes:
        note_events.append(("note_on", note.start, note.pitch, note.velocity))
        note_events.append(("note_off", note.end, note.pitch, 0))
    note_events.sort(key=lambda x: (x[1], x[2], 0 if x[0] == "note_off" else 1))

    for event_type, event_time, pitch, velocity in note_events:
        time_diff = max(0.0, event_time - cur_time)
        time_steps = int(round(time_diff * 100))
        while time_steps >= RANGE_TIME_SHIFT:
            events.append(start_idx["time_shift"] + RANGE_TIME_SHIFT - 1)
            time_steps -= RANGE_TIME_SHIFT
        if time_steps > 0:
            events.append(start_idx["time_shift"] + time_steps - 1)

        if event_type == "note_on":
            mod_vel = int(max(0, min(31, velocity // 4)))
            if mod_vel != cur_vel:
                events.append(start_idx["velocity"] + mod_vel)
                cur_vel = mod_vel
            events.append(start_idx["note_on"] + int(pitch))
        else:
            events.append(start_idx["note_off"] + int(pitch))

        cur_time = event_time

    return events


def load_midi_notes(file_path: str | Path) -> List[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(midi_file=str(file_path))
    notes: List[pretty_midi.Note] = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def build_training_sequence(
    cond_tokens: Sequence[int],
    tgt_tokens: Sequence[int],
    meta: dict,
    sequence_format: str,
    default_role: str,
) -> list[int]:
    if sequence_format == SEQUENCE_FORMAT_CONTROL_V1:
        return build_control_sequence(
            cond_tokens,
            tgt_tokens,
            role=str(meta.get("role", default_role)),
            tempo_bpm=float(meta.get("tempo_bpm", 120.0)),
        )
    if sequence_format == SEQUENCE_FORMAT_LEGACY_SEP:
        return build_legacy_sequence(cond_tokens, tgt_tokens)
    raise ValueError(f"Unsupported sequence_format: {sequence_format}")


def build_stage_b_record_sequence(record: dict, meta: dict, default_role: str) -> list[int]:
    target_notes = load_midi_notes(record["target_path"])
    if not target_notes:
        return []
    return build_stage_b_sequence(
        target_notes,
        tempo_bpm=float(meta.get("tempo_bpm", 120.0)),
        chords=meta.get("chord_progression"),
        role=str(meta.get("role", default_role)),
    )


def write_tokenized_records(
    records: Sequence[dict],
    split_name: str,
    split_dir: Path,
    sequence_format: str,
    default_role: str,
) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, record in enumerate(records):
        try:
            meta = json.loads(Path(record["meta_path"]).read_text())
            if sequence_format == SEQUENCE_FORMAT_STAGE_B_V1:
                seq = build_stage_b_record_sequence(record, meta=meta, default_role=default_role)
            else:
                cond_tokens = encode_midi_simple(str(record["conditioning_path"]))
                tgt_tokens = encode_midi_simple(str(record["target_path"]))
                if not cond_tokens or not tgt_tokens:
                    continue
                seq = build_training_sequence(
                    cond_tokens,
                    tgt_tokens,
                    meta=meta,
                    sequence_format=sequence_format,
                    default_role=default_role,
                )
        except Exception as e:
            print(f"Tokenize skip ({split_name}): {record['sample_id']} ({e})")
            continue

        if not seq:
            continue
        np.save(split_dir / f"{i:06d}.npy", np.array(seq, dtype=np.int32))
        saved += 1
    return saved


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare role-conditioned MIDI dataset")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio/Brad Mehldau")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data/roles")
    parser.add_argument("--role", type=str, default="lead")
    parser.add_argument("--split_pitch", type=int, default=60, help="conditioning uses <= split_pitch")
    parser.add_argument("--min_conditioning_notes", type=int, default=8)
    parser.add_argument("--min_target_notes", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--transpose_range", type=int, default=0)
    parser.add_argument("--transpose_all_keys", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sequence_format",
        choices=SEQUENCE_FORMAT_CHOICES_WITH_STAGE_B,
        default=SEQUENCE_FORMAT_CONTROL_V1,
        help="Token sequence format for conditioning/target training examples.",
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir) / args.role

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    input_splits, input_mode = resolve_input_splits(args)
    total_input_files = sum(len(files) for files in input_splits.values())
    print(f"Found {total_input_files} MIDI files via {input_mode} input")

    sample_records: List[dict] = []
    manifest_records: dict[str, List[dict]] = {"train": [], "val": []}
    sample_idx = 0

    for source_split, midi_files in input_splits.items():
        for midi_path in midi_files:
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_path))
            except Exception as e:
                print(f"Skip unreadable MIDI: {midi_path} ({e})")
                continue

            all_notes: List[pretty_midi.Note] = []
            for inst in pm.instruments:
                if not inst.is_drum:
                    all_notes.extend(inst.notes)

            if len(all_notes) < max(args.min_conditioning_notes, args.min_target_notes):
                continue

            all_notes = sorted(all_notes, key=lambda x: (x.start, x.pitch))
            tempo = infer_tempo(pm)
            base_conditioning, base_target = split_conditioning_target(
                all_notes,
                split_pitch=args.split_pitch,
                min_conditioning_notes=args.min_conditioning_notes,
                min_target_notes=args.min_target_notes,
            )

            if len(base_conditioning) < 2 or len(base_target) < 8:
                continue

            for semitones in iter_transpositions(args.transpose_all_keys, args.transpose_range):
                conditioning = transpose_notes_if_valid(base_conditioning, semitones)
                target = transpose_notes_if_valid(base_target, semitones)
                if not conditioning or not target:
                    continue

                sample_id = f"{sample_idx:06d}"
                sample_dir = output_root / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)

                cond_path = sample_dir / "conditioning.mid"
                tgt_path = sample_dir / "target.mid"
                meta_path = sample_dir / "meta.json"

                notes_to_midi(conditioning, tempo).write(str(cond_path))
                notes_to_midi(target, tempo).write(str(tgt_path))

                control_tokens = control_prefix_tokens(role=args.role, tempo_bpm=tempo)
                meta = {
                    "sample_id": sample_id,
                    "role": args.role,
                    "source_midi": str(midi_path),
                    "source_split": source_split if input_mode == "manifest" else None,
                    "transpose": int(semitones),
                    "split_pitch": int(args.split_pitch),
                    "tempo_bpm": float(tempo),
                    "conditioning_notes": len(conditioning),
                    "target_notes": len(target),
                    "sequence_format": args.sequence_format,
                    "control_tokens": control_tokens,
                    "control_token_names": token_names(control_tokens),
                }
                meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2))

                record = {
                    "sample_id": sample_id,
                    "conditioning_path": cond_path,
                    "target_path": tgt_path,
                    "meta_path": meta_path,
                    "source_split": source_split,
                }
                sample_records.append(record)
                if input_mode == "manifest":
                    manifest_records[source_split].append(record)
                sample_idx += 1

    if not sample_records:
        raise ValueError("No usable samples generated. Try lowering split/threshold parameters.")

    print(f"Generated {len(sample_records)} role samples under {output_root}")

    if input_mode == "manifest":
        train_records = manifest_records["train"]
        val_records = manifest_records["val"]
        if not train_records:
            raise ValueError("Train manifest produced no usable samples")
        if not val_records:
            raise ValueError("Val manifest produced no usable samples")
    else:
        random.shuffle(sample_records)
        split_idx = int(len(sample_records) * args.train_ratio)
        train_records = sample_records[:split_idx]
        val_records = sample_records[split_idx:]
        if not val_records:
            # Keep at least one sample for val for stable training scripts.
            val_records = train_records[-1:]
            train_records = train_records[:-1]

    tokenized_root = output_root / "tokenized"
    train_dir = tokenized_root / "train"
    val_dir = tokenized_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_count = write_tokenized_records(train_records, "train", train_dir, args.sequence_format, args.role)
    val_count = write_tokenized_records(val_records, "val", val_dir, args.sequence_format, args.role)

    summary = {
        "role": args.role,
        "input_mode": input_mode,
        "input_dir": str(input_dir) if input_mode == "directory" else None,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "input_file_count": total_input_files,
        "input_split_file_counts": {name: len(files) for name, files in input_splits.items()},
        "output_root": str(output_root),
        "num_samples": len(sample_records),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "tokenized_train": train_count,
        "tokenized_val": val_count,
        "transpose_all_keys": bool(args.transpose_all_keys),
        "transpose_range": int(args.transpose_range),
        "sequence_format": args.sequence_format,
        "seed": int(args.seed),
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2)
    )

    print("Done")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
