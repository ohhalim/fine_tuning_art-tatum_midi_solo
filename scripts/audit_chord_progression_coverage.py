"""Audit chord-progression annotation coverage before pitch-role reference work."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


CHORD_FIELD_KEYWORDS = (
    "chord",
    "chords",
    "chord_progression",
    "harmony",
    "harmonic",
    "changes",
    "lead_sheet",
    "leadsheet",
)

SIDECAR_SUFFIXES = {
    ".json",
    ".csv",
    ".tsv",
    ".txt",
    ".lab",
    ".jams",
    ".xml",
    ".musicxml",
    ".mxl",
}

CHORD_SYMBOL_RE = re.compile(
    r"(?<![A-Za-z0-9#b])"
    r"([A-G](?:#|b)?(?:maj7|maj9|maj13|maj|min7|min9|min13|min|m7b5|m7|m9|m13|m|"
    r"dim7|dim|aug|sus4|sus2|sus|add9|6|7|9|11|13|ø|°|Δ)"
    r"(?:/[A-G](?:#|b)?)?)"
    r"(?![A-Za-z0-9#b])"
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def count_chord_symbols(text: str) -> int:
    return len(CHORD_SYMBOL_RE.findall(text))


def normalize_key(key: str) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def is_chord_key(key: str) -> bool:
    normalized = normalize_key(key)
    return any(keyword in normalized for keyword in CHORD_FIELD_KEYWORDS)


def value_has_chord_like_content(value: Any) -> bool:
    if isinstance(value, str):
        return count_chord_symbols(value) > 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(value_has_chord_like_content(item) for item in value)
    if isinstance(value, dict):
        return any(value_has_chord_like_content(item) for item in value.values())
    return False


def find_chord_fields(payload: Any, prefix: str = "") -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if is_chord_key(str(key)) or value_has_chord_like_content(value):
                matches.append(
                    {
                        "path": path,
                        "key": str(key),
                        "is_chord_key": bool(is_chord_key(str(key))),
                        "value_type": type(value).__name__,
                        "chord_symbol_hits": int(count_chord_symbols(json.dumps(value, ensure_ascii=True))),
                        "preview": compact_preview(value),
                    }
                )
            matches.extend(find_chord_fields(value, path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload[:50]):
            matches.extend(find_chord_fields(value, f"{prefix}[{index}]"))
    return matches


def compact_preview(value: Any, max_len: int = 160) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    return text[:max_len]


def iter_limited(paths: Iterable[Path], limit: int | None) -> list[Path]:
    result: list[Path] = []
    for path in paths:
        if limit is not None and len(result) >= int(limit):
            break
        result.append(path)
    return result


def iter_role_meta_files(roots: Sequence[Path], limit: int | None = None) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("meta.json")))
    return iter_limited(sorted(set(files)), limit)


def audit_role_meta_files(paths: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_midi_counter: Counter[str] = Counter()
    for path in paths:
        row: dict[str, Any] = {"path": str(path), "readable": False, "chord_field_count": 0}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        matches = find_chord_fields(payload)
        source_midi = str(payload.get("source_midi") or "")
        if source_midi:
            source_midi_counter[source_midi] += 1
        row.update(
            {
                "readable": True,
                "source_midi": source_midi or None,
                "sequence_format": payload.get("sequence_format"),
                "sample_id": payload.get("sample_id"),
                "chord_field_count": len(matches),
                "chord_fields": matches[:8],
            }
        )
        rows.append(row)
    with_chords = [row for row in rows if int(row.get("chord_field_count", 0) or 0) > 0]
    return {
        "scanned_file_count": int(len(rows)),
        "readable_file_count": int(sum(1 for row in rows if row.get("readable"))),
        "with_chord_field_count": int(len(with_chords)),
        "with_chord_field_ratio": float(len(with_chords) / len(rows)) if rows else 0.0,
        "unique_source_midi_count": int(len(source_midi_counter)),
        "top_source_midi": dict(source_midi_counter.most_common(20)),
        "matches_head": with_chords[:20],
    }


def iter_sidecar_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    if not input_dir.exists():
        return []
    files = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SIDECAR_SUFFIXES
    ]
    return iter_limited(files, limit)


def read_mxl_text(path: Path, max_chars: int = 200_000) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith((".xml", ".musicxml")) and "meta-inf/" not in name.lower()
        ]
        if not names:
            names = [name for name in archive.namelist() if name.lower().endswith((".xml", ".musicxml"))]
        if not names:
            return ""
        data = archive.read(sorted(names)[0])[:max_chars]
    return data.decode("utf-8", errors="ignore")


def read_text_lossy(path: Path, max_chars: int = 200_000) -> str:
    if path.suffix.lower() == ".mxl":
        return read_mxl_text(path, max_chars=max_chars)
    data = path.read_bytes()[:max_chars]
    return data.decode("utf-8", errors="ignore")


def audit_sidecar_files(paths: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    suffix_counts: Counter[str] = Counter()
    for path in paths:
        suffix_counts[path.suffix.lower()] += 1
        row: dict[str, Any] = {"path": str(path), "suffix": path.suffix.lower(), "readable": False}
        try:
            text = read_text_lossy(path)
        except Exception as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        chord_hits = count_chord_symbols(text)
        keyword_hits = sum(text.lower().count(keyword) for keyword in CHORD_FIELD_KEYWORDS)
        row.update(
            {
                "readable": True,
                "size_bytes": int(path.stat().st_size),
                "chord_symbol_hits": int(chord_hits),
                "chord_keyword_hits": int(keyword_hits),
                "candidate": bool(chord_hits >= 4 or keyword_hits > 0),
                "preview": compact_preview(text[:240]),
            }
        )
        rows.append(row)
    candidates = [row for row in rows if row.get("candidate")]
    return {
        "scanned_file_count": int(len(rows)),
        "readable_file_count": int(sum(1 for row in rows if row.get("readable"))),
        "candidate_file_count": int(len(candidates)),
        "candidate_file_ratio": float(len(candidates) / len(rows)) if rows else 0.0,
        "suffix_counts": {str(key): int(value) for key, value in suffix_counts.most_common()},
        "matches_head": candidates[:30],
    }


def iter_midi_files(input_dir: Path, limit: int | None = None) -> list[Path]:
    if not input_dir.exists():
        return []
    files = sorted([*input_dir.rglob("*.mid"), *input_dir.rglob("*.midi")])
    return iter_limited(files, limit)


def midi_text_events(path: Path) -> list[str]:
    midi = pretty_midi.PrettyMIDI(str(path))
    texts: list[str] = []
    texts.extend(str(lyric.text) for lyric in getattr(midi, "lyrics", []))
    texts.extend(str(text.text) for text in getattr(midi, "text_events", []))
    return [text for text in texts if text.strip()]


def audit_midi_text_events(paths: Sequence[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {"path": str(path), "readable": False}
        try:
            texts = midi_text_events(path)
        except Exception as exc:
            row["error"] = str(exc)
            rows.append(row)
            continue
        joined = " ".join(texts)
        chord_hits = count_chord_symbols(joined)
        row.update(
            {
                "readable": True,
                "text_event_count": int(len(texts)),
                "chord_symbol_hits": int(chord_hits),
                "candidate": bool(chord_hits >= 4),
                "texts_head": texts[:12],
            }
        )
        rows.append(row)
    with_text = [row for row in rows if int(row.get("text_event_count", 0) or 0) > 0]
    candidates = [row for row in rows if row.get("candidate")]
    return {
        "scanned_file_count": int(len(rows)),
        "readable_file_count": int(sum(1 for row in rows if row.get("readable"))),
        "total_text_event_count": int(sum(int(row.get("text_event_count", 0) or 0) for row in rows)),
        "with_text_event_count": int(len(with_text)),
        "with_text_event_ratio": float(len(with_text) / len(rows)) if rows else 0.0,
        "candidate_file_count": int(len(candidates)),
        "candidate_file_ratio": float(len(candidates) / len(rows)) if rows else 0.0,
        "matches_head": candidates[:30],
        "text_events_head": with_text[:20],
    }


def build_decision(summary: dict[str, Any]) -> dict[str, Any]:
    role_meta_hits = int(summary["role_meta"]["with_chord_field_count"])
    sidecar_hits = int(summary["sidecars"]["candidate_file_count"])
    midi_text_hits = int(summary["midi_text_events"]["candidate_file_count"])
    has_usable_annotation = role_meta_hits > 0 or sidecar_hits > 0 or midi_text_hits > 0
    if role_meta_hits > 0:
        next_step = "build_chord_annotated_role_subset_from_meta"
    elif sidecar_hits > 0 or midi_text_hits > 0:
        next_step = "inspect_chord_sidecars_or_midi_text_events"
    else:
        next_step = "create_chord_inference_or_lead_sheet_alignment_issue"
    return {
        "has_usable_chord_annotation_candidate": bool(has_usable_annotation),
        "role_meta_chord_hits": int(role_meta_hits),
        "sidecar_chord_candidate_hits": int(sidecar_hits),
        "midi_text_chord_candidate_hits": int(midi_text_hits),
        "next_step": next_step,
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# Chord Progression Coverage Audit",
        "",
        f"- input dir: `{report['input_dir']}`",
        f"- role meta roots: `{', '.join(report['role_meta_roots'])}`",
        f"- usable chord annotation candidate: `{str(decision['has_usable_chord_annotation_candidate']).lower()}`",
        f"- next step: `{decision['next_step']}`",
        "",
        "## Summary",
        "",
        "| source | scanned | hits | ratio |",
        "|---|---:|---:|---:|",
        "| role meta | {scanned_file_count} | {with_chord_field_count} | {with_chord_field_ratio:.3f} |".format(
            **report["role_meta"]
        ),
        "| sidecars | {scanned_file_count} | {candidate_file_count} | {candidate_file_ratio:.3f} |".format(
            **report["sidecars"]
        ),
        "| MIDI files scanned for text events | {scanned_file_count} | {candidate_file_count} | {candidate_file_ratio:.3f} |".format(
            **report["midi_text_events"]
        ),
        "",
        "## Role Meta",
        "",
        f"- unique source MIDI count: `{report['role_meta']['unique_source_midi_count']}`",
        f"- with chord fields: `{report['role_meta']['with_chord_field_count']}`",
        "",
        "## Sidecars",
        "",
        f"- suffix counts: `{report['sidecars']['suffix_counts']}`",
        f"- chord candidates: `{report['sidecars']['candidate_file_count']}`",
        "",
        "## MIDI Text Events",
        "",
        f"- scanned MIDI files: `{report['midi_text_events']['scanned_file_count']}`",
        f"- total text event occurrences: `{report['midi_text_events']['total_text_event_count']}`",
        f"- files with any text event: `{report['midi_text_events']['with_text_event_count']}`",
        f"- chord candidates: `{report['midi_text_events']['candidate_file_count']}`",
        "",
        "## Decision",
        "",
        f"`{decision['next_step']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def parse_csv_paths(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit chord progression annotation coverage")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset")
    parser.add_argument("--role_meta_roots", type=str, default="./data,./outputs")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "chord_coverage_audit"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_role_meta_files", type=int, default=None)
    parser.add_argument("--max_sidecar_files", type=int, default=None)
    parser.add_argument("--max_midi_files", type=int, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    input_dir = Path(args.input_dir)
    role_meta_roots = parse_csv_paths(args.role_meta_roots)

    role_meta = audit_role_meta_files(iter_role_meta_files(role_meta_roots, args.max_role_meta_files))
    sidecars = audit_sidecar_files(iter_sidecar_files(input_dir, args.max_sidecar_files))
    midi_text_events = audit_midi_text_events(iter_midi_files(input_dir, args.max_midi_files))
    report = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "input_dir": str(input_dir),
        "role_meta_roots": [str(path) for path in role_meta_roots],
        "max_role_meta_files": args.max_role_meta_files,
        "max_sidecar_files": args.max_sidecar_files,
        "max_midi_files": args.max_midi_files,
        "role_meta": role_meta,
        "sidecars": sidecars,
        "midi_text_events": midi_text_events,
    }
    report["decision"] = build_decision(report)
    write_json(run_dir / "chord_coverage_audit.json", report)
    (run_dir / "chord_coverage_audit.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
