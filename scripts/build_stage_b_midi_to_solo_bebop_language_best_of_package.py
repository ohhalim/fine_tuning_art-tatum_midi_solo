"""Build a best-of bebop-language MIDI/WAV package from prior packages."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_midi_to_solo_bebop_language_package import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    BebopLanguagePackageError,
    RenderConfig,
    add_context,
    build_listen_first_package,
    candidate_gate_penalty,
    candidate_score,
    markdown_report,
    objective_metrics,
    read_json,
    render_wav,
    sha256_file,
    validate_report,
    write_json,
    write_text,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import resolve_soundfont  # noqa: E402


DEFAULT_PACKAGE_GLOBS = ("manual_2026_06_13_bebop_language_*/bebop_language_package.json",)


def parse_globs(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()] or list(DEFAULT_PACKAGE_GLOBS)


def package_paths(source_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(source_root.glob(pattern))
    return sorted({path.resolve() for path in paths if path.exists()})


def candidate_rows(
    *,
    paths: list[Path],
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for package_path in paths:
        report = read_json(package_path)
        generation = report.get("generation", {})
        bars = int(generation.get("bars") or 8)
        bpm = float(generation.get("bpm") or 124.0)
        for item in report.get("selected_candidates", []):
            midi_path = Path(str(item.get("midi_path") or item.get("raw_midi_path") or ""))
            if not midi_path.exists():
                continue
            midi_sha = sha256_file(midi_path)
            if midi_sha in seen_hashes:
                continue
            seen_hashes.add(midi_sha)
            chords = list(item.get("chords") or [])
            if not chords:
                continue
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            metrics = objective_metrics(pm, chords, bars=bars, bpm=bpm)
            score = candidate_score(
                metrics,
                target_chord_tone_ratio=target_chord_tone_ratio,
                target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            )
            gate_penalty = candidate_gate_penalty(metrics)
            rows.append(
                {
                    "source_run_id": package_path.parent.name,
                    "source_package": str(package_path),
                    "source_midi_path": str(midi_path),
                    "source_midi_sha256": midi_sha,
                    "case_label": str(item.get("case_label") or "unknown_case"),
                    "chords": chords,
                    "variant_index": int(item.get("variant_index") or 0),
                    "seed": int(item.get("seed") or 0),
                    "generation_meta": dict(item.get("generation_meta") or {}),
                    "objective_metrics": metrics,
                    "score": float(score),
                    "gate_penalty": float(gate_penalty),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            float(row["gate_penalty"]),
            str(row["case_label"]),
            str(row["source_run_id"]),
            int(row["variant_index"]),
        ),
    )


def filter_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    max_gate_penalty: float | None,
    max_offbeat_non_chord_ratio: float | None,
    max_unresolved_offbeat_non_chord_ratio: float | None,
    max_dominant_altered_offbeat_ratio: float | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["objective_metrics"]
        if max_gate_penalty is not None and float(row["gate_penalty"]) > float(max_gate_penalty):
            continue
        if (
            max_offbeat_non_chord_ratio is not None
            and float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio)
        ):
            continue
        if (
            max_unresolved_offbeat_non_chord_ratio is not None
            and float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(max_unresolved_offbeat_non_chord_ratio)
        ):
            continue
        if (
            max_dominant_altered_offbeat_ratio is not None
            and float(metrics["dominant_altered_offbeat_ratio"]) > float(max_dominant_altered_offbeat_ratio)
        ):
            continue
        filtered.append(row)
    return filtered


def select_candidates(rows: list[dict[str, Any]], *, selected_count: int, max_per_case: int) -> list[dict[str, Any]]:
    if not rows:
        raise BebopLanguagePackageError("no candidate rows for best-of package")
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    case_counts: dict[str, int] = defaultdict(int)
    cases = sorted({str(row["case_label"]) for row in rows})
    for case_label in cases:
        case_best = next(row for row in rows if str(row["case_label"]) == case_label)
        selected.append(case_best)
        selected_hashes.add(str(case_best["source_midi_sha256"]))
        case_counts[case_label] += 1
    for row in rows:
        if len(selected) >= int(selected_count):
            break
        midi_sha = str(row["source_midi_sha256"])
        case_label = str(row["case_label"])
        if midi_sha in selected_hashes:
            continue
        if case_counts[case_label] >= int(max_per_case):
            continue
        selected.append(row)
        selected_hashes.add(midi_sha)
        case_counts[case_label] += 1
    if len(selected) < int(selected_count):
        raise BebopLanguagePackageError("not enough selected candidates after max-per-case filter")
    return selected[: int(selected_count)]


def aggregate_metrics(*, generated_count: int, rendered: list[dict[str, Any]], listen_first: dict[str, Any]) -> dict[str, Any]:
    metrics = [item["objective_metrics"] for item in rendered]

    def avg(key: str) -> float:
        return mean([float(item[key]) for item in metrics]) if metrics else 0.0

    return {
        "generated_candidate_count": int(generated_count),
        "selected_candidate_count": len(rendered),
        "listen_first_case_count": int(listen_first["case_count"]),
        "avg_score": mean([float(item["score"]) for item in rendered]) if rendered else 0.0,
        "avg_gate_penalty": mean([float(item.get("gate_penalty") or 0.0) for item in rendered]) if rendered else 0.0,
        "max_gate_penalty": max((float(item.get("gate_penalty") or 0.0) for item in rendered), default=0.0),
        "avg_unique_pitch_count": avg("unique_pitch_count"),
        "avg_step_motion_ratio": avg("step_motion_ratio"),
        "avg_third_fourth_motion_ratio": avg("third_fourth_motion_ratio"),
        "avg_large_leap_ratio": avg("large_leap_ratio"),
        "avg_chord_tone_ratio": avg("chord_tone_ratio"),
        "avg_tension_ratio": avg("tension_ratio"),
        "avg_strong_beat_chord_tone_ratio": avg("strong_beat_chord_tone_ratio"),
        "avg_offbeat_non_chord_ratio": avg("offbeat_non_chord_ratio"),
        "avg_offbeat_non_chord_resolution_ratio": avg("offbeat_non_chord_resolution_ratio"),
        "avg_offbeat_unresolved_non_chord_ratio": avg("offbeat_unresolved_non_chord_ratio"),
        "avg_chromatic_step_ratio": avg("chromatic_step_ratio"),
        "avg_enclosure_proxy_ratio": avg("enclosure_proxy_ratio"),
        "avg_dominant_altered_offbeat_ratio": avg("dominant_altered_offbeat_ratio"),
        "avg_two_note_cycle_ratio": avg("two_note_cycle_ratio"),
        "avg_bar_half_repeat_ratio": avg("bar_half_repeat_ratio"),
        "avg_max_bar_pitch_class_jaccard": avg("max_bar_pitch_class_jaccard"),
        "avg_bar_pitch_shape_repeat_ratio": avg("bar_pitch_shape_repeat_ratio"),
        "min_bar_unique_pitch_count_min": min((int(item["min_bar_unique_pitch_count"]) for item in metrics), default=0),
        "all_final_landings_chord_tone": all(bool(item["final_landing_is_chord_tone"]) for item in metrics),
    }


def listen_first_consonance_score(item: dict[str, Any]) -> float:
    metrics = item["objective_metrics"]
    return (
        float(item.get("gate_penalty") or 0.0) * 3.0
        + float(metrics["offbeat_unresolved_non_chord_ratio"]) * 6.0
        + abs(float(metrics["offbeat_non_chord_ratio"]) - 0.40625) * 1.4
        + max(0.0, float(metrics["dominant_altered_offbeat_ratio"]) - 0.1875) * 2.0
        + max(0.0, 0.10 - float(metrics["dominant_altered_offbeat_ratio"])) * 0.5
        + max(0.0, float(metrics["max_bar_pitch_class_jaccard"]) - 0.72) * 1.2
        + max(0.0, 14.0 - float(metrics["unique_pitch_count"])) * 0.04
        + float(metrics["two_note_cycle_ratio"]) * 1.0
        + float(item["score"]) * 0.12
    )


def order_rendered_for_listen_first(rendered: list[dict[str, Any]], *, listen_first_mode: str) -> list[dict[str, Any]]:
    if listen_first_mode == "rank":
        return rendered
    if listen_first_mode != "consonance":
        raise BebopLanguagePackageError(f"unknown listen-first mode: {listen_first_mode}")

    best_by_case: dict[str, dict[str, Any]] = {}
    for item in rendered:
        case_label = str(item["case_label"])
        current = best_by_case.get(case_label)
        if current is None or listen_first_consonance_score(item) < listen_first_consonance_score(current):
            best_by_case[case_label] = item

    priority_hashes = {str(item["midi_sha256"]) for item in best_by_case.values()}
    priority_rows = sorted(best_by_case.values(), key=lambda item: str(item["case_label"]))
    remaining = [item for item in rendered if str(item["midi_sha256"]) not in priority_hashes]
    return priority_rows + remaining


def build_best_of_package(
    *,
    output_dir: Path,
    source_root: Path,
    package_globs: list[str],
    render_config: RenderConfig,
    bars: int,
    bpm: float,
    selected_count: int,
    max_per_case: int,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_gate_penalty: float | None = None,
    max_offbeat_non_chord_ratio: float | None = None,
    max_unresolved_offbeat_non_chord_ratio: float | None = None,
    max_dominant_altered_offbeat_ratio: float | None = None,
    listen_first_mode: str = "rank",
) -> dict[str, Any]:
    paths = package_paths(source_root, package_globs)
    rows = candidate_rows(
        paths=paths,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    selection_rows = filter_candidate_rows(
        rows,
        max_gate_penalty=max_gate_penalty,
        max_offbeat_non_chord_ratio=max_offbeat_non_chord_ratio,
        max_unresolved_offbeat_non_chord_ratio=max_unresolved_offbeat_non_chord_ratio,
        max_dominant_altered_offbeat_ratio=max_dominant_altered_offbeat_ratio,
    )
    if not selection_rows:
        raise BebopLanguagePackageError("no candidate rows after best-of selection filters")
    selected = select_candidates(selection_rows, selected_count=selected_count, max_per_case=max_per_case)
    solo_dir = output_dir / "midi"
    mix_midi_dir = output_dir / "midi_with_context"
    solo_audio_dir = output_dir / "audio"
    mix_audio_dir = output_dir / "audio_with_context"
    rendered: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        source_midi = Path(str(item["source_midi_path"]))
        pm = pretty_midi.PrettyMIDI(str(source_midi))
        context_pm = add_context(pm, item["chords"], bars=bars, bpm=bpm)
        safe_case = str(item["case_label"]).replace("/", "_").replace(" ", "_")
        stem = f"candidate_{rank:02d}_{safe_case}_variant_{int(item['variant_index']):02d}_best_of"
        solo_midi_path = solo_dir / f"{stem}.mid"
        mix_midi_path = mix_midi_dir / f"{stem}_with_context.mid"
        solo_midi_path.parent.mkdir(parents=True, exist_ok=True)
        mix_midi_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_midi, solo_midi_path)
        context_pm.write(str(mix_midi_path))
        rendered.append(
            {
                **item,
                "rank": int(rank),
                "midi_path": str(solo_midi_path),
                "midi_sha256": sha256_file(solo_midi_path),
                "context_midi_path": str(mix_midi_path),
                "context_midi_sha256": sha256_file(mix_midi_path),
                "solo_audio": render_wav(render_config, solo_midi_path, solo_audio_dir / f"{stem}.wav"),
                "context_audio": render_wav(render_config, mix_midi_path, mix_audio_dir / f"{stem}_with_context.wav"),
            }
        )
    listen_first_rendered = order_rendered_for_listen_first(rendered, listen_first_mode=listen_first_mode)
    listen_first = build_listen_first_package(output_dir, listen_first_rendered)
    report = {
        "schema_version": "stage_b_midi_to_solo_bebop_language_best_of_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_root": str(source_root),
        "source_package_globs": package_globs,
        "source_package_count": len(paths),
        "boundary": "stage_b_midi_to_solo_bebop_language_best_of_package",
        "generation": {
            "bars": int(bars),
            "bpm": float(bpm),
            "selected_count": int(selected_count),
            "max_per_case": int(max_per_case),
            "candidate_pool_count": len(rows),
            "target_chord_tone_ratio": float(target_chord_tone_ratio),
            "target_offbeat_non_chord_ratio": float(target_offbeat_non_chord_ratio),
            "selection_pool_count": len(selection_rows),
            "max_gate_penalty": max_gate_penalty,
            "max_offbeat_non_chord_ratio": max_offbeat_non_chord_ratio,
            "max_unresolved_offbeat_non_chord_ratio": max_unresolved_offbeat_non_chord_ratio,
            "max_dominant_altered_offbeat_ratio": max_dominant_altered_offbeat_ratio,
            "listen_first_mode": listen_first_mode,
        },
        "renderer": render_config.renderer,
        "soundfont": render_config.soundfont,
        "aggregate": aggregate_metrics(generated_count=len(rows), rendered=rendered, listen_first=listen_first),
        "listen_first": listen_first,
        "selected_candidates": rendered,
        "quality_claimed": False,
        "model_direct_claimed": False,
        "not_proven": [
            "human_audio_preference",
            "musical_quality",
            "model_direct_quality",
        ],
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_bebop_language_best_of_package",
            "next_boundary": "listening_review_input",
            "critical_user_input_required": False,
        },
    }
    write_json(output_dir / "bebop_language_best_of_package.json", report)
    write_json(output_dir / "bebop_language_best_of_package_summary.json", validate_report(report, render_config.sample_rate))
    write_text(output_dir / "bebop_language_best_of_package.md", markdown_report(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a best-of bebop-language MIDI/WAV review package")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--package_globs", default=",".join(DEFAULT_PACKAGE_GLOBS))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT / "best_of"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--renderer", default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument("--selected_count", type=int, default=16)
    parser.add_argument("--max_per_case", type=int, default=4)
    parser.add_argument("--target_chord_tone_ratio", type=float, default=0.78)
    parser.add_argument("--target_offbeat_non_chord_ratio", type=float, default=0.38)
    parser.add_argument("--max_gate_penalty", type=float, default=None)
    parser.add_argument("--max_offbeat_non_chord_ratio", type=float, default=None)
    parser.add_argument("--max_unresolved_offbeat_non_chord_ratio", type=float, default=None)
    parser.add_argument("--max_dominant_altered_offbeat_ratio", type=float, default=None)
    parser.add_argument("--listen_first_mode", choices=["rank", "consonance"], default="rank")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    renderer = str(args.renderer or "")
    if not renderer:
        raise BebopLanguagePackageError("fluidsynth renderer not found")
    soundfont = resolve_soundfont(str(args.soundfont or ""))
    if not soundfont:
        raise BebopLanguagePackageError("soundfont not found")
    run_id = str(args.run_id or datetime.now(timezone.utc).strftime("manual_%Y_%m_%d_bebop_language_best_of_%H%M%S"))
    output_dir = Path(args.output_root) / run_id
    report = build_best_of_package(
        output_dir=output_dir,
        source_root=Path(args.source_root),
        package_globs=parse_globs(str(args.package_globs)),
        render_config=RenderConfig(renderer=renderer, soundfont=soundfont, sample_rate=int(args.sample_rate)),
        bars=int(args.bars),
        bpm=float(args.bpm),
        selected_count=int(args.selected_count),
        max_per_case=int(args.max_per_case),
        target_chord_tone_ratio=float(args.target_chord_tone_ratio),
        target_offbeat_non_chord_ratio=float(args.target_offbeat_non_chord_ratio),
        max_gate_penalty=args.max_gate_penalty,
        max_offbeat_non_chord_ratio=args.max_offbeat_non_chord_ratio,
        max_unresolved_offbeat_non_chord_ratio=args.max_unresolved_offbeat_non_chord_ratio,
        max_dominant_altered_offbeat_ratio=args.max_dominant_altered_offbeat_ratio,
        listen_first_mode=str(args.listen_first_mode),
    )
    print(json.dumps(validate_report(report, int(args.sample_rate)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
