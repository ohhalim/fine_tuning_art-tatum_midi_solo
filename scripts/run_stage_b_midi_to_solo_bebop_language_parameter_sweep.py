"""Run a small parameter sweep for bebop-language MIDI/WAV packages."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_midi_to_solo_bebop_language_package import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SOURCE_PACKAGE,
    BebopLanguagePackageError,
    RenderConfig,
    build_package,
    validate_report,
    write_json,
    write_text,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import resolve_soundfont  # noqa: E402


SCHEMA_VERSION = "stage_b_midi_to_solo_bebop_language_parameter_sweep_v1"


class BebopLanguageParameterSweepError(ValueError):
    pass


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not items:
        raise BebopLanguageParameterSweepError("float list must not be empty")
    return [float(item) for item in items]


def sweep_score(summary: dict[str, Any]) -> float:
    chord_tone = float(summary.get("avg_chord_tone_ratio") or 0.0)
    offbeat = float(summary.get("avg_offbeat_non_chord_ratio") or 0.0)
    resolution = float(summary.get("avg_offbeat_non_chord_resolution_ratio") or 0.0)
    unresolved = float(summary.get("avg_offbeat_unresolved_non_chord_ratio") or 0.0)
    strong = float(summary.get("avg_strong_beat_chord_tone_ratio") or 0.0)
    chromatic = float(summary.get("avg_chromatic_step_ratio") or 0.0)
    enclosure = float(summary.get("avg_enclosure_proxy_ratio") or 0.0)
    altered = float(summary.get("avg_dominant_altered_offbeat_ratio") or 0.0)
    cycle = float(summary.get("avg_two_note_cycle_ratio") or 0.0)
    half_repeat = float(summary.get("avg_bar_half_repeat_ratio") or 0.0)
    min_bar_unique = int(summary.get("min_bar_unique_pitch_count_min") or 0)
    final_ok = bool(summary.get("all_final_landings_chord_tone", False))
    return (
        abs(chord_tone - 0.78) * 2.0
        + abs(offbeat - 0.42) * 1.2
        + max(0.0, 0.86 - resolution) * 2.4
        + unresolved * 5.0
        + max(0.0, 1.0 - strong) * 4.0
        + max(0.0, 0.18 - chromatic) * 1.0
        + max(0.0, 0.04 - enclosure) * 0.8
        + max(0.0, 0.05 - altered) * 0.5
        + cycle * 1.5
        + half_repeat * 1.2
        + max(0, 4 - min_bar_unique) * 0.3
        + (0.0 if final_ok else 3.0)
    )


def config_label(index: int, *, non_chord: float, chord_tone: float, offbeat: float, seed: int) -> str:
    def fmt(value: float) -> str:
        return f"{value:.2f}".replace(".", "p")

    return (
        f"config_{index:02d}_nc{fmt(non_chord)}_ct{fmt(chord_tone)}_"
        f"off{fmt(offbeat)}_seed{int(seed)}"
    )


def copy_best_listen_first(best_package_dir: Path, target_dir: Path) -> dict[str, Any]:
    source_dir = best_package_dir / "listen_first_by_progression"
    if not source_dir.exists():
        raise BebopLanguageParameterSweepError(f"best listen-first dir missing: {source_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    files = sorted(path for path in target_dir.iterdir() if path.is_file())
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "file_count": len(files),
        "files": [str(path) for path in files],
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage B MIDI-to-Solo Bebop Language Parameter Sweep",
        "",
        "## Summary",
        "",
        f"- config count: `{report['aggregate']['config_count']}`",
        f"- best config: `{report['best_config']['config_label']}`",
        f"- best score: `{float(report['best_config']['sweep_score']):.4f}`",
        f"- best chord-tone: `{float(report['best_config']['summary']['avg_chord_tone_ratio']):.4f}`",
        f"- best offbeat non-chord: `{float(report['best_config']['summary']['avg_offbeat_non_chord_ratio']):.4f}`",
        f"- best offbeat resolution: `{float(report['best_config']['summary']['avg_offbeat_non_chord_resolution_ratio']):.4f}`",
        f"- best unresolved offbeat non-chord: `{float(report['best_config']['summary']['avg_offbeat_unresolved_non_chord_ratio']):.4f}`",
        f"- best chromatic step: `{float(report['best_config']['summary']['avg_chromatic_step_ratio']):.4f}`",
        f"- best enclosure proxy: `{float(report['best_config']['summary']['avg_enclosure_proxy_ratio']):.4f}`",
        f"- best dominant altered offbeat: `{float(report['best_config']['summary']['avg_dominant_altered_offbeat_ratio']):.4f}`",
        f"- best two-note cycle: `{float(report['best_config']['summary']['avg_two_note_cycle_ratio']):.4f}`",
        f"- best bar half-repeat: `{float(report['best_config']['summary']['avg_bar_half_repeat_ratio']):.4f}`",
        f"- best min bar unique pitch: `{int(report['best_config']['summary']['min_bar_unique_pitch_count_min'])}`",
        f"- best listen-first dir: `{report['best_listen_first']['target_dir']}`",
        "",
        "## Configs",
        "",
        "| rank | config | score | chord-tone | offbeat non-chord | resolved | unresolved | chromatic | enclosure | altered | cycle | half repeat | min bar unique | package |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(report["ranked_configs"], start=1):
        summary = row["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(row["config_label"]),
                    f"{float(row['sweep_score']):.4f}",
                    f"{float(summary['avg_chord_tone_ratio']):.4f}",
                    f"{float(summary['avg_offbeat_non_chord_ratio']):.4f}",
                    f"{float(summary['avg_offbeat_non_chord_resolution_ratio']):.4f}",
                    f"{float(summary['avg_offbeat_unresolved_non_chord_ratio']):.4f}",
                    f"{float(summary['avg_chromatic_step_ratio']):.4f}",
                    f"{float(summary['avg_enclosure_proxy_ratio']):.4f}",
                    f"{float(summary['avg_dominant_altered_offbeat_ratio']):.4f}",
                    f"{float(summary['avg_two_note_cycle_ratio']):.4f}",
                    f"{float(summary['avg_bar_half_repeat_ratio']):.4f}",
                    str(int(summary["min_bar_unique_pitch_count_min"])),
                    str(row["package_dir"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- quality claimed: `false`",
            "- model direct claimed: `false`",
            "- human listening preference completed: `false`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_sweep(
    *,
    source_package: Path,
    output_dir: Path,
    render_config: RenderConfig,
    bars: int,
    bpm: float,
    variants_per_progression: int,
    selected_count: int,
    seed_base: int,
    non_chord_probabilities: list[float],
    target_chord_tone_ratios: list[float],
    target_offbeat_non_chord_ratios: list[float],
    max_configs: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    config_index = 0
    for non_chord in non_chord_probabilities:
        for chord_tone in target_chord_tone_ratios:
            for offbeat in target_offbeat_non_chord_ratios:
                if config_index >= max_configs:
                    break
                config_index += 1
                config_seed = seed_base + config_index * 10000
                label = config_label(
                    config_index,
                    non_chord=non_chord,
                    chord_tone=chord_tone,
                    offbeat=offbeat,
                    seed=config_seed,
                )
                package_dir = output_dir / label
                package = build_package(
                    source_package_path=source_package,
                    output_dir=package_dir,
                    render_config=render_config,
                    bars=bars,
                    bpm=bpm,
                    variants_per_progression=variants_per_progression,
                    selected_count=selected_count,
                    seed_base=config_seed,
                    non_chord_probability=non_chord,
                    target_chord_tone_ratio=chord_tone,
                    target_offbeat_non_chord_ratio=offbeat,
                )
                summary = validate_report(package, render_config.sample_rate)
                rows.append(
                    {
                        "config_label": label,
                        "package_dir": str(package_dir),
                        "parameters": {
                            "non_chord_probability": float(non_chord),
                            "target_chord_tone_ratio": float(chord_tone),
                            "target_offbeat_non_chord_ratio": float(offbeat),
                            "seed_base": int(config_seed),
                        },
                        "summary": summary,
                        "sweep_score": sweep_score(summary),
                    }
                )
            if config_index >= max_configs:
                break
        if config_index >= max_configs:
            break
    if not rows:
        raise BebopLanguageParameterSweepError("no sweep configs completed")
    rows.sort(key=lambda row: (float(row["sweep_score"]), str(row["config_label"])))
    best = rows[0]
    best_listen_first = copy_best_listen_first(
        Path(str(best["package_dir"])),
        output_dir / "best_listen_first_by_progression",
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": "stage_b_midi_to_solo_bebop_language_parameter_sweep",
        "source_package": str(source_package),
        "aggregate": {
            "config_count": len(rows),
            "bars": int(bars),
            "bpm": float(bpm),
            "variants_per_progression": int(variants_per_progression),
            "selected_count": int(selected_count),
        },
        "best_config": best,
        "best_listen_first": best_listen_first,
        "ranked_configs": rows,
        "quality_claimed": False,
        "model_direct_claimed": False,
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_bebop_language_parameter_sweep",
            "selected_next_target": "listen_to_best_parameter_package",
            "critical_user_input_required": False,
        },
    }
    write_json(output_dir / "bebop_language_parameter_sweep.json", report)
    write_text(output_dir / "bebop_language_parameter_sweep.md", build_markdown(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bebop-language parameter sweep")
    parser.add_argument("--source_package", default=str(DEFAULT_SOURCE_PACKAGE))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT / "parameter_sweep"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--renderer", default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument("--variants_per_progression", type=int, default=48)
    parser.add_argument("--selected_count", type=int, default=16)
    parser.add_argument("--seed_base", type=int, default=200000)
    parser.add_argument("--non_chord_probabilities", default="0.26,0.30,0.34")
    parser.add_argument("--target_chord_tone_ratios", default="0.74,0.76,0.80")
    parser.add_argument("--target_offbeat_non_chord_ratios", default="0.34,0.38,0.42,0.44")
    parser.add_argument("--max_configs", type=int, default=36)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    renderer = str(args.renderer or "")
    if not renderer:
        raise BebopLanguagePackageError("renderer missing")
    if not Path(renderer).exists():
        raise BebopLanguagePackageError(f"renderer not found: {renderer}")
    soundfont = resolve_soundfont(str(args.soundfont or ""))
    if not soundfont or not Path(soundfont).exists():
        raise BebopLanguagePackageError(f"soundfont not found: {soundfont}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = run_sweep(
        source_package=Path(str(args.source_package)),
        output_dir=Path(str(args.output_root)) / run_id,
        render_config=RenderConfig(renderer=renderer, soundfont=soundfont, sample_rate=int(args.sample_rate)),
        bars=int(args.bars),
        bpm=float(args.bpm),
        variants_per_progression=int(args.variants_per_progression),
        selected_count=int(args.selected_count),
        seed_base=int(args.seed_base),
        non_chord_probabilities=parse_float_list(str(args.non_chord_probabilities)),
        target_chord_tone_ratios=parse_float_list(str(args.target_chord_tone_ratios)),
        target_offbeat_non_chord_ratios=parse_float_list(str(args.target_offbeat_non_chord_ratios)),
        max_configs=int(args.max_configs),
    )
    print(json.dumps(report["best_config"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
