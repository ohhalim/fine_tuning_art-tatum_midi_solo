"""
Evaluate the MVP generation contract across seeds, chords, and densities.

This script measures whether the model-first path can produce gated MIDI after
repair, and how often the pipeline falls back to the rule-based generator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_PATH = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT_PATH))

from inference.app.generator import (
    DEFAULT_CONDITIONING_MIDI,
    DEFAULT_LORA_PATH,
    PROJECT_ROOT,
    generate_midi_phrase,
)
from inference.app.schemas import GenerationMetrics, GenerationRequest


DEFAULT_CHORD_SETS = (
    "Cm7,Fm7,Bb7,Ebmaj7",
    "Dm7,G7,Cmaj7,A7",
    "Am7,D7,Gmaj7,E7",
)


def parse_int_csv(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_str_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_chord_sets(value: str) -> list[list[str]]:
    sets: list[list[str]] = []
    for raw_set in value.split(";"):
        chords = [chord.strip() for chord in raw_set.split(",") if chord.strip()]
        if chords:
            sets.append(chords)
    return sets


def metric_payload(metrics: GenerationMetrics | None) -> dict[str, Any]:
    if metrics is None:
        return {}
    return metrics.to_dict()


def flatten_result(
    result: Any,
    progression_index: int,
    chord_progression: list[str],
    density: str,
    seed: int,
    use_model: bool,
) -> dict[str, Any]:
    metrics = metric_payload(result.metrics)
    return {
        "job_id": result.job_id,
        "progression_index": progression_index,
        "chord_progression": chord_progression,
        "density": density,
        "seed": seed,
        "use_model": use_model,
        "status": result.status,
        "fallback_used": result.fallback_used,
        "model_repaired": result.model_repaired,
        "failure_reason": result.failure_reason,
        "model_failure_reason": result.model_failure_reason,
        "midi_path": result.midi_path,
        "metrics_path": result.metrics_path,
        **metrics,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    completed = [row for row in rows if row["status"] == "COMPLETED"]
    failed = [row for row in rows if row["status"] == "FAILED"]
    model_success = [row for row in completed if not row["fallback_used"]]
    fallback = [row for row in completed if row["fallback_used"]]
    repaired = [row for row in completed if row["model_repaired"]]

    def avg(key: str) -> float | None:
        values = [row[key] for row in completed if isinstance(row.get(key), (int, float))]
        return mean(values) if values else None

    by_density: dict[str, dict[str, Any]] = {}
    for density in sorted({row["density"] for row in rows}):
        density_rows = [row for row in rows if row["density"] == density]
        density_completed = [row for row in density_rows if row["status"] == "COMPLETED"]
        density_model_success = [row for row in density_completed if not row["fallback_used"]]
        by_density[density] = {
            "total": len(density_rows),
            "completed": len(density_completed),
            "model_success": len(density_model_success),
            "model_success_rate": len(density_model_success) / max(1, len(density_rows)),
            "fallback_used": sum(1 for row in density_completed if row["fallback_used"]),
        }

    return {
        "total": total,
        "completed": len(completed),
        "failed": len(failed),
        "model_success": len(model_success),
        "model_success_rate": len(model_success) / max(1, total),
        "fallback_used": len(fallback),
        "fallback_rate": len(fallback) / max(1, total),
        "model_repaired": len(repaired),
        "avg_generation_time_ms": avg("generation_time_ms"),
        "avg_note_count": avg("note_count"),
        "avg_note_density": avg("note_density"),
        "avg_dead_air_ratio": avg("dead_air_ratio"),
        "avg_repetition_score": avg("repetition_score"),
        "by_density": by_density,
    }


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Generation Contract Sweep",
        "",
        "## Summary",
        "",
        f"- total: {summary['total']}",
        f"- completed: {summary['completed']}",
        f"- failed: {summary['failed']}",
        f"- model_success: {summary['model_success']} ({summary['model_success_rate']:.3f})",
        f"- fallback_used: {summary['fallback_used']} ({summary['fallback_rate']:.3f})",
        f"- model_repaired: {summary['model_repaired']}",
        f"- avg_note_density: {summary['avg_note_density']}",
        f"- avg_dead_air_ratio: {summary['avg_dead_air_ratio']}",
        "",
        "## Rows",
        "",
        "| Job | Density | Seed | Status | Model Success | Repaired | Fallback | Notes | Density Metric | Dead-air |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model_success = row["status"] == "COMPLETED" and not row["fallback_used"]
        lines.append(
            "| {job} | {density} | {seed} | {status} | {model_success} | {repaired} | "
            "{fallback} | {notes} | {density_metric} | {dead_air} |".format(
                job=row["job_id"],
                density=row["density"],
                seed=row["seed"],
                status=row["status"],
                model_success=str(model_success).lower(),
                repaired=str(row["model_repaired"]).lower(),
                fallback=str(row["fallback_used"]).lower(),
                notes=row.get("note_count", ""),
                density_metric=(
                    f"{row['note_density']:.3f}" if isinstance(row.get("note_density"), (int, float)) else ""
                ),
                dead_air=(
                    f"{row['dead_air_ratio']:.3f}"
                    if isinstance(row.get("dead_air_ratio"), (int, float))
                    else ""
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a generation contract quality sweep")
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--time_signature", type=str, default="4/4")
    parser.add_argument("--section", type=str, default="drop")
    parser.add_argument("--energy", type=str, default="high")
    parser.add_argument("--densities", type=str, default="medium")
    parser.add_argument("--seeds", type=str, default="11,13,17")
    parser.add_argument(
        "--chord_progressions",
        type=str,
        default=";".join(DEFAULT_CHORD_SETS),
        help="Semicolon-separated progressions; chords inside a progression are comma-separated.",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "generated"))
    parser.add_argument("--summary_json", type=str, default=str(PROJECT_ROOT / "outputs" / "sweeps" / "generation_contract_sweep.json"))
    parser.add_argument("--summary_md", type=str, default=str(PROJECT_ROOT / "outputs" / "sweeps" / "generation_contract_sweep.md"))
    parser.add_argument("--no_model", action="store_true", help="Skip Stage A model and measure fallback only")
    parser.add_argument("--lora_path", type=str, default=str(DEFAULT_LORA_PATH))
    parser.add_argument("--conditioning_midi", type=str, default=str(DEFAULT_CONDITIONING_MIDI))
    parser.add_argument("--primer_max_tokens", type=int, default=64)
    parser.add_argument("--max_sequence", type=int, default=512)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = parse_int_csv(args.seeds)
    densities = parse_str_csv(args.densities)
    chord_sets = parse_chord_sets(args.chord_progressions)

    rows: list[dict[str, Any]] = []
    for progression_index, chord_progression in enumerate(chord_sets):
        for density in densities:
            for seed in seeds:
                mode = "fallback" if args.no_model else "model"
                job_id = f"sweep_{mode}_p{progression_index}_{density}_s{seed}"
                request = GenerationRequest(
                    job_id=job_id,
                    bpm=args.bpm,
                    chord_progression=chord_progression,
                    bars=args.bars,
                    time_signature=args.time_signature,
                    section=args.section,
                    energy=args.energy,
                    density=density,
                    seed=seed,
                )
                result = generate_midi_phrase(
                    request=request,
                    output_dir=args.output_dir,
                    use_model=not args.no_model,
                    lora_path=args.lora_path,
                    conditioning_midi=args.conditioning_midi,
                    primer_max_tokens=args.primer_max_tokens,
                    max_sequence=args.max_sequence,
                )
                row = flatten_result(
                    result=result,
                    progression_index=progression_index,
                    chord_progression=chord_progression,
                    density=density,
                    seed=seed,
                    use_model=not args.no_model,
                )
                rows.append(row)
                print(json.dumps({"job_id": row["job_id"], "status": row["status"], "fallback_used": row["fallback_used"], "model_repaired": row["model_repaired"]}, ensure_ascii=True))

    summary = summarize(rows)
    report = {
        "config": {
            "bpm": args.bpm,
            "bars": args.bars,
            "time_signature": args.time_signature,
            "section": args.section,
            "energy": args.energy,
            "densities": densities,
            "seeds": seeds,
            "chord_progressions": chord_sets,
            "use_model": not args.no_model,
        },
        "summary": summary,
        "rows": rows,
    }

    summary_json = Path(args.summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(report, ensure_ascii=True, indent=2) + "\n")
    write_markdown(Path(args.summary_md), summary, rows)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    print(f"Saved JSON: {summary_json}")
    print(f"Saved Markdown: {args.summary_md}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
