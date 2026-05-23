"""Extract objective-clean Stage B review candidates into a compact package."""

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

from scripts.build_listening_review_notes import write_json  # noqa: E402


SCHEMA_VERSION = "stage_b_clean_review_package_v1"


class CleanReviewPackageError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_id_from_review_row(row: dict[str, Any]) -> str:
    mode = str(row.get("mode") or "candidate").strip() or "candidate"
    rank = int(row.get("review_rank", 0) or 0)
    sample_index = int(row.get("sample_index", 0) or 0)
    if rank and sample_index:
        return f"{mode}_rank_{rank}_sample_{sample_index}"
    return str(row.get("sample_id") or row.get("review_midi_path") or mode)


def objective_candidates_by_id(objective_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = objective_report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CleanReviewPackageError("objective report must contain non-empty candidates")
    indexed: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise CleanReviewPackageError("objective candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise CleanReviewPackageError("objective candidate_id is required")
        indexed[candidate_id] = candidate
    return indexed


def selected_clean_candidates(
    review_manifest: dict[str, Any],
    objective_report: dict[str, Any],
    *,
    allowed_modes: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = review_manifest.get("candidates")
    if not isinstance(rows, list) or not rows:
        raise CleanReviewPackageError("review manifest must contain non-empty candidates")
    objective_by_id = objective_candidates_by_id(objective_report)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise CleanReviewPackageError("review manifest candidate must be an object")
        mode = str(row.get("mode") or "")
        if allowed_modes and mode not in allowed_modes:
            continue
        candidate_id = candidate_id_from_review_row(row)
        objective = objective_by_id.get(candidate_id)
        if not objective:
            continue
        if objective.get("objective_bucket") != "clean":
            continue
        if objective.get("objective_flags"):
            continue
        selected.append({"candidate_id": candidate_id, "review": row, "objective": objective})
    selected.sort(
        key=lambda item: (
            -int(item["objective"].get("objective_priority_score", 0) or 0),
            str(item["review"].get("mode") or ""),
            int(item["review"].get("review_rank", 0) or 0),
        )
    )
    return selected


def maybe_copy(source: str, target_dir: Path, *, enabled: bool) -> str:
    if not source:
        return ""
    source_path = Path(source)
    if not enabled:
        return str(source_path)
    if not source_path.exists():
        return str(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return str(target_path)


def compact_candidate(item: dict[str, Any], output_dir: Path, *, copy_files: bool) -> dict[str, Any]:
    review = item["review"]
    objective = item["objective"]
    metrics = objective.get("metrics", {})
    midi_path = maybe_copy(str(review.get("review_midi_path") or ""), output_dir / "midi", enabled=copy_files)
    context_path = maybe_copy(
        str(review.get("context_midi_path") or ""),
        output_dir / "context_midi",
        enabled=copy_files,
    )
    return {
        "candidate_id": item["candidate_id"],
        "mode": str(review.get("mode") or ""),
        "review_rank": int(review.get("review_rank", 0) or 0),
        "sample_index": int(review.get("sample_index", 0) or 0),
        "review_midi_path": midi_path,
        "context_midi_path": context_path,
        "objective_priority_score": int(objective.get("objective_priority_score", 0) or 0),
        "objective_bucket": str(objective.get("objective_bucket") or ""),
        "objective_flags": list(objective.get("objective_flags") or []),
        "metrics": {
            "note_count": int(metrics.get("note_count", 0) or 0),
            "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
            "stepwise_interval_ratio": float(metrics.get("stepwise_interval_ratio", 0.0) or 0.0),
            "chromatic_interval_ratio": float(metrics.get("chromatic_interval_ratio", 0.0) or 0.0),
            "large_leap_interval_ratio": float(metrics.get("large_leap_interval_ratio", 0.0) or 0.0),
            "unresolved_large_leap_ratio": float(metrics.get("unresolved_large_leap_ratio", 0.0) or 0.0),
            "chord_tone_ratio": float(metrics.get("chord_tone_ratio", 0.0) or 0.0),
            "tension_ratio": float(metrics.get("tension_ratio", 0.0) or 0.0),
            "outside_ratio": float(metrics.get("outside_ratio", 0.0) or 0.0),
            "root_tone_ratio": float(metrics.get("root_tone_ratio", 0.0) or 0.0),
        },
    }


def build_clean_review_package(
    review_manifest: dict[str, Any],
    objective_report: dict[str, Any],
    *,
    output_dir: Path,
    allowed_modes: set[str] | None = None,
    copy_files: bool = False,
) -> dict[str, Any]:
    selected = selected_clean_candidates(review_manifest, objective_report, allowed_modes=allowed_modes)
    candidates = [compact_candidate(item, output_dir, copy_files=copy_files) for item in selected]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_review_manifest": str(review_manifest.get("output_dir") or ""),
        "source_objective_report": str(objective_report.get("report_path") or ""),
        "allowed_modes": sorted(allowed_modes) if allowed_modes else [],
        "copy_files": bool(copy_files),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
    }


def markdown_report(package: dict[str, Any]) -> str:
    lines = [
        "# Stage B Objective Clean Review Package",
        "",
        f"- candidate count: `{package['candidate_count']}`",
        f"- allowed modes: `{', '.join(package['allowed_modes']) if package['allowed_modes'] else 'all'}`",
        "",
        "| candidate | mode | score | notes | unique | unresolved leap | chord tone | tension | MIDI | context |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    if not package["candidates"]:
        lines.append("| none | - | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |")
    for candidate in package["candidates"]:
        metrics = candidate["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["candidate_id"],
                    candidate["mode"],
                    str(candidate["objective_priority_score"]),
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    f"{metrics['unresolved_large_leap_ratio']:.3f}",
                    f"{metrics['chord_tone_ratio']:.3f}",
                    f"{metrics['tension_ratio']:.3f}",
                    f"`{candidate['review_midi_path']}`",
                    f"`{candidate['context_midi_path']}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build objective-clean Stage B review package")
    parser.add_argument("--review_manifest", type=str, required=True)
    parser.add_argument("--objective_report", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_clean_review_package"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--allowed_modes", type=str, default="")
    parser.add_argument("--copy_files", action="store_true")
    parser.add_argument("--min_candidates", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    allowed_modes = {mode.strip() for mode in args.allowed_modes.split(",") if mode.strip()} or None
    package = build_clean_review_package(
        read_json(Path(args.review_manifest)),
        read_json(Path(args.objective_report)),
        output_dir=output_dir,
        allowed_modes=allowed_modes,
        copy_files=bool(args.copy_files),
    )
    write_json(output_dir / "clean_review_package.json", package)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_review_package.md").write_text(markdown_report(package), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": package["candidate_count"],
                "package_path": str(output_dir / "clean_review_package.json"),
                "markdown_path": str(output_dir / "clean_review_package.md"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if int(package["candidate_count"]) >= int(args.min_candidates) else 3


if __name__ == "__main__":
    raise SystemExit(main())
