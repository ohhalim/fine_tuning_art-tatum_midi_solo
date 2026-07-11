"""Build a focused Stage B review package from structured review notes."""

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


SCHEMA_VERSION = "stage_b_focused_review_package_v1"


class FocusedReviewPackageError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def objective_candidates_by_id(objective_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not objective_report:
        return {}
    candidates = objective_report.get("candidates")
    if not isinstance(candidates, list):
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("candidate_id"):
            indexed[str(candidate["candidate_id"])] = candidate
    return indexed


def maybe_copy(source: str, target_dir: Path, *, enabled: bool) -> str:
    if not source:
        return ""
    source_path = Path(source)
    if not enabled or not source_path.exists():
        return str(source_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return str(target_path)


def selected_review_candidates(review_notes: dict[str, Any], *, decision: str) -> list[dict[str, Any]]:
    candidates = review_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FocusedReviewPackageError("review notes must contain non-empty candidates")
    selected = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise FocusedReviewPackageError("review note candidate must be an object")
        listening = candidate.get("listening")
        if not isinstance(listening, dict):
            raise FocusedReviewPackageError("candidate listening review must be an object")
        if str(decision) == "all" or str(listening.get("decision") or "") == decision:
            selected.append(candidate)
    selected.sort(
        key=lambda item: (
            str(item.get("review_metadata", {}).get("mode") or ""),
            int(item.get("review_metadata", {}).get("review_rank", 0) or 0),
            str(item.get("candidate_id") or ""),
        )
    )
    return selected


def compact_candidate(
    candidate: dict[str, Any],
    *,
    output_dir: Path,
    copy_files: bool,
    objective_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
    objective = objective_by_id.get(candidate_id, {})
    return {
        "candidate_id": candidate_id,
        "review_metadata": dict(candidate.get("review_metadata") or {}),
        "review_files": {
            "midi_path": maybe_copy(str(files.get("midi_path") or ""), output_dir / "midi", enabled=copy_files),
            "context_midi_path": maybe_copy(
                str(files.get("context_midi_path") or ""),
                output_dir / "context_midi",
                enabled=copy_files,
            ),
            "source_midi_path": str(files.get("source_midi_path") or ""),
        },
        "source_metrics": dict(candidate.get("source_metrics") or {}),
        "listening": dict(candidate.get("listening") or {}),
        "objective_review": dict(candidate.get("objective_review") or {}),
        "objective_first_16_notes": list(objective.get("first_16_notes") or []),
    }


def build_focused_review_package(
    review_notes: dict[str, Any],
    *,
    output_dir: Path,
    decision: str = "keep",
    copy_files: bool = False,
    objective_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = selected_review_candidates(review_notes, decision=decision)
    objective_by_id = objective_candidates_by_id(objective_report)
    candidates = [
        compact_candidate(
            candidate,
            output_dir=output_dir,
            copy_files=copy_files,
            objective_by_id=objective_by_id,
        )
        for candidate in selected
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_review_notes": str(review_notes.get("review_notes_path") or ""),
        "source_review_manifest": str(review_notes.get("source_review_manifest") or ""),
        "source_objective_midi_review_report": str(review_notes.get("source_objective_midi_review_report") or ""),
        "decision_filter": str(decision),
        "copy_files": bool(copy_files),
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
    }


def markdown_report(package: dict[str, Any]) -> str:
    lines = [
        "# Stage B Focused Review Package",
        "",
        f"- decision filter: `{package['decision_filter']}`",
        f"- candidate count: `{package['candidate_count']}`",
        f"- copy files: `{package['copy_files']}`",
        "",
        "This package is for focused context review. It is not a final musical-quality claim.",
        "",
        "| candidate | decision | phrase | timing | chord fit | notes | unique | tension | flags | MIDI | context |",
        "|---|---|---|---|---|---:|---:|---:|---|---|---|",
    ]
    if not package["candidates"]:
        lines.append("| none | - | - | - | - | 0 | 0 | 0.000 | - | - | - |")
    for candidate in package["candidates"]:
        listening = candidate["listening"]
        source_metrics = candidate["source_metrics"]
        objective_flags = candidate.get("objective_review", {}).get("objective_flags", [])
        files = candidate["review_files"]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["candidate_id"],
                    str(listening.get("decision") or ""),
                    str(listening.get("phrase_quality") or ""),
                    str(listening.get("timing") or ""),
                    str(listening.get("chord_fit") or ""),
                    str(int(source_metrics.get("note_count", 0) or 0)),
                    str(int(source_metrics.get("unique_pitch_count", 0) or 0)),
                    f"{float(source_metrics.get('tension_ratio', 0.0) or 0.0):.3f}",
                    ",".join(objective_flags) if objective_flags else "ok",
                    f"`{files.get('midi_path', '')}`",
                    f"`{files.get('context_midi_path', '')}`",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build focused Stage B review package")
    parser.add_argument("--review_notes", type=str, required=True)
    parser.add_argument("--objective_report", type=str, default="")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_focused_review_package"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--decision", type=str, default="keep")
    parser.add_argument("--copy_files", action="store_true")
    parser.add_argument("--min_candidates", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    review_notes_path = Path(args.review_notes)
    review_notes = read_json(review_notes_path)
    review_notes["review_notes_path"] = str(review_notes_path)
    objective_report = read_json(Path(args.objective_report)) if args.objective_report else None
    package = build_focused_review_package(
        review_notes,
        output_dir=output_dir,
        decision=args.decision,
        copy_files=bool(args.copy_files),
        objective_report=objective_report,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "focused_review_package.json", package)
    (output_dir / "focused_review_package.md").write_text(markdown_report(package), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": package["candidate_count"],
                "package_path": str(output_dir / "focused_review_package.json"),
                "markdown_path": str(output_dir / "focused_review_package.md"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if int(package["candidate_count"]) >= int(args.min_candidates) else 3


if __name__ == "__main__":
    raise SystemExit(main())
