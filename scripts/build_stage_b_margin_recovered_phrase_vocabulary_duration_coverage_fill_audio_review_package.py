"""Build a reviewer-facing package for duration/coverage fill audio review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillAudioReviewPackageError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def file_info(path: str, *, required: bool) -> dict[str, Any]:
    if not path:
        return {
            "path": "",
            "exists": False,
            "required": required,
            "size_bytes": 0,
            "sha256": "",
        }
    file_path = Path(path)
    exists = file_path.exists()
    if required and not exists:
        raise DurationCoverageFillAudioReviewPackageError(f"required file not found: {path}")
    if not exists:
        return {
            "path": path,
            "exists": False,
            "required": required,
            "size_bytes": 0,
            "sha256": "",
        }
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return {
        "path": path,
        "exists": True,
        "required": required,
        "size_bytes": int(file_path.stat().st_size),
        "sha256": digest,
    }


def selected_candidate_id(review_fill_guard: dict[str, Any]) -> str:
    candidate_id = str(review_fill_guard.get("candidate_id") or "")
    if not candidate_id:
        raise DurationCoverageFillAudioReviewPackageError("review fill guard must contain candidate_id")
    return candidate_id


def review_items(boundary: dict[str, Any]) -> list[dict[str, Any]]:
    items = boundary.get("review_items")
    if not isinstance(items, list) or len(items) != 2:
        raise DurationCoverageFillAudioReviewPackageError("boundary must contain two review items")
    return items


def compact_item(item: dict[str, Any]) -> dict[str, Any]:
    role = str(item.get("role") or "")
    midi_required = role in {"source_constrained_partial", "duration_coverage_fill_keep"}
    context_required = role == "duration_coverage_fill_keep"
    return {
        "role": role,
        "candidate_id": str(item.get("candidate_id") or ""),
        "prior_decision": str(item.get("prior_decision") or ""),
        "metric_summary": dict(item.get("metric_summary") or {}),
        "midi_file": file_info(str(item.get("midi_path") or ""), required=midi_required),
        "context_midi_file": file_info(str(item.get("context_midi_path") or ""), required=context_required),
        "note_signature_count": int(item.get("note_signature_count", 0) or 0),
    }


def review_input_template(candidate_id: str) -> dict[str, Any]:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_v1",
        "candidate_id": candidate_id,
        "reviewer": "",
        "audio_render_used": False,
        "preference": "",
        "timing": "",
        "phrase": "",
        "vocabulary": "",
        "notes": "",
        "allowed_values": {
            "preference": [
                "source_constrained_partial",
                "duration_coverage_fill_keep",
                "tie",
                "reject_both",
            ],
            "timing_phrase_vocabulary": [
                "source_constrained_partial",
                "duration_coverage_fill_keep",
                "tie",
                "unclear",
            ],
        },
    }


def build_audio_review_package(
    human_audio_boundary: dict[str, Any],
    review_fill_guard: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_id = selected_candidate_id(review_fill_guard)
    fill_status = str(review_fill_guard.get("fill_status") or "")
    claim = review_fill_guard.get("claim_boundary") if isinstance(review_fill_guard.get("claim_boundary"), dict) else {}
    if fill_status != "pending_review_input":
        raise DurationCoverageFillAudioReviewPackageError(f"expected pending_review_input, got {fill_status}")
    if bool(claim.get("preference_claimed", True)):
        raise DurationCoverageFillAudioReviewPackageError("preference must not be claimed before package build")
    items = [compact_item(item) for item in review_items(human_audio_boundary)]
    selected_items = [item for item in items if item["candidate_id"] == candidate_id]
    if len(selected_items) != 1:
        raise DurationCoverageFillAudioReviewPackageError(f"candidate not found in review items: {candidate_id}")
    template = review_input_template(candidate_id)
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_human_audio_boundary_schema": str(human_audio_boundary.get("schema_version") or ""),
        "source_review_fill_schema": str(review_fill_guard.get("schema_version") or ""),
        "candidate_id": candidate_id,
        "review_items": items,
        "review_input_template": template,
        "package_boundary": {
            "status": "ready_for_external_review_input",
            "audio_render_status": "not_rendered_by_harness",
            "preference_claimed": False,
            "requires_external_human_or_audio_review": True,
        },
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill external review input fill",
    }


def validate_audio_review_package(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    require_files_exist: bool,
    require_no_preference: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillAudioReviewPackageError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    items = report.get("review_items")
    if not isinstance(items, list) or len(items) != 2:
        raise DurationCoverageFillAudioReviewPackageError("report must contain two review items")
    if require_files_exist:
        missing = []
        for item in items:
            for key in ("midi_file", "context_midi_file"):
                file_meta = item.get(key) if isinstance(item.get(key), dict) else {}
                if bool(file_meta.get("required", False)) and not bool(file_meta.get("exists", False)):
                    missing.append(str(file_meta.get("path") or key))
        if missing:
            raise DurationCoverageFillAudioReviewPackageError(f"missing required files: {missing}")
    boundary = report.get("package_boundary") if isinstance(report.get("package_boundary"), dict) else {}
    if require_no_preference and bool(boundary.get("preference_claimed", True)):
        raise DurationCoverageFillAudioReviewPackageError("preference must not be claimed")
    template = report.get("review_input_template") if isinstance(report.get("review_input_template"), dict) else {}
    if str(template.get("candidate_id") or "") != candidate_id:
        raise DurationCoverageFillAudioReviewPackageError("review input template candidate mismatch")
    return {
        "candidate_id": candidate_id,
        "review_item_count": len(items),
        "package_status": str(boundary.get("status") or ""),
        "audio_render_status": str(boundary.get("audio_render_status") or ""),
        "preference_claimed": bool(boundary.get("preference_claimed", True)),
        "required_file_count": sum(
            1
            for item in items
            for key in ("midi_file", "context_midi_file")
            if bool(item.get(key, {}).get("required", False))
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["package_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Audio Review Package",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- package status: `{boundary['status']}`",
        f"- audio render status: `{boundary['audio_render_status']}`",
        f"- preference claimed: `{boundary['preference_claimed']}`",
        "",
        "| role | candidate | MIDI exists | context exists | notes | focused notes | dead-air | sha256 |",
        "|---|---|:---:|:---:|---:|---:|---:|---|",
    ]
    for item in report.get("review_items", []):
        metrics = item["metric_summary"]
        midi_file = item["midi_file"]
        context_file = item["context_midi_file"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["role"]),
                    str(item["candidate_id"]),
                    str(bool(midi_file["exists"])),
                    str(bool(context_file["exists"])),
                    str(metrics.get("note_count", "")),
                    str(metrics.get("focused_note_count", "")),
                    f"{float(metrics.get('dead_air_ratio', 0.0) or 0.0):.4f}",
                    str(midi_file["sha256"][:12]),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build duration/coverage fill audio review package")
    parser.add_argument("--human_audio_boundary", type=str, required=True)
    parser.add_argument("--review_fill_guard", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--require_files_exist", action="store_true")
    parser.add_argument("--require_no_preference", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_review_package(
        read_json(Path(args.human_audio_boundary)),
        read_json(Path(args.review_fill_guard)),
        output_dir=output_dir,
    )
    summary = validate_audio_review_package(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        require_files_exist=bool(args.require_files_exist),
        require_no_preference=bool(args.require_no_preference),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_audio_review_package.json"
    markdown_path = output_dir / "duration_coverage_fill_audio_review_package.md"
    template_path = output_dir / "duration_coverage_fill_human_audio_review_input_template.json"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_audio_review_package_validation_summary.json", summary)
    write_json(template_path, report["review_input_template"])
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                **summary,
                "report_path": str(report_path),
                "markdown_path": str(markdown_path),
                "template_path": str(template_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
