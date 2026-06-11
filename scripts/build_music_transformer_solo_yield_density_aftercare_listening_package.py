"""Build listening package for density-aftercare repaired solo-yield candidates."""

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

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.render_music_transformer_solo_yield_density_aftercare_audio import (  # noqa: E402
    SCHEMA_VERSION as AUDIO_PACKAGE_SCHEMA_VERSION,
    build_audio_package,
    load_or_build_repair_sweep,
    read_json,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file, wav_meta  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_density_aftercare_listening_package_v1"
INPUT_SCHEMA_VERSION = "music_transformer_solo_yield_density_aftercare_listening_input_v1"
BOUNDARY = "music_transformer_solo_yield_density_aftercare_listening_package"
NEXT_BOUNDARY = "music_transformer_solo_yield_density_aftercare_listening_input_guard"


class SoloYieldDensityAftercareListeningPackageError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldDensityAftercareListeningPackageError(f"unexpected quality claim: {claimed}")


def _validate_audio_package(report: dict[str, Any], *, min_candidates: int) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != AUDIO_PACKAGE_SCHEMA_VERSION:
        raise SoloYieldDensityAftercareListeningPackageError("audio package schema required")
    _require_no_quality_claim(report)
    aggregate = _dict(report.get("aggregate"))
    if not bool(aggregate.get("technical_wav_validation", False)):
        raise SoloYieldDensityAftercareListeningPackageError("technical WAV validation required")
    rows = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(rows) < int(min_candidates):
        raise SoloYieldDensityAftercareListeningPackageError("rendered audio file count below requirement")
    for row in rows:
        midi_path = Path(str(row.get("repaired_midi_path") or ""))
        wav_path = Path(str(_dict(row.get("wav_file")).get("path") or ""))
        if not midi_path.exists():
            raise SoloYieldDensityAftercareListeningPackageError(f"MIDI missing: {midi_path}")
        if not wav_path.exists():
            raise SoloYieldDensityAftercareListeningPackageError(f"WAV missing: {wav_path}")
    return rows


def build_review_input_template(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "review_status": "pending",
        "reviewer_notes": "",
        "overall_decision": "pending",
        "preferred_review_index": None,
        "candidates": [
            {
                "review_index": _int(row.get("review_index")),
                "case_label": str(row.get("case_label") or ""),
                "decision": "pending",
                "usable_as_jazz_solo_phrase": None,
                "primary_failure": None,
                "notes": "",
            }
            for row in candidates
        ],
        "allowed_primary_failures": [
            "not_solo_like",
            "too_mechanical",
            "too_sparse",
            "too_dense",
            "rhythm_not_swinging",
            "outside_harmony",
            "weak_phrase_shape",
            "audio_render_issue",
            "none",
        ],
    }


def copy_candidates(audio_rows: list[dict[str, Any]], *, output_dir: Path) -> list[dict[str, Any]]:
    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"
    midi_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for row in audio_rows:
        review_index = _int(row.get("review_index"))
        case_label = str(row.get("case_label") or "candidate")
        source_midi = Path(str(row.get("repaired_midi_path") or ""))
        source_wav = Path(str(_dict(row.get("wav_file")).get("path") or ""))
        target_stem = f"candidate_{review_index:02d}_{case_label}_density_aftercare"
        target_midi = midi_dir / f"{target_stem}.mid"
        target_wav = audio_dir / f"{target_stem}.wav"
        shutil.copy2(source_midi, target_midi)
        shutil.copy2(source_wav, target_wav)
        copied.append(
            {
                "review_index": review_index,
                "case_label": case_label,
                "source_midi_path": str(source_midi),
                "source_wav_path": str(source_wav),
                "review_midi_path": str(target_midi),
                "review_wav_path": str(target_wav),
                "review_midi_sha256": sha256_file(target_midi),
                "review_wav_file": wav_meta(target_wav),
            }
        )
    return copied


def build_listening_package(
    audio_package: dict[str, Any],
    *,
    output_dir: Path,
    min_candidates: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_rows = _validate_audio_package(audio_package, min_candidates=int(min_candidates))
    candidates = copy_candidates(audio_rows[: int(min_candidates)], output_dir=output_dir)
    review_input_template = build_review_input_template(candidates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_audio_package": {
            "schema_version": audio_package.get("schema_version"),
            "output_dir": audio_package.get("output_dir"),
            "rendered_wav_count": _int(_dict(audio_package.get("aggregate")).get("rendered_wav_count")),
            "technical_wav_validation": bool(
                _dict(audio_package.get("aggregate")).get("technical_wav_validation", False)
            ),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "review_input_template": review_input_template,
        "readiness": {
            "listening_package_ready": bool(candidates),
            "candidate_midi_files_copied": len(candidates),
            "candidate_wav_files_copied": len(candidates),
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": "density_aftercare_listening_input_guard",
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "reason": "MIDI/WAV review package ready; preference input remains pending",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "listening_review_package.json", report)
    write_json(output_dir / "listening_review_input_template.json", review_input_template)
    write_text(output_dir / "listening_review_package.md", markdown_report(report))
    write_json(output_dir / "listening_review_package_summary.json", validate_report(report))
    return report


def load_or_build_audio_package(
    *,
    audio_package_report_path: Path,
    output_dir: Path,
    repair_sweep_report_path: Path,
    source_repair_sweep_report_path: Path,
    objective_decision_report_path: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
) -> dict[str, Any]:
    if audio_package_report_path.exists():
        return read_json(audio_package_report_path)
    repair_sweep = load_or_build_repair_sweep(
        repair_sweep_report_path=repair_sweep_report_path,
        output_dir=output_dir / "source_repair_build",
        source_repair_sweep_report_path=source_repair_sweep_report_path,
        objective_decision_report_path=objective_decision_report_path,
    )
    return build_audio_package(
        repair_sweep,
        output_dir=output_dir / "source_audio_package",
        renderer_path=renderer_path,
        soundfont_path=soundfont_path,
        sample_rate=sample_rate,
    )


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldDensityAftercareListeningPackageError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    candidate_count = _int(report.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldDensityAftercareListeningPackageError("candidate count below requirement")
    if not bool(readiness.get("listening_package_ready", False)):
        raise SoloYieldDensityAftercareListeningPackageError("listening package readiness required")
    _require_no_quality_claim(report)
    decision = _dict(report.get("decision"))
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "candidate_midi_files_copied": _int(readiness.get("candidate_midi_files_copied")),
        "candidate_wav_files_copied": _int(readiness.get("candidate_wav_files_copied")),
        "review_input_template_written": bool(readiness.get("review_input_template_written", False)),
        "validated_listening_input_present": bool(readiness.get("validated_listening_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "audio_rendered_quality_claimed": bool(readiness.get("audio_rendered_quality_claimed", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    source = report["source_audio_package"]
    lines = [
        "# Music Transformer Solo Yield Density Aftercare Listening Package",
        "",
        "## Summary",
        "",
        f"- source rendered WAV count: `{source['rendered_wav_count']}`",
        f"- source technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- review candidate count: `{report['candidate_count']}`",
        f"- MIDI files copied: `{readiness['candidate_midi_files_copied']}`",
        f"- WAV files copied: `{readiness['candidate_wav_files_copied']}`",
        f"- review input template written: `{_bool_token(readiness['review_input_template_written'])}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(readiness['preference_fill_allowed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Candidates",
        "",
    ]
    for row in report.get("candidates", []):
        wav = row["review_wav_file"]
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - MIDI: `{row['review_midi_path']}`",
                f"  - WAV: `{row['review_wav_path']}`",
                f"  - duration: `{float(wav['duration_seconds']):.3f}`",
                f"  - sample rate: `{wav['sample_rate']}`",
            ]
        )
    lines.extend(["", "## Review Input", "", f"- `{report['output_dir']}/listening_review_input_template.json`"])
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build density aftercare listening package")
    parser.add_argument(
        "--audio_package_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare_audio/"
            "issue_1296_density_aftercare_audio_package/density_aftercare_audio_package.json"
        ),
    )
    parser.add_argument(
        "--repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare/"
            "issue_1294_density_aftercare_sweep/density_aftercare_sweep.json"
        ),
    )
    parser.add_argument(
        "--source_repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair/"
            "issue_1284_chord_role_balance_repair_sweep/chord_role_balance_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair_objective_next/"
            "issue_1292_chord_role_balance_objective_next/"
            "chord_role_balance_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare_listening_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default="")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--min_candidates", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    audio_package = load_or_build_audio_package(
        audio_package_report_path=Path(args.audio_package_report),
        output_dir=output_dir,
        repair_sweep_report_path=Path(args.repair_sweep_report),
        source_repair_sweep_report_path=Path(args.source_repair_sweep_report),
        objective_decision_report_path=Path(args.objective_decision_report),
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
    )
    report = build_listening_package(
        audio_package,
        output_dir=output_dir,
        min_candidates=int(args.min_candidates),
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    write_json(output_dir / "listening_review_package_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
