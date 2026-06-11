"""Build listening-review package for repaired Music Transformer solo candidates."""

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
from scripts.render_stage_b_midi_to_solo_candidate_audio import sha256_file, wav_meta  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_listening_package_v1"


class SoloYieldListeningPackageError(ValueError):
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


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldListeningPackageError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def find_case(package_path: Path, sweep_report: dict[str, Any]) -> dict[str, Any]:
    package_path_str = str(package_path)
    for case in _list(sweep_report.get("cases")):
        row = _dict(case)
        if str(_dict(row.get("package")).get("package_report_path") or "") == package_path_str:
            return row
    return {}


def candidate_audio(candidate: dict[str, Any], rendered_audio_files: list[Any]) -> dict[str, Any]:
    rank = _int(candidate.get("rank"))
    for item in rendered_audio_files:
        row = _dict(item)
        if _int(row.get("rank")) == rank:
            return row
    return {}


def copy_candidate_files(
    *,
    package_report: dict[str, Any],
    package_path: Path,
    case: dict[str, Any],
    output_dir: Path,
    start_index: int,
) -> list[dict[str, Any]]:
    midi_dir = output_dir / "midi"
    audio_dir = output_dir / "audio"
    midi_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rendered_audio = _list(package_report.get("rendered_audio_files"))
    case_label = str(case.get("label") or package_path.parent.name)
    chords = str(case.get("chords") or "")
    for offset, candidate in enumerate(_list(package_report.get("top_candidates")), start=0):
        item = _dict(candidate)
        review_index = int(start_index + offset)
        rank = _int(item.get("rank"))
        source_midi = Path(str(item.get("midi_path") or ""))
        if not source_midi.exists():
            raise SoloYieldListeningPackageError(f"candidate MIDI missing: {source_midi}")
        audio = candidate_audio(item, rendered_audio)
        source_wav = Path(str(_dict(audio.get("wav_file")).get("path") or ""))
        if not source_wav.exists():
            raise SoloYieldListeningPackageError(f"candidate WAV missing: {source_wav}")

        stem = f"candidate_{review_index:02d}_{case_label}_rank_{rank:02d}"
        target_midi = midi_dir / f"{stem}.mid"
        target_wav = audio_dir / f"{stem}.wav"
        shutil.copy2(source_midi, target_midi)
        shutil.copy2(source_wav, target_wav)
        rows.append(
            {
                "review_index": review_index,
                "case_label": case_label,
                "chords": chords,
                "case_seed": _int(case.get("seed")),
                "case_strict_yield_rate": _float(case.get("strict_yield_rate")),
                "source_package_path": str(package_path),
                "source_midi_path": str(source_midi),
                "source_wav_path": str(source_wav),
                "rank": rank,
                "sample_index": _int(item.get("sample_index")),
                "sample_seed": _int(item.get("sample_seed")),
                "score": _float(item.get("score")),
                "note_count": _int(item.get("note_count")),
                "unique_pitch_count": _int(item.get("unique_pitch_count")),
                "dead_air_ratio": _float(item.get("dead_air_ratio")),
                "direction_change_ratio": _float(item.get("direction_change_ratio")),
                "syncopated_onset_ratio": _float(item.get("syncopated_onset_ratio")),
                "chord_tone_ratio": _float(item.get("chord_tone_ratio")),
                "tension_ratio": _float(item.get("tension_ratio")),
                "review_midi_path": str(target_midi),
                "review_wav_path": str(target_wav),
                "review_midi_sha256": sha256_file(target_midi),
                "review_wav_file": wav_meta(target_wav),
            }
        )
    return rows


def build_review_input_template(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "music_transformer_solo_yield_listening_input_v1",
        "review_status": "pending",
        "reviewer_notes": "",
        "overall_decision": "pending",
        "preferred_review_index": None,
        "candidates": [
            {
                "review_index": _int(row.get("review_index")),
                "case_label": row.get("case_label"),
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


def build_listening_package(
    sweep_report: dict[str, Any],
    *,
    output_dir: Path,
    max_candidates: int,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    package_paths = [
        Path(str(_dict(case.get("package")).get("package_report_path") or ""))
        for case in _list(sweep_report.get("cases"))
    ]
    package_paths = [path for path in package_paths if str(path)]
    if not package_paths:
        raise SoloYieldListeningPackageError("no package paths in sweep report")
    candidates: list[dict[str, Any]] = []
    for package_path in package_paths:
        if len(candidates) >= int(max_candidates):
            break
        package_report = read_json(package_path)
        case = find_case(package_path, sweep_report)
        copied = copy_candidate_files(
            package_report=package_report,
            package_path=package_path,
            case=case,
            output_dir=output_dir,
            start_index=len(candidates) + 1,
        )
        candidates.extend(copied[: max(0, int(max_candidates) - len(candidates))])
    review_input_template = build_review_input_template(candidates)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_sweep": {
            "schema_version": sweep_report.get("schema_version"),
            "output_dir": sweep_report.get("output_dir"),
            "case_count": _int(_dict(sweep_report.get("aggregate")).get("case_count")),
            "sample_count": _int(_dict(sweep_report.get("aggregate")).get("sample_count")),
            "strict_valid_sample_count": _int(
                _dict(sweep_report.get("aggregate")).get("strict_valid_sample_count")
            ),
            "strict_yield_rate": _float(_dict(sweep_report.get("aggregate")).get("strict_yield_rate")),
            "rendered_audio_file_count": _int(_dict(sweep_report.get("aggregate")).get("rendered_audio_file_count")),
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "review_input_template": review_input_template,
        "readiness": {
            "listening_review_package_ready": bool(candidates),
            "candidate_midi_files_copied": len(candidates),
            "candidate_wav_files_copied": len(candidates),
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_candidate_listening_review",
            "next_boundary": "music_transformer_solo_yield_listening_input_guard",
            "critical_user_input_required": False,
            "reason": "technical candidate package ready; listening preference input remains pending",
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
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int, require_no_quality_claim: bool) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    if _int(report.get("candidate_count")) < int(min_candidates):
        raise SoloYieldListeningPackageError("candidate count below requirement")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldListeningPackageError(f"unexpected quality claim: {claimed}")
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(report.get("candidate_count")),
        "candidate_midi_files_copied": _int(readiness.get("candidate_midi_files_copied")),
        "candidate_wav_files_copied": _int(readiness.get("candidate_wav_files_copied")),
        "listening_review_package_ready": bool(readiness.get("listening_review_package_ready", False)),
        "validated_listening_input_present": bool(readiness.get("validated_listening_input_present", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "next_boundary": str(_dict(report.get("decision")).get("next_boundary") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_sweep"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Music Transformer Solo Yield Listening Review Package",
        "",
        "## Summary",
        "",
        f"- source strict yield: `{source['strict_valid_sample_count']}` / `{source['sample_count']}`",
        f"- source strict yield rate: `{float(source['strict_yield_rate']):.4f}`",
        f"- source rendered audio files: `{source['rendered_audio_file_count']}`",
        f"- review candidate count: `{report['candidate_count']}`",
        f"- MIDI files copied: `{readiness['candidate_midi_files_copied']}`",
        f"- WAV files copied: `{readiness['candidate_wav_files_copied']}`",
        f"- validated listening input present: `{_bool_token(readiness['validated_listening_input_present'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Candidates",
        "",
        "| review | case | chords | score | notes | dead air | WAV | MIDI |",
        "|---:|---|---|---:|---:|---:|---|---|",
    ]
    for row in report.get("candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["review_index"]),
                    f"`{row['case_label']}`",
                    f"`{row['chords']}`",
                    f"{float(row['score']):.3f}",
                    str(row["note_count"]),
                    f"{float(row['dead_air_ratio']):.4f}",
                    f"`{row['review_wav_path']}`",
                    f"`{row['review_midi_path']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Review Input", "", f"- `{report['output_dir']}/listening_review_input_template.json`"])
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build solo-yield listening review package")
    parser.add_argument("--sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_listening_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=1)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_package(
        read_json(Path(args.sweep_report)),
        output_dir=output_dir,
        max_candidates=int(args.max_candidates),
    )
    summary = validate_report(
        report,
        min_candidates=int(args.min_candidates),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "listening_review_package_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
