"""Render phrase-bank CLI user-input smoke MIDI candidates to WAV."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.check_stage_b_midi_to_solo_phrase_bank_cli_user_input_smoke import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import resolve_soundfont, sha256_file  # noqa: E402
from scripts.render_stage_b_midi_to_solo_phrase_bank_audio import (  # noqa: E402
    build_render_plan,
    execute_render_plan,
    validate_audio_render_report as validate_generic_audio_render_report,
)


class StageBMidiToSoloPhraseBankCliAudioSmokeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_listening_review_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke_v1"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_source_report(report: dict[str, Any], *, expected_count: int) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("CLI user-input smoke boundary required")
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("CLI smoke must route to audio render smoke")
    if not bool(readiness.get("user_input_smoke_completed", False)):
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("user-input smoke completion required")
    if not bool(readiness.get("explicit_input_path_used", False)):
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("explicit input path smoke required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="CLI user-input smoke readiness")
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    if len(candidates) < int(expected_count):
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("not enough smoke MIDI candidates")
    compacted: list[dict[str, Any]] = []
    for item in candidates[: int(expected_count)]:
        midi_path = str(item.get("repaired_midi_path") or "")
        if not midi_path or not Path(midi_path).exists():
            raise StageBMidiToSoloPhraseBankCliAudioSmokeError(f"repaired MIDI missing: {midi_path}")
        if not bool(item.get("objective_supported", False)):
            raise StageBMidiToSoloPhraseBankCliAudioSmokeError("candidate objective support required")
        compacted.append(
            {
                "rank": _int(item.get("rank")),
                "mode": "cli_user_input_smoke",
                "sample_index": _int(item.get("rank")),
                "sample_seed": _int(item.get("sample_seed")),
                "midi_path": midi_path,
                "note_count": _int(item.get("note_count")),
                "unique_pitch_count": _int(item.get("unique_pitch_count")),
                "dead_air_ratio": _float(item.get("dead_air_ratio")),
                "phrase_coverage_ratio": _float(item.get("phrase_coverage_ratio")),
            }
        )
    return compacted


def build_audio_smoke_report(
    source_report: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
    expected_file_count: int,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    candidates = validate_source_report(source_report, expected_count=expected_file_count)
    resolved_renderer = renderer_path or shutil.which("fluidsynth") or ""
    resolved_soundfont = resolve_soundfont(soundfont_path)
    plan = build_render_plan(
        candidates,
        output_dir=output_dir,
        renderer_path=resolved_renderer,
        soundfont_path=resolved_soundfont,
        sample_rate=sample_rate,
    )
    rendered = execute_render_plan(plan, runner=runner) if runner else execute_render_plan(plan)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_boundary": SOURCE_BOUNDARY,
        "source_input": _dict(source_report.get("input")),
        "renderer": {
            "name": "fluidsynth",
            "path": resolved_renderer,
        },
        "soundfont": {
            "path": str(Path(resolved_soundfont).expanduser()),
            "sha256": sha256_file(Path(resolved_soundfont).expanduser()),
        },
        "rendered_audio_files": rendered,
        "audio_render_boundary": {
            "boundary": BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": int(len(rendered)),
            "technical_wav_validation": True,
            "cli_user_input_audio_render_completed": True,
            "phrase_bank_ranked_audio_render_completed": True,
            "phrase_bank_listening_review_package_required": True,
            "audio_output_claimed": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "explicit-input CLI repaired MIDI candidates rendered to WAV; listening preference remains unclaimed",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo phrase-bank CLI listening review package",
    }


def validate_audio_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_file_count: int,
    expected_sample_rate: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    summary = validate_generic_audio_render_report(
        report,
        expected_boundary=expected_boundary,
        expected_next_boundary=expected_next_boundary,
        expected_file_count=int(expected_file_count),
        expected_sample_rate=int(expected_sample_rate),
        require_phrase_bank_audio_path=True,
        require_no_quality_claim=require_no_quality_claim,
    )
    boundary = _dict(report.get("audio_render_boundary"))
    if not bool(boundary.get("cli_user_input_audio_render_completed", False)):
        raise StageBMidiToSoloPhraseBankCliAudioSmokeError("CLI audio smoke completion required")
    summary["source_boundary"] = str(report.get("source_boundary") or "")
    summary["cli_user_input_audio_render_completed"] = bool(
        boundary.get("cli_user_input_audio_render_completed", False)
    )
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["audio_render_boundary"]
    decision = report["decision"]
    source_input = report["source_input"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank CLI Audio Render Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- input MIDI: `{source_input.get('midi_path', '')}`",
        f"- render attempted: `{_bool_token(boundary['render_attempted'])}`",
        f"- rendered audio file count: `{boundary['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(boundary['technical_wav_validation'])}`",
        f"- CLI user-input audio render completed: `{_bool_token(boundary['cli_user_input_audio_render_completed'])}`",
        f"- audio rendered quality claimed: `{_bool_token(boundary['audio_rendered_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        "",
        "## Rendered Files",
        "",
    ]
    for item in report.get("rendered_audio_files", []):
        wav_file = item["wav_file"]
        lines.append(
            f"- rank `{item['rank']}` seed `{item['sample_seed']}`: "
            f"`{wav_file['path']}`, duration `{float(wav_file['duration_seconds']):.3f}`, "
            f"sample rate `{wav_file['sample_rate']}`, size `{wav_file['size_bytes']}`, "
            f"sha256 `{str(wav_file['sha256'])[:12]}`"
        )
    lines.extend(["", "## Claim Boundary", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render explicit-input phrase-bank CLI smoke WAV files")
    parser.add_argument("--user_input_smoke_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=3)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_audio_smoke_report(
        read_json(Path(args.user_input_smoke_report)),
        output_dir=output_dir,
        renderer_path=str(args.renderer or ""),
        soundfont_path=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
        expected_file_count=int(args.expected_file_count),
    )
    summary = validate_audio_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_file_count=int(args.expected_file_count),
        expected_sample_rate=int(args.sample_rate),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_phrase_bank_cli_audio_render_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
