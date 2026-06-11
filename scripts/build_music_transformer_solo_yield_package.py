"""Package ranked Music Transformer solo-yield candidates.

This script consumes a Stage B generation probe report, ranks generated MIDI
samples by objective solo-line proxies, copies the top MIDI files, and renders
them to WAV when a local FluidSynth setup is available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text
from scripts.render_stage_b_midi_to_solo_candidate_audio import (
    resolve_soundfont,
    sha256_file,
    wav_meta,
)


SCHEMA_VERSION = "music_transformer_solo_yield_package_v1"


class SoloYieldPackageError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldPackageError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sample_score(row: dict[str, Any]) -> float:
    metrics = _dict(row.get("metrics"))
    contour = _dict(row.get("phrase_contour"))
    rhythm = _dict(row.get("rhythm_profile"))
    pitch_roles = _dict(row.get("pitch_roles"))
    approach = _dict(row.get("approach_resolution"))
    coverage = _dict(row.get("temporal_coverage"))
    collapse = _dict(row.get("collapse"))
    postprocess = _dict(row.get("postprocess"))

    score = 0.0
    if bool(row.get("strict_valid", False)):
        score += 100.0
    if bool(row.get("valid", False)):
        score += 40.0
    if bool(row.get("grammar_gate_passed", False)):
        score += 20.0

    score += min(_float(metrics.get("note_count")), 24.0) * 0.8
    score += min(_float(metrics.get("unique_pitch_count")), 10.0) * 2.0
    score += _float(metrics.get("phrase_coverage_ratio")) * 10.0
    score += _float(coverage.get("position_span_ratio")) * 8.0
    score += _float(contour.get("direction_change_ratio")) * 8.0
    score += _float(rhythm.get("syncopated_onset_ratio")) * 8.0
    score += _float(rhythm.get("unique_bar_position_pattern_ratio")) * 5.0
    score += _float(approach.get("approach_resolution_ratio")) * 6.0
    score += _float(pitch_roles.get("tension_ratio")) * 6.0
    score += _float(pitch_roles.get("chord_tone_ratio")) * 4.0

    score -= _float(metrics.get("dead_air_ratio")) * 8.0
    score -= _float(contour.get("adjacent_repeated_pitch_ratio")) * 15.0
    score -= _float(collapse.get("repeated_position_pitch_pair_ratio")) * 12.0
    score -= _float(postprocess.get("postprocess_removal_ratio")) * 8.0
    score -= _float(rhythm.get("most_common_duration_ratio")) * 3.0

    pitch_span = _float(contour.get("pitch_span"))
    if pitch_span < 5:
        score -= 10.0
    if pitch_span > 24:
        score -= 6.0
    if _int(contour.get("longest_same_pitch_run")) >= 3:
        score -= 8.0
    return round(score, 6)


def rank_samples(report: dict[str, Any]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for row in _list(report.get("samples")):
        sample = _dict(row)
        midi_path = Path(str(sample.get("midi_path") or ""))
        if not midi_path.exists():
            continue
        metrics = _dict(sample.get("metrics"))
        contour = _dict(sample.get("phrase_contour"))
        rhythm = _dict(sample.get("rhythm_profile"))
        pitch_roles = _dict(sample.get("pitch_roles"))
        scored = {
            "sample_index": _int(sample.get("sample_index")),
            "sample_seed": _int(sample.get("sample_seed")),
            "source_midi_path": str(midi_path),
            "valid": bool(sample.get("valid", False)),
            "strict_valid": bool(sample.get("strict_valid", False)),
            "grammar_gate_passed": bool(sample.get("grammar_gate_passed", False)),
            "failure_reason": sample.get("failure_reason"),
            "score": sample_score(sample),
            "note_count": _int(metrics.get("note_count")),
            "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
            "pitch_min": metrics.get("pitch_min"),
            "pitch_max": metrics.get("pitch_max"),
            "duration_sec": _float(metrics.get("duration_sec")),
            "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
            "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
            "repetition_score": _float(metrics.get("repetition_score")),
            "direction_change_ratio": _float(contour.get("direction_change_ratio")),
            "adjacent_repeated_pitch_ratio": _float(contour.get("adjacent_repeated_pitch_ratio")),
            "syncopated_onset_ratio": _float(rhythm.get("syncopated_onset_ratio")),
            "chord_tone_ratio": _float(pitch_roles.get("chord_tone_ratio")),
            "tension_ratio": _float(pitch_roles.get("tension_ratio")),
        }
        ranked.append(scored)
    return sorted(
        ranked,
        key=lambda item: (
            bool(item["strict_valid"]),
            bool(item["valid"]),
            bool(item["grammar_gate_passed"]),
            float(item["score"]),
            int(item["note_count"]),
        ),
        reverse=True,
    )


def copy_ranked_midis(ranked: Sequence[dict[str, Any]], output_dir: Path, top_n: int) -> list[dict[str, Any]]:
    generated_dir = output_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for rank, item in enumerate(list(ranked)[: int(top_n)], start=1):
        source = Path(str(item["source_midi_path"]))
        target = generated_dir / f"candidate_{rank:02d}_sample_{int(item['sample_index']):02d}.mid"
        shutil.copy2(source, target)
        copied.append(
            {
                **item,
                "rank": rank,
                "midi_path": str(target),
                "midi_sha256": sha256_file(target),
            }
        )
    return copied


def render_wav(
    candidate: dict[str, Any],
    *,
    output_dir: Path,
    renderer_path: str,
    soundfont_path: str,
    sample_rate: int,
) -> dict[str, Any]:
    wav_path = output_dir / "audio" / f"candidate_{int(candidate['rank']):02d}_sample_{int(candidate['sample_index']):02d}.wav"
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(renderer_path),
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(sample_rate),
        str(soundfont_path),
        str(candidate["midi_path"]),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SoloYieldPackageError(
            f"audio render failed for rank {candidate['rank']}: {completed.stderr or completed.stdout}"
        )
    return {
        "rank": int(candidate["rank"]),
        "sample_index": int(candidate["sample_index"]),
        "sample_seed": int(candidate["sample_seed"]),
        "source_midi_path": str(candidate["midi_path"]),
        "score": candidate["score"],
        "wav_file": wav_meta(wav_path),
        "command": command,
        "stdout_tail": (completed.stdout or "")[-1000:],
        "stderr_tail": (completed.stderr or "")[-1000:],
    }


def render_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    renderer: str,
    soundfont: str,
    sample_rate: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    renderer_path = renderer or shutil.which("fluidsynth") or ""
    soundfont_path = resolve_soundfont(soundfont)
    render_setup = {
        "renderer_path": renderer_path,
        "soundfont_path": soundfont_path,
        "sample_rate": int(sample_rate),
        "render_attempted": bool(renderer_path and soundfont_path),
    }
    if not render_setup["render_attempted"]:
        return [], render_setup

    renderer_file = Path(renderer_path)
    soundfont_file = Path(soundfont_path).expanduser()
    if not renderer_file.exists():
        raise SoloYieldPackageError(f"renderer not found: {renderer_file}")
    if not soundfont_file.exists():
        raise SoloYieldPackageError(f"soundfont not found: {soundfont_file}")
    render_setup["soundfont_sha256"] = sha256_file(soundfont_file)
    return [
        render_wav(
            candidate,
            output_dir=output_dir,
            renderer_path=str(renderer_file),
            soundfont_path=str(soundfont_file),
            sample_rate=int(sample_rate),
        )
        for candidate in candidates
    ], render_setup


def build_package(
    probe_report: dict[str, Any],
    *,
    output_dir: Path,
    top_n: int,
    renderer: str,
    soundfont: str,
    sample_rate: int,
) -> dict[str, Any]:
    ranked = rank_samples(probe_report)
    if not ranked:
        raise SoloYieldPackageError("no rankable MIDI samples found")
    top_candidates = copy_ranked_midis(ranked, output_dir=output_dir, top_n=int(top_n))
    rendered, render_setup = render_candidates(
        top_candidates,
        output_dir=output_dir,
        renderer=renderer,
        soundfont=soundfont,
        sample_rate=int(sample_rate),
    )

    summary = _dict(probe_report.get("summary"))
    sample_count = _int(summary.get("sample_count"))
    valid_count = _int(summary.get("valid_sample_count"))
    strict_count = _int(summary.get("strict_valid_sample_count"))
    grammar_count = _int(summary.get("grammar_gate_sample_count"))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_probe": {
            "run_dir": probe_report.get("run_dir"),
            "checkpoint_dir": probe_report.get("checkpoint_dir"),
            "generation_mode": probe_report.get("generation_mode"),
            "request": probe_report.get("request"),
            "constrained_note_groups_per_bar": probe_report.get("constrained_note_groups_per_bar"),
            "chord_aware_pitches": probe_report.get("chord_aware_pitches"),
            "jazz_rhythm_positions": probe_report.get("jazz_rhythm_positions"),
            "jazz_duration_tokens": probe_report.get("jazz_duration_tokens"),
        },
        "yield_summary": {
            "sample_count": sample_count,
            "valid_sample_count": valid_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": grammar_count,
            "valid_yield_rate": float(valid_count / sample_count) if sample_count else 0.0,
            "strict_yield_rate": float(strict_count / sample_count) if sample_count else 0.0,
            "grammar_yield_rate": float(grammar_count / sample_count) if sample_count else 0.0,
            "selected_candidate_count": int(len(top_candidates)),
            "rendered_audio_file_count": int(len(rendered)),
        },
        "top_candidates": top_candidates,
        "render_setup": render_setup,
        "rendered_audio_files": rendered,
        "readiness": {
            "music_transformer_checkpoint_generation_used": True,
            "constrained_decoding_used": True,
            "solo_candidate_yield_measured": True,
            "ranked_midi_candidates_exported": bool(top_candidates),
            "technical_wav_render_completed": bool(rendered),
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }


def markdown_report(report: dict[str, Any]) -> str:
    yield_summary = report["yield_summary"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield MVP",
        "",
        "## Summary",
        "",
        f"- sample count: `{yield_summary['sample_count']}`",
        f"- valid yield: `{yield_summary['valid_sample_count']}` / `{yield_summary['sample_count']}`",
        f"- strict yield: `{yield_summary['strict_valid_sample_count']}` / `{yield_summary['sample_count']}`",
        f"- grammar yield: `{yield_summary['grammar_gate_sample_count']}` / `{yield_summary['sample_count']}`",
        f"- selected MIDI candidates: `{yield_summary['selected_candidate_count']}`",
        f"- rendered WAV files: `{yield_summary['rendered_audio_file_count']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Top Candidates",
        "",
        "| rank | sample | score | notes | unique pitches | dead air | direction change | MIDI |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("top_candidates", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["sample_index"]),
                    f"{float(item['score']):.3f}",
                    str(item["note_count"]),
                    str(item["unique_pitch_count"]),
                    f"{float(item['dead_air_ratio']):.3f}",
                    f"{float(item['direction_change_ratio']):.3f}",
                    str(item["midi_path"]),
                ]
            )
            + " |"
        )
    if report.get("rendered_audio_files"):
        lines.extend(
            [
                "",
                "## WAV Files",
                "",
                "| rank | sample | duration | wav path |",
                "|---:|---:|---:|---|",
            ]
        )
        for item in report["rendered_audio_files"]:
            wav_file = item["wav_file"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["rank"]),
                        str(item["sample_index"]),
                        f"{float(wav_file['duration_seconds']):.3f}",
                        str(wav_file["path"]),
                    ]
                )
                + " |"
            )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package ranked Music Transformer solo-yield candidates")
    parser.add_argument("--probe_report", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/music_transformer_finetune_mvp/solo_yield_mvp")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--renderer", type=str, default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", type=str, default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_package(
        read_json(Path(args.probe_report)),
        output_dir=output_dir,
        top_n=int(args.top_n),
        renderer=str(args.renderer or ""),
        soundfont=str(args.soundfont or ""),
        sample_rate=int(args.sample_rate),
    )
    write_json(output_dir / "solo_yield_package.json", report)
    write_text(output_dir / "solo_yield_package.md", markdown_report(report))
    print(json.dumps(report["yield_summary"], ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
