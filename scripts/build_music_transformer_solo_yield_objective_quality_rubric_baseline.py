"""Build an objective quality rubric baseline for solo-yield review candidates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


SCHEMA_VERSION = "music_transformer_solo_yield_objective_quality_rubric_baseline_v1"

RUBRIC_THRESHOLDS = {
    "note_count_min": 28,
    "note_count_watch_max": 36,
    "dead_air_watch_max": 0.66,
    "dead_air_major_max": 0.68,
    "direction_change_min": 0.50,
    "syncopated_onset_min": 0.70,
    "chord_tone_min": 0.40,
    "chord_tone_max": 0.70,
    "tension_min": 0.20,
    "tension_watch_max": 0.36,
}

REPAIR_TARGET_BY_LABEL = {
    "dead_air_high": "dead_air_density_balance_repair",
    "weak_direction_change": "phrase_direction_balance_repair",
    "low_tension_color": "tension_color_balance_repair",
    "low_syncopation": "rhythm_syncopation_balance_repair",
    "chord_tone_out_of_range": "chord_role_balance_repair",
    "low_note_density": "dead_air_density_balance_repair",
}

REPAIR_PRIORITY = [
    "dead_air_density_balance_repair",
    "phrase_direction_balance_repair",
    "tension_color_balance_repair",
    "rhythm_syncopation_balance_repair",
    "chord_role_balance_repair",
]


class SoloYieldObjectiveQualityRubricBaselineError(ValueError):
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


def _format_counts(value: dict[str, Any]) -> str:
    if not value:
        return "none"
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldObjectiveQualityRubricBaselineError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reject_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldObjectiveQualityRubricBaselineError(
            f"unexpected quality claim: {claimed}"
        )


def label_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    note_count = _int(candidate.get("note_count"))
    dead_air_ratio = _float(candidate.get("dead_air_ratio"))
    direction_change_ratio = _float(candidate.get("direction_change_ratio"))
    syncopated_onset_ratio = _float(candidate.get("syncopated_onset_ratio"))
    chord_tone_ratio = _float(candidate.get("chord_tone_ratio"))
    tension_ratio = _float(candidate.get("tension_ratio"))

    major_labels: list[str] = []
    watch_labels: list[str] = []

    if note_count < RUBRIC_THRESHOLDS["note_count_min"]:
        major_labels.append("low_note_density")
    elif note_count > RUBRIC_THRESHOLDS["note_count_watch_max"]:
        watch_labels.append("note_density_high_watch")

    if dead_air_ratio > RUBRIC_THRESHOLDS["dead_air_major_max"]:
        major_labels.append("dead_air_high")
    elif dead_air_ratio > RUBRIC_THRESHOLDS["dead_air_watch_max"]:
        watch_labels.append("dead_air_watch")

    if direction_change_ratio < RUBRIC_THRESHOLDS["direction_change_min"]:
        major_labels.append("weak_direction_change")

    if syncopated_onset_ratio < RUBRIC_THRESHOLDS["syncopated_onset_min"]:
        major_labels.append("low_syncopation")

    if not (
        RUBRIC_THRESHOLDS["chord_tone_min"]
        <= chord_tone_ratio
        <= RUBRIC_THRESHOLDS["chord_tone_max"]
    ):
        major_labels.append("chord_tone_out_of_range")

    if tension_ratio < RUBRIC_THRESHOLDS["tension_min"]:
        major_labels.append("low_tension_color")
    elif tension_ratio > RUBRIC_THRESHOLDS["tension_watch_max"]:
        watch_labels.append("tension_high_watch")

    return {
        "review_index": _int(candidate.get("review_index")),
        "case_label": str(candidate.get("case_label") or ""),
        "rank": _int(candidate.get("rank")),
        "score": _float(candidate.get("score")),
        "note_count": note_count,
        "dead_air_ratio": dead_air_ratio,
        "direction_change_ratio": direction_change_ratio,
        "syncopated_onset_ratio": syncopated_onset_ratio,
        "chord_tone_ratio": chord_tone_ratio,
        "tension_ratio": tension_ratio,
        "major_labels": major_labels,
        "watch_labels": watch_labels,
        "quality_proxy_pass": not major_labels,
        "review_midi_path": str(candidate.get("review_midi_path") or ""),
        "review_wav_path": str(candidate.get("review_wav_path") or ""),
    }


def select_repair_target(major_label_counts: Counter[str]) -> str:
    target_counts: Counter[str] = Counter()
    for label, count in major_label_counts.items():
        target = REPAIR_TARGET_BY_LABEL.get(label)
        if target:
            target_counts[target] += count
    if not target_counts:
        return "listening_review_or_broader_repeatability"
    return sorted(
        target_counts.items(),
        key=lambda item: (-item[1], REPAIR_PRIORITY.index(item[0]) if item[0] in REPAIR_PRIORITY else 999),
    )[0][0]


def next_boundary_for_target(target: str) -> str:
    return {
        "dead_air_density_balance_repair": "music_transformer_solo_yield_dead_air_density_balance_repair",
        "phrase_direction_balance_repair": "music_transformer_solo_yield_phrase_direction_balance_repair",
        "tension_color_balance_repair": "music_transformer_solo_yield_tension_color_balance_repair",
        "rhythm_syncopation_balance_repair": "music_transformer_solo_yield_rhythm_syncopation_balance_repair",
        "chord_role_balance_repair": "music_transformer_solo_yield_chord_role_balance_repair",
    }.get(target, "music_transformer_solo_yield_listening_review_or_broader_repeatability")


def build_rubric_report(
    *,
    listening_package: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _reject_quality_claim(listening_package)
    candidates = [label_candidate(_dict(item)) for item in _list(listening_package.get("candidates"))]
    major_label_counts = Counter(label for item in candidates for label in item["major_labels"])
    watch_label_counts = Counter(label for item in candidates for label in item["watch_labels"])
    selected_target = select_repair_target(major_label_counts)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_package": {
            "schema_version": listening_package.get("schema_version"),
            "output_dir": listening_package.get("output_dir"),
            "candidate_count": _int(listening_package.get("candidate_count")),
        },
        "thresholds": dict(RUBRIC_THRESHOLDS),
        "candidate_labels": candidates,
        "aggregate": {
            "candidate_count": len(candidates),
            "quality_proxy_pass_count": sum(1 for item in candidates if item["quality_proxy_pass"]),
            "quality_proxy_fail_count": sum(1 for item in candidates if not item["quality_proxy_pass"]),
            "major_label_counts": dict(sorted(major_label_counts.items())),
            "watch_label_counts": dict(sorted(watch_label_counts.items())),
        },
        "decision": {
            "current_boundary": "music_transformer_solo_yield_objective_quality_rubric_baseline",
            "selected_repair_target": selected_target,
            "next_boundary": next_boundary_for_target(selected_target),
            "critical_user_input_required": False,
            "reason": (
                "objective rubric found candidate-level proxy failures; select highest-priority aggregate repair target"
                if major_label_counts
                else "objective rubric found no major proxy failures; keep candidates for listening review or broader repeatability"
            ),
        },
        "readiness": {
            "objective_quality_rubric_completed": True,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "rubric_threshold_calibration",
            "repair_effectiveness",
        ],
    }


def validate_rubric_report(
    report: dict[str, Any],
    *,
    min_candidate_count: int,
    expected_target: str,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    if str(report.get("schema_version")) != SCHEMA_VERSION:
        raise SoloYieldObjectiveQualityRubricBaselineError("schema version mismatch")
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    if _int(aggregate.get("candidate_count")) < min_candidate_count:
        raise SoloYieldObjectiveQualityRubricBaselineError("candidate count below requirement")
    if str(decision.get("selected_repair_target") or "") != expected_target:
        raise SoloYieldObjectiveQualityRubricBaselineError("selected repair target mismatch")
    if require_no_quality_claim:
        claimed = [
            key
            for key in ("musical_quality_claimed", "artist_style_claimed", "production_ready_claimed")
            if bool(readiness.get(key, True))
        ]
        if claimed:
            raise SoloYieldObjectiveQualityRubricBaselineError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "schema_version": str(report.get("schema_version")),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "quality_proxy_pass_count": _int(aggregate.get("quality_proxy_pass_count")),
        "quality_proxy_fail_count": _int(aggregate.get("quality_proxy_fail_count")),
        "major_label_counts": _dict(aggregate.get("major_label_counts")),
        "watch_label_counts": _dict(aggregate.get("watch_label_counts")),
        "selected_repair_target": str(decision.get("selected_repair_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Objective Quality Rubric Baseline",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- quality proxy pass/fail: `{aggregate['quality_proxy_pass_count']}` / `{aggregate['quality_proxy_fail_count']}`",
        f"- major label counts: `{_format_counts(aggregate['major_label_counts'])}`",
        f"- watch label counts: `{_format_counts(aggregate['watch_label_counts'])}`",
        f"- selected repair target: `{decision['selected_repair_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate Labels",
        "",
        "| idx | case | rank | notes | dead air | direction | tension | major labels | watch labels | MIDI |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report.get("candidate_labels", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["review_index"]),
                    f"`{item['case_label']}`",
                    str(item["rank"]),
                    str(item["note_count"]),
                    f"{float(item['dead_air_ratio']):.4f}",
                    f"{float(item['direction_change_ratio']):.4f}",
                    f"{float(item['tension_ratio']):.4f}",
                    f"`{','.join(item['major_labels']) or 'none'}`",
                    f"`{','.join(item['watch_labels']) or 'none'}`",
                    f"`{item['review_midi_path']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build solo-yield objective quality rubric baseline")
    parser.add_argument("--listening_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_objective_quality_rubric",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=8)
    parser.add_argument("--expected_target", type=str, default="dead_air_density_balance_repair")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_rubric_report(
        listening_package=read_json(Path(args.listening_package)),
        output_dir=output_dir,
    )
    summary = validate_rubric_report(
        report,
        min_candidate_count=int(args.min_candidate_count),
        expected_target=str(args.expected_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "objective_quality_rubric_baseline.json", report)
    write_json(output_dir / "objective_quality_rubric_baseline_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "objective_quality_rubric_baseline.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
