"""Build a review handoff for bebop-language best-of packages."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SCHEMA_VERSION = "stage_b_midi_to_solo_bebop_language_review_handoff_v1"
BOUNDARY = "stage_b_midi_to_solo_bebop_language_review_handoff"
NEXT_BOUNDARY = "listening_review_input_or_motion_balance_guard_tightening"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "stage_b_midi_to_solo_bebop_language_review_handoff"


class BebopLanguageReviewHandoffError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BebopLanguageReviewHandoffError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _report_id(path: Path) -> str:
    return path.parent.name


def _metric(report: dict[str, Any], key: str) -> float:
    return _float(_dict(report.get("aggregate")).get(key))


def _path_exists(path_text: str) -> bool:
    return bool(path_text) and Path(path_text).expanduser().exists()


def _wav_path(candidate: dict[str, Any], key: str) -> str:
    return str(_dict(_dict(candidate.get(key)).get("wav_file")).get("path") or "")


def _validate_claims(report: dict[str, Any], *, label: str) -> None:
    if bool(report.get("quality_claimed", True)):
        raise BebopLanguageReviewHandoffError(f"{label} quality claim must be false")
    if bool(report.get("model_direct_claimed", True)):
        raise BebopLanguageReviewHandoffError(f"{label} model-direct claim must be false")


def validate_best_of_package(
    report: dict[str, Any],
    *,
    package_path: Path,
    expected_candidate_count: int,
) -> list[dict[str, Any]]:
    if str(report.get("schema_version") or "") != "stage_b_midi_to_solo_bebop_language_best_of_package_v1":
        raise BebopLanguageReviewHandoffError("bebop-language best-of package required")
    if str(report.get("boundary") or "") != "stage_b_midi_to_solo_bebop_language_best_of_package":
        raise BebopLanguageReviewHandoffError("best-of package boundary required")
    _validate_claims(report, label="best-of package")
    selected = [_dict(item) for item in _list(report.get("selected_candidates"))]
    if len(selected) != int(expected_candidate_count):
        raise BebopLanguageReviewHandoffError(
            f"expected selected candidate count {expected_candidate_count}, got {len(selected)}"
        )
    aggregate_selected = _int(_dict(report.get("aggregate")).get("selected_candidate_count"))
    if aggregate_selected != int(expected_candidate_count):
        raise BebopLanguageReviewHandoffError(
            f"expected aggregate selected count {expected_candidate_count}, got {aggregate_selected}"
        )

    ranks = sorted(_int(item.get("rank")) for item in selected)
    if ranks != list(range(1, int(expected_candidate_count) + 1)):
        raise BebopLanguageReviewHandoffError(f"candidate ranks must be consecutive: {ranks}")

    for item in selected:
        required_paths = [
            str(item.get("midi_path") or ""),
            str(item.get("context_midi_path") or ""),
            _wav_path(item, "solo_audio"),
            _wav_path(item, "context_audio"),
        ]
        missing = [path for path in required_paths if not _path_exists(path)]
        if missing:
            raise BebopLanguageReviewHandoffError(
                f"missing review artifact for package {_report_id(package_path)} rank {item.get('rank')}: {missing}"
            )
    return selected


def validate_note_review(
    report: dict[str, Any],
    *,
    package_path: Path,
    expected_candidate_count: int,
) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != "stage_b_midi_to_solo_bebop_language_note_review_v1":
        raise BebopLanguageReviewHandoffError("bebop-language note review required")
    _validate_claims(report, label="note review")
    if _int(report.get("candidate_count")) != int(expected_candidate_count):
        raise BebopLanguageReviewHandoffError("note review candidate count mismatch")
    source_package = Path(str(report.get("source_package") or ""))
    if not source_package.is_absolute():
        source_package = ROOT_DIR / source_package
    if source_package.resolve() != package_path.resolve():
        raise BebopLanguageReviewHandoffError("note review source package mismatch")
    candidates = [_dict(item) for item in _list(report.get("candidates"))]
    if len(candidates) != int(expected_candidate_count):
        raise BebopLanguageReviewHandoffError("note review candidate rows mismatch")
    empty_note_rows = [
        _int(item.get("rank"))
        for item in candidates
        if not _list(item.get("first_notes"))
    ]
    if empty_note_rows:
        raise BebopLanguageReviewHandoffError(f"note review first-note rows missing: {empty_note_rows}")
    return {
        "source_package": str(source_package),
        "candidate_scope": str(report.get("candidate_scope") or ""),
        "candidate_count": len(candidates),
        "max_notes_per_candidate": _int(report.get("max_notes_per_candidate")),
    }


def build_candidate_rows(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        metrics = _dict(item.get("objective_metrics"))
        motion_repair = _dict(item.get("motion_balance_repair"))
        rhythm_repair = _dict(item.get("rhythm_articulation_repair"))
        rows.append(
            {
                "rank": _int(item.get("rank")),
                "case_label": str(item.get("case_label") or ""),
                "variant_index": _int(item.get("variant_index")),
                "chords": list(_list(item.get("chords"))),
                "score": _float(item.get("score")),
                "gate_penalty": _float(item.get("gate_penalty")),
                "step_motion_ratio": _float(metrics.get("step_motion_ratio")),
                "chromatic_step_ratio": _float(metrics.get("chromatic_step_ratio")),
                "third_fourth_motion_ratio": _float(metrics.get("third_fourth_motion_ratio")),
                "large_leap_ratio": _float(metrics.get("large_leap_ratio")),
                "chord_tone_ratio": _float(metrics.get("chord_tone_ratio")),
                "strong_beat_chord_tone_ratio": _float(metrics.get("strong_beat_chord_tone_ratio")),
                "offbeat_non_chord_ratio": _float(metrics.get("offbeat_non_chord_ratio")),
                "offbeat_non_chord_resolution_ratio": _float(
                    metrics.get("offbeat_non_chord_resolution_ratio")
                ),
                "offbeat_unresolved_non_chord_ratio": _float(
                    metrics.get("offbeat_unresolved_non_chord_ratio")
                ),
                "dominant_altered_offbeat_ratio": _float(metrics.get("dominant_altered_offbeat_ratio")),
                "enclosure_proxy_ratio": _float(metrics.get("enclosure_proxy_ratio")),
                "adjacent_repeat_ratio": _float(metrics.get("adjacent_repeat_ratio")),
                "interval_trigram_repeat_ratio": _float(metrics.get("interval_trigram_repeat_ratio")),
                "max_bar_pitch_class_jaccard": _float(metrics.get("max_bar_pitch_class_jaccard")),
                "unique_pitch_count": _int(metrics.get("unique_pitch_count")),
                "motion_balance_changed": bool(motion_repair.get("changed", False)),
                "motion_balance_step_count": _int(motion_repair.get("step_count")),
                "rhythm_articulation_accepted": bool(rhythm_repair.get("accepted", False)),
                "midi_path": str(item.get("midi_path") or ""),
                "context_midi_path": str(item.get("context_midi_path") or ""),
                "solo_wav": _wav_path(item, "solo_audio"),
                "context_wav": _wav_path(item, "context_audio"),
            }
        )
    return rows


def build_repair_summary(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "motion_balance_changed_candidates": sum(1 for item in candidate_rows if item["motion_balance_changed"]),
        "motion_balance_pitch_repair_steps": sum(_int(item["motion_balance_step_count"]) for item in candidate_rows),
        "rhythm_articulation_accepted_candidates": sum(
            1 for item in candidate_rows if item["rhythm_articulation_accepted"]
        ),
    }


def build_package_comparison(current: dict[str, Any], baseline: dict[str, Any] | None) -> list[dict[str, Any]]:
    if baseline is None:
        return []
    specs = [
        ("avg_step_motion_ratio", "higher"),
        ("avg_chromatic_step_ratio", "higher"),
        ("avg_large_leap_ratio", "lower"),
        ("avg_enclosure_proxy_ratio", "higher"),
        ("avg_offbeat_non_chord_ratio", "watch_increase"),
        ("avg_max_bar_pitch_class_jaccard", "watch_increase"),
        ("max_gate_penalty", "lower"),
        ("avg_adjacent_repeat_ratio", "lower"),
        ("avg_offbeat_non_chord_resolution_ratio", "higher"),
        ("avg_offbeat_unresolved_non_chord_ratio", "lower"),
    ]
    rows: list[dict[str, Any]] = []
    for key, direction in specs:
        current_value = _metric(current, key)
        baseline_value = _metric(baseline, key)
        delta = current_value - baseline_value
        if abs(delta) < 1e-12:
            status = "same"
        elif direction == "higher":
            status = "improved" if delta > 0 else "regressed"
        elif direction == "lower":
            status = "improved" if delta < 0 else "regressed"
        elif direction == "watch_increase":
            status = "tradeoff_watch" if delta > 0 else "lower_or_same"
        else:
            status = "observed"
        rows.append(
            {
                "metric": key,
                "current": current_value,
                "baseline": baseline_value,
                "delta": delta,
                "direction": direction,
                "status": status,
            }
        )
    return rows


def validate_review_readiness(
    package: dict[str, Any],
    *,
    selected: list[dict[str, Any]],
    note_review_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    listen_first = _dict(package.get("listen_first"))
    listen_first_files = [_dict(item) for item in _list(listen_first.get("files"))]
    solo_audio_count = sum(1 for item in selected if _path_exists(_wav_path(item, "solo_audio")))
    context_audio_count = sum(1 for item in selected if _path_exists(_wav_path(item, "context_audio")))
    listen_first_audio_count = 0
    for item in listen_first_files:
        if _path_exists(str(item.get("solo_wav") or "")) and _path_exists(str(item.get("context_wav") or "")):
            listen_first_audio_count += 1
    return {
        "selected_candidate_count": len(selected),
        "solo_audio_count": solo_audio_count,
        "context_audio_count": context_audio_count,
        "listen_first_case_count": len(listen_first_files),
        "listen_first_audio_pair_count": listen_first_audio_count,
        "note_review_validated": note_review_summary is not None,
        "quality_claimed": bool(package.get("quality_claimed", False)),
        "model_direct_claimed": bool(package.get("model_direct_claimed", False)),
        "review_ready": bool(selected)
        and solo_audio_count == len(selected)
        and context_audio_count == len(selected)
        and len(listen_first_files) > 0
        and listen_first_audio_count == len(listen_first_files)
        and note_review_summary is not None
        and not bool(package.get("quality_claimed", True))
        and not bool(package.get("model_direct_claimed", True)),
    }


def build_handoff(
    *,
    package_path: Path,
    output_dir: Path,
    expected_candidate_count: int,
    baseline_package_path: Path | None = None,
    note_review_path: Path | None = None,
) -> dict[str, Any]:
    package = read_json(package_path)
    selected = validate_best_of_package(
        package,
        package_path=package_path,
        expected_candidate_count=expected_candidate_count,
    )
    baseline = read_json(baseline_package_path) if baseline_package_path is not None else None
    if baseline is not None:
        validate_best_of_package(
            baseline,
            package_path=baseline_package_path or Path(""),
            expected_candidate_count=expected_candidate_count,
        )
    note_review_summary = None
    if note_review_path is not None:
        note_review_summary = validate_note_review(
            read_json(note_review_path),
            package_path=package_path,
            expected_candidate_count=expected_candidate_count,
        )

    candidate_rows = build_candidate_rows(selected)
    handoff = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "boundary": BOUNDARY,
        "source_package": str(package_path),
        "source_package_id": _report_id(package_path),
        "baseline_package": str(baseline_package_path) if baseline_package_path is not None else "",
        "baseline_package_id": _report_id(baseline_package_path) if baseline_package_path is not None else "",
        "expected_candidate_count": int(expected_candidate_count),
        "aggregate": _dict(package.get("aggregate")),
        "baseline_comparison": build_package_comparison(package, baseline),
        "repair_summary": build_repair_summary(candidate_rows),
        "candidate_rows": candidate_rows,
        "note_review": note_review_summary or {},
        "review_readiness": validate_review_readiness(
            package,
            selected=selected,
            note_review_summary=note_review_summary,
        ),
        "quality_claimed": False,
        "model_direct_claimed": False,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
            "review_input_required_for_quality_claim": True,
        },
    }
    write_json(output_dir / "bebop_language_review_handoff.json", handoff)
    write_text(output_dir / "bebop_language_review_handoff.md", markdown_report(handoff))
    return handoff


def markdown_report(report: dict[str, Any]) -> str:
    readiness = _dict(report.get("review_readiness"))
    lines = [
        "# Stage B MIDI-to-Solo Bebop Language Review Handoff",
        "",
        f"- source package: `{report['source_package_id']}`",
        f"- baseline package: `{report['baseline_package_id'] or 'none'}`",
        f"- selected candidates: `{readiness['selected_candidate_count']}`",
        f"- solo/context WAV: `{readiness['solo_audio_count']} / {readiness['context_audio_count']}`",
        f"- listen-first audio pairs: `{readiness['listen_first_audio_pair_count']}`",
        f"- note review validated: `{_bool_token(readiness['note_review_validated'])}`",
        f"- review ready: `{_bool_token(readiness['review_ready'])}`",
        f"- quality claimed: `{_bool_token(report['quality_claimed'])}`",
        f"- model direct claimed: `{_bool_token(report['model_direct_claimed'])}`",
        "",
        "## Aggregate Comparison",
        "",
        "| metric | baseline | current | delta | status |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report.get("baseline_comparison", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["metric"]),
                    f"{float(item['baseline']):.4f}",
                    f"{float(item['current']):.4f}",
                    f"{float(item['delta']):+.4f}",
                    str(item["status"]),
                ]
            )
            + " |"
        )
    repair = _dict(report.get("repair_summary"))
    lines.extend(
        [
            "",
            "## Repair Summary",
            "",
            f"- motion balance changed candidates: `{repair['motion_balance_changed_candidates']}`",
            f"- motion balance pitch repair steps: `{repair['motion_balance_pitch_repair_steps']}`",
            f"- rhythm articulation accepted candidates: `{repair['rhythm_articulation_accepted_candidates']}`",
            "",
            "## Candidate Files",
            "",
            "| rank | case | score | gate | step | chromatic | large leap | offbeat | unresolved | bar sim | repair steps | solo WAV | context WAV |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for item in report["candidate_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["case_label"]),
                    f"{item['score']:.4f}",
                    f"{item['gate_penalty']:.4f}",
                    f"{item['step_motion_ratio']:.4f}",
                    f"{item['chromatic_step_ratio']:.4f}",
                    f"{item['large_leap_ratio']:.4f}",
                    f"{item['offbeat_non_chord_ratio']:.4f}",
                    f"{item['offbeat_unresolved_non_chord_ratio']:.4f}",
                    f"{item['max_bar_pitch_class_jaccard']:.4f}",
                    str(item["motion_balance_step_count"]),
                    item["solo_wav"],
                    item["context_wav"],
                ]
            )
            + " |"
        )
    decision = _dict(report.get("decision"))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- current boundary: `{decision['current_boundary']}`",
            f"- next boundary: `{decision['next_boundary']}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- review input required for quality claim: `{_bool_token(decision['review_input_required_for_quality_claim'])}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a bebop-language review handoff")
    parser.add_argument("--package", required=True)
    parser.add_argument("--baseline_package")
    parser.add_argument("--note_review")
    parser.add_argument("--expected_candidate_count", type=int, default=8)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_root) / str(args.run_id)
    report = build_handoff(
        package_path=Path(args.package),
        baseline_package_path=Path(args.baseline_package) if args.baseline_package else None,
        note_review_path=Path(args.note_review) if args.note_review else None,
        expected_candidate_count=int(args.expected_candidate_count),
        output_dir=output_dir,
    )
    readiness = _dict(report.get("review_readiness"))
    print(
        json.dumps(
            {
                "run_id": str(args.run_id),
                "review_ready": bool(readiness.get("review_ready", False)),
                "selected_candidate_count": _int(readiness.get("selected_candidate_count")),
                "solo_audio_count": _int(readiness.get("solo_audio_count")),
                "context_audio_count": _int(readiness.get("context_audio_count")),
                "motion_balance_pitch_repair_steps": _int(
                    _dict(report.get("repair_summary")).get("motion_balance_pitch_repair_steps")
                ),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
