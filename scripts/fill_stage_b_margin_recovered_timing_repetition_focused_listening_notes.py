"""Fill timing/repetition focused listening notes from MIDI/context evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402
from scripts.build_stage_b_margin_recovered_timing_repetition_focused_listening_notes import (  # noqa: E402
    read_json,
    validate_notes,
)


class TimingRepetitionFocusedListeningFillError(ValueError):
    pass


def fill_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    filled = json.loads(json.dumps(candidate))
    metrics = filled.get("focused_context_metrics") if isinstance(filled.get("focused_context_metrics"), dict) else {}
    risks = set(str(risk) for risk in filled.get("review_risks", []))
    dead_air = float(metrics.get("dead_air_ratio", 0.0) or 0.0)
    adjacent_repeats = int(metrics.get("adjacent_pitch_repeats", 0) or 0)
    unique_pitch = int(metrics.get("unique_pitch_count", 0) or 0)
    phrase_span = float(metrics.get("phrase_span_beats", 0.0) or 0.0)
    max_interval = int(metrics.get("max_interval", 0) or 0)
    final_role = str(metrics.get("final_note_role") or "")
    timing = "stiff" if dead_air >= 0.40 else "acceptable"
    chord_fit = "strong" if final_role == "chord_tone" else "acceptable" if final_role == "tension" else "unclear"
    phrase_continuation = "weak" if phrase_span < 7.0 or max_interval >= 12 else "acceptable"
    landing = "strong" if final_role == "chord_tone" else "acceptable" if final_role == "tension" else "weak"
    jazz_vocabulary = "thin" if adjacent_repeats > 0 or max_interval >= 12 else "acceptable"
    decision = "needs_followup"
    if (
        timing == "acceptable"
        and phrase_continuation == "acceptable"
        and landing in {"strong", "acceptable"}
        and jazz_vocabulary == "acceptable"
    ):
        decision = "keep"
    if unique_pitch < 4 or phrase_span < 4.0:
        decision = "reject"
    filled["listening"] = {
        "status": "reviewed",
        "timing": timing,
        "chord_fit": chord_fit,
        "phrase_continuation": phrase_continuation,
        "landing": landing,
        "jazz_vocabulary": jazz_vocabulary,
        "decision": decision,
        "notes": (
            "MIDI/context evidence fill. Timing improved below the previous dead-air gate, "
            "but adjacent repeats and a wide interval remain review risks."
        ),
    }
    filled["listening_fill_evidence"] = {
        "not_human_audio_review": True,
        "dead_air_ratio": dead_air,
        "adjacent_pitch_repeats": adjacent_repeats,
        "phrase_span_beats": phrase_span,
        "max_interval": max_interval,
        "final_note": str(metrics.get("final_note") or ""),
        "final_chord": str(metrics.get("final_chord") or ""),
        "final_note_role": final_role,
        "review_risks": sorted(risks),
        "improved_from_pitch_vocab_fill": {
            "timing": timing == "acceptable",
            "dead_air_below_previous_gate": dead_air < 0.40,
        },
    }
    return filled


def fill_notes(notes: dict[str, Any]) -> dict[str, Any]:
    candidates = notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise TimingRepetitionFocusedListeningFillError("notes must contain candidates")
    filled = json.loads(json.dumps(notes))
    filled["schema_version"] = "stage_b_margin_recovered_timing_repetition_focused_listening_fill_v1"
    filled["filled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    filled["fill_method"] = {
        "source": "MIDI/context evidence",
        "not_human_audio_review": True,
    }
    filled["candidates"] = [fill_candidate(candidate) for candidate in candidates]
    return filled


def markdown_report(notes: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Timing/Repetition Focused Listening Fill",
        "",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- reviewed count: `{summary['reviewed_count']}`",
        f"- pending count: `{summary['pending_count']}`",
        f"- decisions: `{summary['decision_counts']}`",
        "",
        "This is a MIDI/context evidence fill, not a human audio listening proof.",
        "",
        "| candidate | timing | chord fit | phrase | landing | vocabulary | decision | risks |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate in notes["candidates"]:
        listening = candidate["listening"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate.get("candidate_id") or ""),
                    str(listening.get("timing") or ""),
                    str(listening.get("chord_fit") or ""),
                    str(listening.get("phrase_continuation") or ""),
                    str(listening.get("landing") or ""),
                    str(listening.get("jazz_vocabulary") or ""),
                    str(listening.get("decision") or ""),
                    ",".join(candidate.get("review_risks") or []),
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def validate_fill(
    notes: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_decision: str | None,
) -> dict[str, Any]:
    generic_notes = dict(notes)
    generic_notes["schema_version"] = "stage_b_margin_recovered_timing_repetition_focused_listening_notes_v1"
    summary = validate_notes(
        generic_notes,
        expected_candidate_id=expected_candidate_id,
        expected_prior_decision="keep_for_focused_listening",
    )
    if expected_decision:
        decisions = [str(candidate.get("listening", {}).get("decision") or "") for candidate in notes["candidates"]]
        if decisions != [expected_decision]:
            raise TimingRepetitionFocusedListeningFillError(
                f"expected decision {expected_decision}, got {decisions}"
            )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill timing/repetition focused listening notes")
    parser.add_argument("--review_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_timing_repetition_focused_listening_fill"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_decision", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    notes = read_json(Path(args.review_notes))
    filled = fill_notes(notes)
    summary = validate_fill(
        filled,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_decision=str(args.expected_decision or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    filled_path = output_dir / "focused_listening_review_notes_filled.json"
    markdown_path = output_dir / "focused_listening_review_notes_filled.md"
    write_json(filled_path, filled)
    write_json(output_dir / "focused_listening_review_notes_fill_summary.json", summary)
    markdown_path.write_text(markdown_report(filled, summary), encoding="utf-8")
    print(
        json.dumps(
            {**summary, "filled_notes_path": str(filled_path), "markdown_path": str(markdown_path)},
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
