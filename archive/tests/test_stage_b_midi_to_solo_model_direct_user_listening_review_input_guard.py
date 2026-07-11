from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.guard_stage_b_midi_to_solo_model_direct_user_listening_review_input import (
    BOUNDARY,
    FILL_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    StageBMidiToSoloModelDirectUserListeningReviewInputGuardError,
    build_user_listening_review_input_guard_report,
    parse_review_input,
    validate_user_listening_review_input_guard_report,
)


PENDING_TEMPLATE = """# Stage B MIDI-to-Solo Model-Direct Listening Review Input

## Review Status

- reviewer: `pending`
- reviewed_at: `pending`
- preferred_rank: `pending`
- reject_all: `pending`
- broad_model_quality_claim_allowed: `false`

## Candidates

| rank | midi | wav | note count | unique pitch | max interval | dead-air ratio | decision |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | a.mid | a.wav | 32 | 11 | 9 | 0.2258 | `pending` |
| 2 | b.mid | b.wav | 32 | 15 | 9 | 0.2258 | `pending` |
| 3 | c.mid | c.wav | 32 | 14 | 8 | 0.2258 | `pending` |

## Per-Candidate Notes

### Rank 1

- musical_acceptance: `pending`
- issue_tags: `pending`
- short_note: `pending`
"""


FILLED_TEMPLATE = PENDING_TEMPLATE.replace("`pending`", "`reviewed`", 2).replace(
    "- preferred_rank: `pending`", "- preferred_rank: `1`"
).replace("- reject_all: `pending`", "- reject_all: `false`").replace(
    "| `pending` |", "| `accept` |"
).replace(
    "- musical_acceptance: `pending`", "- musical_acceptance: `accept`"
).replace(
    "- issue_tags: `pending`", "- issue_tags: `none`"
).replace(
    "- short_note: `pending`", "- short_note: `usable timing`"
)


def package_report(template_path: Path, *, preference_claim: bool = False) -> dict:
    return {
        "listening_review_package_boundary": {
            "boundary": "stage_b_midi_to_solo_model_direct_listening_review_package",
            "source_boundary": "stage_b_midi_to_solo_model_direct_timing_phrase_repair",
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "review_input_template_written": True,
            "human_audio_preference_claimed": preference_claim,
        },
        "decision": {
            "next_boundary": "stage_b_midi_to_solo_model_direct_user_listening_review_fill",
            "critical_user_input_required": False,
        },
        "review_input_template_path": str(template_path),
    }


class StageBMidiToSoloModelDirectUserListeningReviewInputGuardTest(unittest.TestCase):
    def test_parse_pending_review_input(self) -> None:
        parsed = parse_review_input(PENDING_TEMPLATE)

        self.assertFalse(parsed["validated_review_input_present"])
        self.assertIn("reviewer", parsed["pending_status_fields"])
        self.assertEqual(parsed["pending_candidate_decisions"], ["1", "2", "3"])
        self.assertIn("rank_1.musical_acceptance", parsed["pending_candidate_fields"])

    def test_blocks_preference_fill_when_input_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "review.md"
            template_path.write_text(PENDING_TEMPLATE, encoding="utf-8")
            report = build_user_listening_review_input_guard_report(
                package_report(template_path),
                output_dir=root / "out",
            )
            summary = validate_user_listening_review_input_guard_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
                require_guard_completed=True,
                require_pending_input=True,
                require_no_quality_claim=True,
            )

        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertGreater(summary["pending_status_field_count"], 0)
        self.assertGreater(summary["pending_candidate_decision_count"], 0)
        self.assertFalse(summary["human_audio_preference_claimed"])

    def test_routes_to_fill_when_input_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "review.md"
            template_path.write_text(FILLED_TEMPLATE, encoding="utf-8")
            report = build_user_listening_review_input_guard_report(
                package_report(template_path),
                output_dir=root / "out",
            )
            summary = validate_user_listening_review_input_guard_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=FILL_BOUNDARY,
                require_guard_completed=True,
                require_pending_input=False,
                require_no_quality_claim=True,
            )

        self.assertTrue(summary["validated_review_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])

    def test_rejects_upstream_preference_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template_path = root / "review.md"
            template_path.write_text(PENDING_TEMPLATE, encoding="utf-8")
            with self.assertRaises(StageBMidiToSoloModelDirectUserListeningReviewInputGuardError):
                build_user_listening_review_input_guard_report(
                    package_report(template_path, preference_claim=True),
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
