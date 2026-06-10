from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input import (
    BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError,
    build_listening_review_input_guard_report,
    validate_listening_review_input_guard_report,
)


def source_package(*, quality_claim: bool = False, validated_input: bool = False) -> dict:
    review_items = [
        {
            "candidate_index": index,
            "source": "unit_source",
            "rank": index,
            "midi_path": f"/tmp/candidate_{index}.mid",
            "wav_path": f"/tmp/candidate_{index}.wav",
            "duration_seconds": 0.1,
            "sample_rate": 44100,
            "size_bytes": 100,
            "sha256": "abc",
            "repaired_failure_labels": ["rhythmic_monotony"] if index == 4 else [],
            "review_status": "pending",
        }
        for index in range(1, 7)
    ]
    return {
        "boundary": SOURCE_BOUNDARY,
        "source_summary": {
            "rendered_audio_file_count": 6,
            "technical_wav_validation": True,
            "sample_rate": 44100,
            "duration_min_seconds": 0.1,
            "duration_max_seconds": 0.1,
            "failure_label_delta": 3,
            "source_phrase_rhythm_failure_count": 4,
            "repaired_phrase_rhythm_failure_count": 1,
            "phrase_rhythm_failure_delta": 3,
            "remaining_failure_counts": {
                "rhythmic_monotony": 1,
            },
            "source_outside_soloing_repair_evidence_ready": True,
            "source_outside_soloing_repair_pitch_role_risk_count_after": 0,
            "source_outside_soloing_not_evaluable_count": 6,
            "repaired_outside_soloing_not_evaluable_count": 6,
            "repaired_not_evaluable_counts": {
                "outside_soloing_without_context": 6,
                "weak_chord_tone_landing": 6,
            },
            "audio_review_required": True,
        },
        "review_package": {
            "package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "required_input_fields": [
                "candidate_index",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": review_items,
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardTest(unittest.TestCase):
    def test_blocks_preference_fill_when_review_input_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_listening_review_input_guard_report(
                source_package(),
                output_dir=root / "guard",
                issue_number=864,
            )
            summary = validate_listening_review_input_guard_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
                require_guard_completed=True,
                require_preference_blocked=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertEqual(summary["review_item_count"], 6)
            self.assertEqual(summary["required_input_field_count"], 4)
            self.assertEqual(summary["failure_label_delta"], 3)
            self.assertEqual(summary["phrase_rhythm_failure_delta"], 3)
            self.assertTrue(summary["source_outside_soloing_repair_evidence_ready"])
            self.assertEqual(
                summary["source_outside_soloing_repair_pitch_role_risk_count_after"],
                0,
            )
            self.assertEqual(summary["source_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(summary["repaired_outside_soloing_not_evaluable_count"], 6)
            self.assertEqual(
                summary["repaired_not_evaluable_counts"],
                {
                    "outside_soloing_without_context": 6,
                    "weak_chord_tone_landing": 6,
                },
            )
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source_package(quality_claim=True),
                    output_dir=root / "guard",
                    issue_number=864,
                )

    def test_rejects_missing_outside_soloing_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = source_package()
            del source["source_summary"]["repaired_outside_soloing_not_evaluable_count"]

            with self.assertRaises(StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError):
                build_listening_review_input_guard_report(
                    source,
                    output_dir=root / "guard",
                    issue_number=864,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard")
        self.assertEqual(OBJECTIVE_NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision")


if __name__ == "__main__":
    unittest.main()
