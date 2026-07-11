from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_music_transformer_solo_yield_final_status import (
    SoloYieldFinalStatusAuditError,
    build_audit_report,
    validate_audit_report,
)


def repair_sweep(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_sweep_v1",
        "output_dir": "outputs/repair_sweep",
        "aggregate": {
            "case_count": 4,
            "sample_count": 24,
            "strict_valid_sample_count": 22,
            "grammar_gate_sample_count": 24,
            "strict_yield_rate": 0.9167,
            "min_case_strict_yield_rate": 0.8333,
            "rendered_audio_file_count": 8,
        },
        "readiness": {
            "music_transformer_checkpoint_generation_used": True,
            "constrained_decoding_used": True,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def repaired_package() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_package_v1",
        "output_dir": "outputs/repaired_package",
        "candidate_count": 8,
        "readiness": {
            "candidate_midi_files_copied": 8,
            "candidate_wav_files_copied": 8,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def repaired_guard(*, preference_fill_allowed: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_listening_input_guard_v1",
        "output_dir": "outputs/repaired_guard",
        "input_validation": {
            "validated_listening_input_present": preference_fill_allowed,
        },
        "readiness": {
            "preference_fill_allowed": preference_fill_allowed,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def repaired_objective() -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_objective_next_decision_v1",
        "output_dir": "outputs/repaired_objective",
        "objective_summary": {
            "candidate_count": 8,
            "dead_air_min": 0.5152,
            "dead_air_max": 0.7241,
        },
        "readiness": {
            "selected_objective_candidate_count": 4,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def final_handoff(*, missing_file_count: int = 0) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_broader_repaired_final_review_handoff_v1",
        "output_dir": "outputs/final_handoff",
        "aggregate": {
            "candidate_count": 8,
            "midi_count": 8,
            "wav_count": 8,
            "selected_objective_candidate_count": 4,
            "missing_file_count": missing_file_count,
            "checksum_mismatch_count": 0,
        },
        "readiness": {
            "final_review_handoff_ready": True,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


def handoff_audit(*, reproducible: bool = True) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_broader_repaired_handoff_reproducibility_audit_v1",
        "output_dir": "outputs/handoff_audit",
        "aggregate": {
            "candidate_count": 8,
            "selected_objective_candidate_count": 4,
            "missing_midi_count": 0,
            "missing_wav_count": 0,
            "midi_checksum_mismatch_count": 0,
            "wav_checksum_mismatch_count": 0,
            "reproducible_handoff": reproducible,
        },
        "readiness": {
            "reproducibility_audit_completed": True,
            "reproducible_handoff": reproducible,
            "validated_listening_input_present": False,
            "preference_fill_allowed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldFinalStatusAuditTest(unittest.TestCase):
    def test_builds_final_status_audit_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_audit_report(
                repair_sweep=repair_sweep(),
                repaired_package=repaired_package(),
                repaired_guard=repaired_guard(),
                repaired_objective=repaired_objective(),
                output_dir=Path(temp_dir) / "audit",
            )
        summary = validate_audit_report(report)

        self.assertTrue(summary["technical_mvp_evidence_ready"])
        self.assertEqual(summary["strict_valid_sample_count"], 22)
        self.assertEqual(summary["grammar_gate_sample_count"], 24)
        self.assertEqual(summary["rendered_audio_file_count"], 8)
        self.assertEqual(summary["repaired_candidate_count"], 8)
        self.assertEqual(summary["selected_objective_candidate_count"], 4)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], "music_transformer_solo_yield_readme_final_evidence_refresh")

    def test_includes_final_handoff_reproducibility_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_audit_report(
                repair_sweep=repair_sweep(),
                repaired_package=repaired_package(),
                repaired_guard=repaired_guard(),
                repaired_objective=repaired_objective(),
                final_handoff=final_handoff(),
                handoff_audit=handoff_audit(),
                output_dir=Path(temp_dir) / "audit",
            )
        summary = validate_audit_report(report)

        self.assertTrue(summary["technical_mvp_evidence_ready"])
        self.assertTrue(summary["final_review_handoff_present"])
        self.assertTrue(summary["final_review_handoff_ready"])
        self.assertEqual(summary["final_handoff_candidate_count"], 8)
        self.assertEqual(summary["final_handoff_midi_count"], 8)
        self.assertEqual(summary["final_handoff_wav_count"], 8)
        self.assertEqual(summary["final_handoff_missing_file_count"], 0)
        self.assertEqual(summary["final_handoff_checksum_mismatch_count"], 0)
        self.assertTrue(summary["handoff_reproducibility_audit_present"])
        self.assertTrue(summary["handoff_reproducibility_audit_completed"])
        self.assertTrue(summary["reproducible_handoff"])
        self.assertEqual(summary["handoff_missing_midi_count"], 0)
        self.assertEqual(summary["handoff_missing_wav_count"], 0)
        self.assertEqual(summary["handoff_midi_checksum_mismatch_count"], 0)
        self.assertEqual(summary["handoff_wav_checksum_mismatch_count"], 0)

    def test_handoff_mismatch_keeps_technical_ready_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_audit_report(
                repair_sweep=repair_sweep(),
                repaired_package=repaired_package(),
                repaired_guard=repaired_guard(),
                repaired_objective=repaired_objective(),
                final_handoff=final_handoff(missing_file_count=1),
                handoff_audit=handoff_audit(),
                output_dir=Path(temp_dir) / "audit",
            )
        summary = validate_audit_report(report)

        self.assertFalse(summary["technical_mvp_evidence_ready"])
        self.assertEqual(summary["final_handoff_missing_file_count"], 1)

    def test_non_reproducible_handoff_keeps_technical_ready_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_audit_report(
                repair_sweep=repair_sweep(),
                repaired_package=repaired_package(),
                repaired_guard=repaired_guard(),
                repaired_objective=repaired_objective(),
                final_handoff=final_handoff(),
                handoff_audit=handoff_audit(reproducible=False),
                output_dir=Path(temp_dir) / "audit",
            )
        summary = validate_audit_report(report)

        self.assertFalse(summary["technical_mvp_evidence_ready"])
        self.assertFalse(summary["reproducible_handoff"])

    def test_rejects_quality_claim_in_source_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(SoloYieldFinalStatusAuditError):
                build_audit_report(
                    repair_sweep=repair_sweep(quality_claim=True),
                    repaired_package=repaired_package(),
                    repaired_guard=repaired_guard(),
                    repaired_objective=repaired_objective(),
                    output_dir=Path(temp_dir) / "audit",
                )

    def test_preference_fill_keeps_technical_ready_false(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = build_audit_report(
                repair_sweep=repair_sweep(),
                repaired_package=repaired_package(),
                repaired_guard=repaired_guard(preference_fill_allowed=True),
                repaired_objective=repaired_objective(),
                output_dir=Path(temp_dir) / "audit",
            )
        summary = validate_audit_report(report)

        self.assertFalse(summary["technical_mvp_evidence_ready"])
        self.assertTrue(summary["validated_listening_input_present"])
        self.assertTrue(summary["preference_fill_allowed"])


if __name__ == "__main__":
    unittest.main()
