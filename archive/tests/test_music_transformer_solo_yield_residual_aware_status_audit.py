from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.audit_music_transformer_solo_yield_residual_aware_status import (
    NEXT_BOUNDARY,
    SoloYieldResidualAwareStatusAuditError,
    build_status_audit_report,
    validate_status_audit_report,
)


def final_review_package(*, quality_claim: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_final_review_package_v1",
        "output_dir": "outputs/final_review",
        "aggregate": {
            "candidate_count": 8,
            "midi_count": 8,
            "wav_count": 8,
            "quality_proxy_pass_count": 6,
            "quality_proxy_fail_count": 2,
            "major_label_counts": {"low_tension_color": 2},
            "watch_label_counts": {"dead_air_watch": 3},
            "checksum_mismatch_count": 0,
            "missing_file_count": 0,
        },
        "readiness": {
            "residual_aware_final_review_package_ready": True,
            "review_input_template_written": True,
            "validated_listening_input_present": False,
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_listening_input_guard",
            "critical_user_input_required": False,
        },
    }


def input_guard(*, preference_fill: bool = False) -> dict:
    return {
        "schema_version": "music_transformer_solo_yield_residual_aware_listening_input_guard_v1",
        "output_dir": "outputs/input_guard",
        "source_package_summary": {
            "candidate_count": 8,
            "midi_count": 8,
            "wav_count": 8,
            "quality_proxy_pass_count": 6,
            "quality_proxy_fail_count": 2,
            "major_label_counts": {"low_tension_color": 2},
            "watch_label_counts": {"dead_air_watch": 3},
        },
        "guard_result": {
            "validated_listening_input_present": preference_fill,
            "preference_fill_allowed": preference_fill,
            "review_item_count": 8,
        },
        "readiness": {
            "residual_aware_listening_input_guard_completed": True,
            "validated_listening_input_present": preference_fill,
            "preference_fill_allowed": preference_fill,
            "listening_review_completed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": "music_transformer_solo_yield_residual_aware_status_sync",
            "critical_user_input_required": False,
        },
    }


def readme_text(*, missing_status: bool = False) -> str:
    lines = [
        "- 최신 review package: MIDI `8`, WAV `8`",
        "- objective rubric: pass/fail `6 / 2`",
        "- 남은 major label: `low_tension_color=2`",
        "- 남은 watch label: `dead_air_watch=3`",
        "- tension 추가 repair 가능성: current guard 기준 `false`",
        "- validated listening input: `false`",
        "- human/audio preference claim: `false`",
        "- musical quality claim: `false`",
        "- final review package: `residual_aware_final_review_package.md`",
        "- listening input guard doc: `residual_aware_listening_input_guard`",
    ]
    if missing_status:
        lines.remove("- musical quality claim: `false`")
    return "\n".join(lines)


def current_status_text() -> str:
    return "\n".join(
        [
            "- current issue: Issue #1394, Stage B MIDI-to-solo residual-aware status audit",
            "- residual-aware final review package candidate count: `8`",
            "- residual-aware final review MIDI/WAV: `8 / 8`",
            "- residual-aware objective rubric pass/fail: `6 / 2`",
            "- residual-aware residual major label: `low_tension_color=2`",
            "- residual-aware residual watch label: `dead_air_watch=3`",
            "- residual tension repeat feasible under current guard: `false`",
            "- residual-aware listening input guard validated input: `false`",
            "- residual-aware listening input guard preference fill: `false`",
            "- residual-aware next boundary: `music_transformer_solo_yield_residual_aware_status_audit`",
        ]
    )


class MusicTransformerSoloYieldResidualAwareStatusAuditTest(unittest.TestCase):
    def test_builds_status_audit_when_docs_match(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_status_audit_report(
                final_review_package=final_review_package(),
                input_guard_report=input_guard(),
                readme_text=readme_text(),
                current_status_text=current_status_text(),
                output_dir=Path(raw_temp) / "status_audit",
                issue_number=1394,
            )
        summary = validate_status_audit_report(
            report,
            expected_next_boundary=NEXT_BOUNDARY,
            require_docs_synced=True,
            require_pending_input=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["residual_aware_status_synced"])
        self.assertTrue(summary["readme_status_synced"])
        self.assertTrue(summary["current_status_synced"])
        self.assertEqual(summary["candidate_count"], 8)
        self.assertEqual(summary["quality_proxy_pass_count"], 6)
        self.assertFalse(summary["validated_listening_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_validation_rejects_missing_readme_status(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_status_audit_report(
                final_review_package=final_review_package(),
                input_guard_report=input_guard(),
                readme_text=readme_text(missing_status=True),
                current_status_text=current_status_text(),
                output_dir=Path(raw_temp) / "status_audit",
                issue_number=1394,
            )

        with self.assertRaises(SoloYieldResidualAwareStatusAuditError):
            validate_status_audit_report(
                report,
                expected_next_boundary=NEXT_BOUNDARY,
                require_docs_synced=True,
                require_pending_input=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim_in_final_review_package(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            with self.assertRaises(SoloYieldResidualAwareStatusAuditError):
                build_status_audit_report(
                    final_review_package=final_review_package(quality_claim=True),
                    input_guard_report=input_guard(),
                    readme_text=readme_text(),
                    current_status_text=current_status_text(),
                    output_dir=Path(raw_temp) / "status_audit",
                    issue_number=1394,
                )

    def test_validation_rejects_preference_fill_when_pending_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_temp:
            report = build_status_audit_report(
                final_review_package=final_review_package(),
                input_guard_report=input_guard(preference_fill=True),
                readme_text=readme_text(),
                current_status_text=current_status_text(),
                output_dir=Path(raw_temp) / "status_audit",
                issue_number=1394,
            )

        with self.assertRaises(SoloYieldResidualAwareStatusAuditError):
            validate_status_audit_report(
                report,
                require_docs_synced=True,
                require_pending_input=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
