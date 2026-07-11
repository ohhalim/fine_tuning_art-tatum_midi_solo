from __future__ import annotations

import unittest
from pathlib import Path

from scripts.define_stage_b_midi_to_solo_mvp_contract import (
    BOUNDARY,
    NEXT_BOUNDARY,
    TARGET_DATE,
    StageBMidiToSoloMvpContractError,
    build_contract_report,
    validate_contract_report,
)


class StageBMidiToSoloMvpContractTest(unittest.TestCase):
    def test_contract_defines_midi_to_solo_hybrid_mvp(self) -> None:
        report = build_contract_report(output_dir=Path("outputs/midi_to_solo"), issue_number=481)
        summary = validate_contract_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_fallback=True,
            require_no_final_claim=True,
        )

        self.assertEqual(summary["target_date"], TARGET_DATE)
        self.assertEqual(summary["candidate_count"], 32)
        self.assertEqual(summary["export_top_midi_count"], 3)
        self.assertEqual(summary["target_solo_bars"], 8)
        self.assertEqual(summary["max_simultaneous_notes"], 1)
        self.assertEqual(summary["fallback_path"], "phrase_retrieval_data_motif_hybrid")
        self.assertFalse(summary["midi_to_solo_mvp_claimed"])
        self.assertFalse(summary["brad_style_fine_tuning_completed"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_missing_fallback_when_required(self) -> None:
        report = build_contract_report(output_dir=Path("outputs/midi_to_solo"), issue_number=481)
        report["generation_stack"]["fallback_path"] = ""

        with self.assertRaises(StageBMidiToSoloMvpContractError):
            validate_contract_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_fallback=True,
                require_no_final_claim=True,
            )

    def test_rejects_too_few_exported_candidates(self) -> None:
        report = build_contract_report(output_dir=Path("outputs/midi_to_solo"), issue_number=481)
        report["output_contract"]["export_top_midi_count"] = 2

        with self.assertRaises(StageBMidiToSoloMvpContractError):
            validate_contract_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_fallback=True,
                require_no_final_claim=True,
            )

    def test_rejects_polyphonic_solo_gate(self) -> None:
        report = build_contract_report(output_dir=Path("outputs/midi_to_solo"), issue_number=481)
        report["objective_gate"]["max_simultaneous_notes"] = 2

        with self.assertRaises(StageBMidiToSoloMvpContractError):
            validate_contract_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_fallback=True,
                require_no_final_claim=True,
            )

    def test_rejects_final_quality_claims(self) -> None:
        report = build_contract_report(output_dir=Path("outputs/midi_to_solo"), issue_number=481)
        report["claim_boundary"]["brad_style_fine_tuning_completed"] = True

        with self.assertRaises(StageBMidiToSoloMvpContractError):
            validate_contract_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_fallback=True,
                require_no_final_claim=True,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_input_contract")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_context_extraction_mvp")
        self.assertEqual(TARGET_DATE, "2026-06-11")


if __name__ == "__main__":
    unittest.main()
