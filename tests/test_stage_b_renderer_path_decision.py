from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_renderer_path import (
    StageBRendererPathDecisionError,
    build_renderer_path_decision,
    validate_renderer_path_decision,
)


def tooling_report(status: str = "renderer_unavailable") -> dict:
    return {
        "schema_version": "stage_b_local_audio_render_tooling_readiness_v1",
        "tooling_status": status,
        "renderer_probe": {
            "selected_renderer_name": "",
            "fluidsynth_path": "",
            "timidity_path": "",
            "soundfont_path": "",
            "soundfont_exists": False,
        },
        "setup_boundary": {
            "system_modified": False,
            "package_install_executed": False,
            "download_executed": False,
            "generated_audio_created": False,
            "audio_render_attempted": False,
        },
    }


class StageBRendererPathDecisionTest(unittest.TestCase):
    def test_marks_renderer_unavailable_as_critical_user_input(self) -> None:
        report = build_renderer_path_decision(
            tooling_report(),
            output_dir=Path("outputs/renderer_path_decision"),
        )
        summary = validate_renderer_path_decision(
            report,
            expected_decision="renderer_path_or_install_approval_required",
            require_no_execution=True,
        )

        self.assertEqual(summary["tooling_status"], "renderer_unavailable")
        self.assertEqual(summary["blocked_reason"], "renderer_unavailable")
        self.assertTrue(summary["critical_user_input_required"])
        self.assertIn("package_install", report["not_executed"])
        self.assertIn("audio_rendered_quality", report["not_proven"])

    def test_marks_ready_status_as_render_attempt_ready(self) -> None:
        ready = tooling_report("ready_for_local_render")
        ready["renderer_probe"]["selected_renderer_name"] = "fluidsynth"
        ready["renderer_probe"]["fluidsynth_path"] = "/usr/local/bin/fluidsynth"
        ready["renderer_probe"]["soundfont_path"] = "/tmp/piano.sf2"
        ready["renderer_probe"]["soundfont_exists"] = True
        report = build_renderer_path_decision(ready, output_dir=Path("outputs/renderer_path_decision"))
        summary = validate_renderer_path_decision(
            report,
            expected_decision="ready_for_local_audio_render_attempt",
            require_no_execution=True,
        )

        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_tooling_report_with_install_execution(self) -> None:
        report = tooling_report()
        report["setup_boundary"]["package_install_executed"] = True

        with self.assertRaises(StageBRendererPathDecisionError):
            build_renderer_path_decision(report, output_dir=Path("outputs/renderer_path_decision"))


if __name__ == "__main__":
    unittest.main()
