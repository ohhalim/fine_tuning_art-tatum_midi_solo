from __future__ import annotations

import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.check_stage_b_local_audio_render_tooling import (
    StageBLocalAudioRenderToolingError,
    build_tooling_report,
    validate_tooling_report,
)


def fake_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class StageBLocalAudioRenderToolingTest(unittest.TestCase):
    def test_reports_renderer_unavailable_without_system_changes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = build_tooling_report(
                output_dir=Path(temp_dir) / "tooling",
                renderer_paths={"fluidsynth": "", "timidity": ""},
            )
            summary = validate_tooling_report(
                report,
                expected_status="renderer_unavailable",
                require_no_system_modification=True,
            )

            self.assertEqual(summary["tooling_status"], "renderer_unavailable")
            self.assertFalse(summary["system_modified"])
            self.assertFalse(summary["audio_render_attempted"])
            self.assertTrue(report["install_guidance"])

    def test_reports_soundfont_missing_for_fluidsynth_without_soundfont(self) -> None:
        with TemporaryDirectory() as temp_dir:
            renderer = Path(temp_dir) / "fluidsynth"
            fake_executable(renderer)
            report = build_tooling_report(
                output_dir=Path(temp_dir) / "tooling",
                requested_renderer="fluidsynth",
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_tooling_report(
                report,
                expected_status="soundfont_missing",
                require_no_system_modification=True,
            )

            self.assertEqual(summary["selected_renderer_name"], "fluidsynth")
            self.assertFalse(summary["soundfont_exists"])

    def test_reports_ready_when_renderer_and_soundfont_exist(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            renderer = root / "fluidsynth"
            soundfont = root / "piano.sf2"
            fake_executable(renderer)
            soundfont.write_bytes(b"sf2")
            report = build_tooling_report(
                output_dir=root / "tooling",
                requested_renderer="fluidsynth",
                soundfont_path=str(soundfont),
                renderer_paths={"fluidsynth": str(renderer), "timidity": ""},
            )
            summary = validate_tooling_report(
                report,
                expected_status="ready_for_local_render",
                require_no_system_modification=True,
            )

            self.assertEqual(summary["tooling_status"], "ready_for_local_render")
            self.assertTrue(summary["soundfont_exists"])

    def test_rejects_system_modification_claim(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = build_tooling_report(output_dir=Path(temp_dir) / "tooling")
            report["setup_boundary"]["system_modified"] = True

            with self.assertRaises(StageBLocalAudioRenderToolingError):
                validate_tooling_report(
                    report,
                    expected_status=None,
                    require_no_system_modification=True,
                )


if __name__ == "__main__":
    unittest.main()
