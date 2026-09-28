"""Preset launcher builds the runtime command it documents (docs/PERSONALIZATION_STATUS.md)."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts.play_personalized import CHECKPOINTS, build_command, main

ROOT = Path("/r")


def cmd(preset, **kw):
    return build_command(preset, bars=16, bpm=128, chords="Dm7,G7", seed=42,
                         output_dir=Path("/o"), python="py", root=ROOT, **kw)


class PresetTest(unittest.TestCase):
    def test_single_presets_pick_their_checkpoint(self) -> None:
        for preset in ("tatum", "mehldau", "base"):
            c = cmd(preset)
            self.assertEqual(c[c.index("--checkpoint") + 1], str(ROOT / CHECKPOINTS[preset]))
            self.assertNotIn("--swap-adapter", c)
            self.assertIn("--chord-primer", c)
            self.assertIn("--half-bar-blocks", c)

    def test_swap_schedule_and_live_select(self) -> None:
        c = cmd("swap")
        self.assertEqual(c[c.index("--checkpoint") + 1], str(ROOT / CHECKPOINTS["tatum"]))
        self.assertIn(f"mehldau={ROOT / CHECKPOINTS['mehldau']}", c)
        self.assertIn("--allow-different-bases", c)
        self.assertEqual(c[c.index("--adapter-schedule") + 1], "tatum:4,mehldau:4")
        live = cmd("swap", live_select=True, input_port="KB")
        self.assertIn("--adapter-control", live)
        self.assertNotIn("--adapter-schedule", live)
        self.assertEqual(live[live.index("--input-port") + 1], "KB")

    def test_live_select_rules(self) -> None:
        with self.assertRaises(ValueError):
            cmd("tatum", live_select=True, input_port="KB")
        with self.assertRaises(ValueError):
            cmd("swap", live_select=True)

    def test_extra_flags_pass_through(self) -> None:
        self.assertEqual(cmd("tatum", extra=["--", "--no-kv-cache"])[-1], "--no-kv-cache")

    def test_dry_run_prints_without_running(self) -> None:
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(main(["--preset", "swap", "--dry-run"]), 0)
        self.assertIn("--adapter-schedule tatum:4,mehldau:4", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
