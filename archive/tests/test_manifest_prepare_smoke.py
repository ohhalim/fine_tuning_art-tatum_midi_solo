from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_manifest_prepare_smoke import read_manifest_paths, write_limited_manifest


class ManifestPrepareSmokeTest(unittest.TestCase):
    def test_read_manifest_paths_skips_comments_and_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "manifest.txt"
            manifest.write_text(
                """
# comment
a.mid

b.midi
""".strip(),
                encoding="utf-8",
            )

            paths = read_manifest_paths(manifest)

        self.assertEqual(paths, ["a.mid", "b.midi"])

    def test_write_limited_manifest_keeps_requested_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            source = root / "source.txt"
            target = root / "nested" / "target.txt"
            source.write_text("a.mid\nb.mid\nc.mid\n", encoding="utf-8")

            count = write_limited_manifest(source, target, 2)

            self.assertEqual(count, 2)
            self.assertEqual(target.read_text(encoding="utf-8"), "a.mid\nb.mid\n")


if __name__ == "__main__":
    unittest.main()
