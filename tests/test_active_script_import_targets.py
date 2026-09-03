"""Keep active script imports inside the active repository boundary."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE_DIRS = (ROOT_DIR / "scripts", ROOT_DIR / "inference", ROOT_DIR / "tests")


def imported_script_modules(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "scripts":
                modules.update(f"scripts.{alias.name}" for alias in node.names if alias.name != "*")
            else:
                modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return {module for module in modules if module.startswith("scripts.")}


def module_target_exists(module: str) -> bool:
    target = ROOT_DIR.joinpath(*module.split("."))
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


class ActiveScriptImportTargetsTest(unittest.TestCase):
    def test_active_script_import_targets_exist(self) -> None:
        missing: list[str] = []
        for source_dir in ACTIVE_SOURCE_DIRS:
            for source_path in sorted(source_dir.rglob("*.py")):
                for module in sorted(imported_script_modules(source_path)):
                    if not module_target_exists(module):
                        relative_source = source_path.relative_to(ROOT_DIR)
                        missing.append(f"{relative_source}: {module}")

        self.assertEqual([], missing, "missing active script imports:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
