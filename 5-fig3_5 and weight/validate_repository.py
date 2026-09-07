"""Run lightweight checks on the publication-ready Python scripts."""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_PATTERN = "[0-9][0-9]_*.py"
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:\[A-Za-z0-9_]")


def main() -> None:
    scripts = sorted(ROOT.glob(SCRIPT_PATTERN))
    if not scripts:
        raise RuntimeError("No publication scripts were found.")

    failures: list[str] = []

    for script in scripts:
        source = script.read_text(encoding="utf-8")

        try:
            py_compile.compile(str(script), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"{script.name}: syntax/compile failure: {exc}")

        if CJK_PATTERN.search(source):
            failures.append(f"{script.name}: contains Chinese characters.")

        if WINDOWS_DRIVE_PATTERN.search(source):
            failures.append(f"{script.name}: contains a hard-coded Windows drive path.")

    if failures:
        details = "\n".join(f"- {item}" for item in failures)
        raise RuntimeError(f"Repository validation failed:\n{details}")

    print(f"Repository validation passed for {len(scripts)} publication scripts.")


if __name__ == "__main__":
    main()
