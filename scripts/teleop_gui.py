#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Thin source-tree launcher for the packaged teleop GUI."""

import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
TELEOP_PKG_SRC = SCRIPT_PATH.parents[1] / "src" / "teleop_control_py"
if str(TELEOP_PKG_SRC) not in sys.path:
    sys.path.insert(0, str(TELEOP_PKG_SRC))

from teleop_control_py.gui.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
