from __future__ import annotations

import os
import sys
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "teleop_control_py"
sys.path.insert(0, str(PACKAGE_ROOT))
