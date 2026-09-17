#!/usr/bin/env python3
"""Panel-local entry point for the canonical Figure 5e renderer."""

from __future__ import annotations

import sys
from pathlib import Path


FIGURE5_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FIGURE5_ROOT / "code"))

from build_panel_e import main  # noqa: E402


if __name__ == "__main__":
    main()
