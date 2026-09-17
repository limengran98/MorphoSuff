#!/usr/bin/env python3
"""Rebuild the canonical five-panel Figure 1 in dependency order."""

from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "a_workflow/code/build_figure1a.py",
    ROOT / "b_exact_linkage_scale/code/build_figure1b.py",
    ROOT / "c_sparse_assay_topology/code/build_figure1c_assay_topology.py",
    ROOT / "d_ko_response_atlas/code/build_figure1d.py",
    ROOT / "e_same_cell_response_maps/code/build_figure1e.py",
    ROOT / "code/build_figure1_composite.py",
]


def main() -> None:
    env = os.environ.copy()
    mplconfig = Path(tempfile.gettempdir()) / "mplconfig_morphosuff_figure1"
    mplconfig.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mplconfig))
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    for script in SCRIPTS:
        subprocess.run([sys.executable, str(script)], check=True, env=env)
    (ROOT / "published").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "composite/Figure1.pdf", ROOT / "published/Figure1.pdf")


if __name__ == "__main__":
    main()
