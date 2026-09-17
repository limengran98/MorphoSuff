#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

builder = Path(__file__).resolve().parents[2] / "code/build_figure3.py"
sys.argv = [str(builder), "--panel", "b"]
runpy.run_path(str(builder), run_name="__main__")
