from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "studies" / "ops" / "runners" / "raw_image"


# The raw-phase runners need SciPy at import time
# (studies/ops/runners/raw_image/common.py imports scipy.stats.rankdata) and
# PyTorch for the head fit and the fine-tuning backward pass.
@pytest.mark.requires_scipy
@pytest.mark.requires_torch
def test_raw_image_cpu_smoke() -> None:
    completed = subprocess.run(
        [sys.executable, str(RAW / "smoke.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "SMOKE_PASS"' in completed.stdout


@pytest.mark.requires_scipy
@pytest.mark.requires_torch
def test_raw_image_all_entrypoint_contracts() -> None:
    completed = subprocess.run(
        [sys.executable, str(RAW / "contract_smoke.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"status": "CONTRACT_SMOKE_PASS"' in completed.stdout
    assert '"entry_point_help_checks": 8' in completed.stdout
    assert '"finetune_forward_backward": 1' in completed.stdout


def test_raw_image_sources_have_no_private_path() -> None:
    forbidden = ("/" + "data" + "/", "/" + "home" + "/")
    for path in RAW.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".md", ".txt"}:
            text = path.read_text(encoding="utf-8")
            assert all(token not in text for token in forbidden)
