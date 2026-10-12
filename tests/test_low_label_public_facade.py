import argparse
import importlib.util
from pathlib import Path


FACADE_PATH = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "ops"
    / "runners"
    / "low_label"
    / "run.py"
)


def _facade():
    spec = importlib.util.spec_from_file_location("ops_low_label_public_facade", FACADE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, method: str) -> argparse.Namespace:
    return argparse.Namespace(method=method, sampling_root=tmp_path / "sampling")


def _common_assets(tmp_path: Path) -> dict[str, Path]:
    module = _facade()
    assets = {}
    for key in module.COMMON_KEYS:
        path = tmp_path / key
        path.mkdir()
        assets[key] = path
    return assets


def test_classical_method_ignores_unrelated_optional_assets(tmp_path):
    module = _facade()
    assets = _common_assets(tmp_path)
    sampling = tmp_path / "sampling"
    sampling.mkdir()
    assets["tabm_source"] = tmp_path / "does-not-exist-tabm.py"
    assets["strict_plan"] = tmp_path / "does-not-exist-strict-plan.json"
    assert module.missing_requirements(_args(tmp_path, "catboost"), assets) == []


def test_sampling_root_is_required_for_every_method(tmp_path):
    module = _facade()
    assets = _common_assets(tmp_path)
    missing = module.missing_requirements(_args(tmp_path, "ridge"), assets)
    assert missing == [str(tmp_path / "sampling")]


def test_tabm_checks_only_its_pinned_source(tmp_path):
    module = _facade()
    assets = _common_assets(tmp_path)
    (tmp_path / "sampling").mkdir()
    missing = module.missing_requirements(_args(tmp_path, "tabm"), assets)
    assert missing == ["assets:tabm_source"]
    assets["tabm_source"] = tmp_path / "tabm.py"
    missing = module.missing_requirements(_args(tmp_path, "tabm"), assets)
    assert missing == [str(tmp_path / "tabm.py")]
