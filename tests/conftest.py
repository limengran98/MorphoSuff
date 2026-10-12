"""Optional-dependency gating for the test suite.

The dependency-minimal install (``pip install -e ".[test,figures]"``) cannot run
the tests that exercise PyTorch, CatBoost or SciPy code paths.  Those tests are
marked rather than deleted, and this hook skips them only when the dependency is
genuinely absent.

Setting ``MEASUREMENT_SUFFICIENCY_REQUIRE_OPTIONAL=1`` turns the gate off, so a
CI job that installs the optional extras fails loudly instead of silently
skipping.  Without that job, a marked test can rot indefinitely: nothing would
notice that it never runs anywhere.
"""

from __future__ import annotations

import importlib.util
import os

import pytest


OPTIONAL_MARKERS: dict[str, str] = {
    "requires_torch": "torch",
    "requires_catboost": "catboost",
    "requires_scipy": "scipy",
}

STRICT_ENV_VAR = "MEASUREMENT_SUFFICIENCY_REQUIRE_OPTIONAL"


def _is_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        return False


def _strict() -> bool:
    return os.environ.get(STRICT_ENV_VAR) == "1"


def pytest_configure(config: pytest.Config) -> None:
    if not _strict():
        return
    missing = sorted(
        module for module in set(OPTIONAL_MARKERS.values()) if not _is_installed(module)
    )
    if missing:
        raise pytest.UsageError(
            f"{STRICT_ENV_VAR}=1 requires every optional dependency to be present, "
            f"but these are missing: {', '.join(missing)}. Install with "
            'pip install -e ".[test,figures,models,ops-data]" or unset the variable.'
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _strict():
        return
    for item in items:
        for marker, module in OPTIONAL_MARKERS.items():
            if marker in item.keywords and not _is_installed(module):
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            f"optional dependency {module!r} is absent; set "
                            f"{STRICT_ENV_VAR}=1 in an environment that has it to "
                            "require this test to run"
                        )
                    )
                )
