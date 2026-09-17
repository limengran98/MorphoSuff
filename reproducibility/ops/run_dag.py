#!/usr/bin/env python3
"""Print the OPS reproducibility DAG; execution belongs to the toolkit CLI."""
from __future__ import annotations

DAG = (
    ("acquire", "validate_acquisition"),
    ("validate_acquisition", "prepare_canonical_tables"),
    ("prepare_canonical_tables", "validate_pairing_and_schema"),
    ("validate_pairing_and_schema", "freeze_splits_and_sampling"),
    ("freeze_splits_and_sampling", "run_full_label_or_protocol"),
    ("run_full_label_or_protocol", "export_prediction_contract"),
    ("export_prediction_contract", "derive_metrics_and_sufficiency"),
    ("derive_metrics_and_sufficiency", "validate_figure_sources"),
    ("validate_figure_sources", "render_figure"),
)

if __name__ == "__main__":
    for source, target in DAG:
        print(f"{source} -> {target}")
