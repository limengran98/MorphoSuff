import pandas as pd

from measurement_sufficiency.cli import main
from measurement_sufficiency.splits import make_gene_splits


def test_run_fold_filters_train_and_test_from_frozen_manifest(tmp_path):
    observations = pd.DataFrame(
        {
            "observation_id": [f"o{i}" for i in range(12)],
            "perturbation_id": [f"g{i // 3}" for i in range(12)],
            "screen_id": ["s"] * 12,
            "reporter_id": ["r"] * 12,
            "feature": [float(i) for i in range(12)],
        }
    )
    targets = pd.DataFrame(
        {
            "observation_id": observations.observation_id,
            "endpoint_id": ["e"] * 12,
            "y_true": [float(i) / 2 for i in range(12)],
        }
    )
    manifest = make_gene_splits(observations, n_folds=2, seed=3).assignments
    inputs_path, targets_path = tmp_path / "inputs.csv", tmp_path / "targets.csv"
    manifest_path, output_path = tmp_path / "split.csv", tmp_path / "predictions.csv"
    observations.to_csv(inputs_path, index=False)
    targets.to_csv(targets_path, index=False)
    manifest.to_csv(manifest_path, index=False)
    assert main(
        [
            "run-fold",
            "--inputs", str(inputs_path),
            "--targets", str(targets_path),
            "--assignments", str(manifest_path),
            "--split-name", "gene",
            "--fold", "0",
            "--model", "ridge",
            "--features", "feature",
            "--output", str(output_path),
        ]
    ) == 0
    predictions = pd.read_csv(output_path)
    expected = set(manifest.loc[manifest.fold.eq(0) & manifest.role.eq("test"), "observation_id"])
    assert set(predictions.observation_id) == expected
    assert predictions.model_id.eq("ridge").all()
