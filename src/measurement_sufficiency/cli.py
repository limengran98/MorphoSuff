"""Small portable CLI for validation, split manifests, and prediction summaries."""

from __future__ import annotations

import argparse
import pickle
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .metrics import cell_metrics, ko_metrics
from .schemas import validate_predictions, validate_targets
from .splits import LeakageError, canonical_split_name, check_split_leakage, make_field_splits, make_gene_splits, make_whole_screen_splits
from .training import fit_from_manifest, fit_sparse_model, prediction_long_table
from .adapters import LocalManifestAdapter


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _read_targets(path: str, observed_column: str = "is_observed") -> pd.DataFrame:
    """Read a long target table and validate it before anything consumes it.

    A bare read_csv turns a written ``False`` back into the string ``"False"``,
    which is truthy. Validating here means a malformed availability mask fails
    at the boundary rather than becoming a silent training label.
    """
    return validate_targets(pd.read_csv(path), observed_column=observed_column)


def _validate(args: argparse.Namespace) -> int:
    validate_predictions(_read_csv(args.predictions))
    print("valid predictions")
    return 0


def _validate_dataset(args: argparse.Namespace) -> int:
    adapter = LocalManifestAdapter(Path(args.manifest))
    observations = adapter.observations()
    reporters = adapter.reporters()
    assays = adapter.assays()
    pairing = adapter.pairing()
    payload = {
        "status": "PASS",
        "dataset_id": adapter.dataset_id,
        "n_observations": int(len(observations)),
        "n_reporters": int(len(reporters)),
        "n_assays": int(len(assays)),
        "n_pairings": int(len(pairing)),
        "input_schema": adapter.input_schema().schema_id,
        "capabilities": adapter.capabilities.__dict__,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _split(args: argparse.Namespace) -> int:
    observations = _read_csv(args.observations)
    kind = canonical_split_name(args.kind)
    builders = {"field": make_field_splits, "gene": make_gene_splits, "strict_whole_screen": make_whole_screen_splits}
    options: dict[str, object] = {"n_folds": args.folds, "seed": args.seed}
    control_column = getattr(args, "control_column", None)
    if control_column:
        # Only the held-out-perturbation split knows what to do with controls: it
        # spreads them over every fold so each fold's held-out role keeps a
        # same-screen baseline, and exempts them from the leakage check. Accepting
        # the flag on the other split kinds would imply a guarantee they do not make.
        if kind != "gene":
            raise SystemExit(
                f"--control-column applies to the gene split only, not {kind!r}. "
                "Controls are spread across folds so that each held-out role retains a "
                "same-screen baseline, which is a property of perturbation holdout."
            )
        options["control_column"] = control_column
    manifest = builders[kind](observations, **options)  # type: ignore[arg-type]
    manifest.assignments.to_csv(args.output, index=False)
    return 0


def _metrics(args: argparse.Namespace) -> int:
    predictions = _read_csv(args.predictions)
    result = cell_metrics(predictions) if args.level == "cell" else ko_metrics(predictions)
    result.to_csv(args.output, index=False)
    return 0


def _parse_features(value: str | None) -> list[str] | None:
    if value is None:
        return None
    columns = [column.strip() for column in value.split(",") if column.strip()]
    if not columns:
        raise ValueError("--features must name at least one comma-separated column")
    return columns


def _train(args: argparse.Namespace) -> int:
    """Fit only from supplied training rows and serialize the portable artifact."""
    fitted = fit_sparse_model(
        _read_csv(args.inputs),
        _read_targets(args.targets, args.observed_column),
        model_id=args.model,
        task_mode=args.task_mode,
        feature_columns=_parse_features(args.features),
        seed=args.seed,
    )
    with open(args.model_output, "wb") as handle:
        pickle.dump(fitted, handle)
    return 0


def _predict(args: argparse.Namespace) -> int:
    with open(args.model_input, "rb") as handle:
        fitted = pickle.load(handle)
    inputs = _read_csv(args.inputs)
    targets = _read_targets(args.targets, args.observed_column) if args.targets else None
    output = prediction_long_table(
        fitted,
        inputs,
        metadata=inputs,
        split_name=args.split_name,
        fold=args.fold,
        targets=targets,
        include_unobserved=args.include_unobserved,
    )
    output.to_csv(args.output, index=False)
    return 0


def _run_fold(args: argparse.Namespace) -> int:
    """Fit and evaluate one frozen fold without requiring manual row filtering."""
    inputs = _read_csv(args.inputs)
    targets = _read_targets(args.targets, args.observed_column)
    assignments = _read_csv(args.assignments)
    split_name = canonical_split_name(args.split_name)
    heldout = args.heldout_column or {
        "field": "field_id",
        "gene": "perturbation_id",
        "strict_whole_screen": "screen_id",
    }[split_name]
    if heldout not in inputs:
        raise ValueError(f"inputs lacks held-out grouping column {heldout!r}")
    # A gene split deliberately keeps declared controls in every fold so that a
    # control-relative response has a same-screen baseline; they are exempt from
    # the held-out-group check and excluded from every accuracy metric. The
    # exemption is only applied when the caller actually supplies the column.
    control_column = args.control_column
    exempt = control_column if (control_column in inputs and split_name == "gene") else None
    columns = ["observation_id", heldout] + ([exempt] if exempt else [])
    try:
        check_split_leakage(
            inputs[columns],
            assignments.loc[assignments.fold.eq(args.fold)],
            heldout,
            exempt_column=exempt,
        )
    except LeakageError:
        if split_name == "gene" and exempt is None:
            raise LeakageError(
                f"{heldout} leakage in fold {args.fold}. If the overlapping group is a "
                f"control population, supply its flag with --control-column so it can be "
                f"exempted explicitly: the column must be present in --inputs. Controls are "
                f"never inferred from a perturbation label."
            ) from None
        raise
    fitted = fit_from_manifest(
        inputs,
        targets,
        assignments,
        fold=args.fold,
        model_id=args.model,
        task_mode=args.task_mode,
        feature_columns=_parse_features(args.features),
        seed=args.seed,
    )
    test_ids = set(
        assignments.loc[
            assignments.fold.eq(args.fold) & assignments.role.eq("test"),
            "observation_id",
        ]
    )
    if not test_ids:
        raise ValueError(f"fold {args.fold} has no test observations")
    test_inputs = inputs.loc[inputs.observation_id.isin(test_ids)].copy()
    if len(test_inputs) != len(test_ids):
        raise ValueError("split assignment refers to test inputs that are absent")
    test_targets = targets.loc[targets.observation_id.isin(test_ids)].copy()
    output = prediction_long_table(
        fitted,
        test_inputs,
        metadata=test_inputs,
        split_name=split_name,
        fold=args.fold,
        targets=test_targets,
    )
    output.to_csv(args.output, index=False)
    if args.model_output:
        with open(args.model_output, "wb") as handle:
            pickle.dump(fitted, handle)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="measurement-sufficiency")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-predictions")
    validate.add_argument("predictions")
    validate.set_defaults(func=_validate)
    validate_dataset = commands.add_parser("validate-dataset", help="validate a dataset-neutral canonical manifest")
    validate_dataset.add_argument("manifest")
    validate_dataset.set_defaults(func=_validate_dataset)
    split = commands.add_parser("split")
    split.add_argument("kind", choices=("field", "gene", "whole-screen", "strict_whole_screen"))
    split.add_argument("observations")
    split.add_argument("output")
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument(
        "--control-column",
        default=None,
        help="gene split only: boolean column marking control observations. Controls are "
        "spread over every fold so each held-out role keeps a same-screen baseline, and are "
        "exempted from the perturbation leakage check. Without it, a dataset whose controls "
        "share one perturbation label fails the leakage check.",
    )
    split.set_defaults(func=_split)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("level", choices=("cell", "ko"))
    metrics.add_argument("predictions")
    metrics.add_argument("output")
    metrics.set_defaults(func=_metrics)
    train = commands.add_parser("train", help="fit a sparse-target built-in model")
    train.add_argument("--inputs", required=True, help="CSV with observation_id and feature columns")
    train.add_argument("--targets", required=True, help="long CSV: observation_id, endpoint_id, y_true[, is_observed]")
    train.add_argument("--observed-column", default="is_observed",
        help="name of the target availability mask column; every value must be an explicit boolean")
    train.add_argument("--model", choices=("ridge", "gbdt", "mlp"), default="ridge")
    train.add_argument("--task-mode", choices=("specialist", "shared_masked"), default="specialist")
    train.add_argument("--features", help="comma-separated feature columns; by default every numeric column except the reserved identifier, partition and label names (observation_id, reporter_id, endpoint_id, screen_id, perturbation_id, guide_id, well_id, field_id, assay_id, is_control, is_observed, scored, fold, role, split_name, model_id, y_true, y_pred)")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--model-output", required=True, help="path for the fitted local artifact")
    train.set_defaults(func=_train)
    predict = commands.add_parser("predict", help="export canonical long predictions from a fitted artifact")
    predict.add_argument("--model-input", required=True)
    predict.add_argument("--inputs", required=True, help="CSV with features plus screen_id, perturbation_id[, reporter_id]")
    predict.add_argument("--targets", help="long evaluation labels; omit only with --include-unobserved")
    predict.add_argument("--observed-column", default="is_observed",
        help="name of the target availability mask column; every value must be an explicit boolean")
    predict.add_argument("--split-name", required=True)
    predict.add_argument("--fold", required=True, type=int)
    predict.add_argument("--output", required=True)
    predict.add_argument("--include-unobserved", action="store_true", help="also export unlabelled endpoint predictions")
    predict.set_defaults(func=_predict)
    run_fold = commands.add_parser("run-fold", help="fit and evaluate one frozen split fold")
    run_fold.add_argument("--inputs", required=True, help="full CSV with features and split metadata; metadata columns are excluded from the feature set by name, so a numeric screen_id or an integer-coded is_control is never trained on")
    run_fold.add_argument("--targets", required=True, help="full long target CSV")
    run_fold.add_argument("--observed-column", default="is_observed",
        help="name of the target availability mask column; every value must be an explicit boolean")
    run_fold.add_argument("--assignments", required=True, help="frozen observation/fold/role CSV")
    run_fold.add_argument("--split-name", choices=("field", "gene", "whole-screen", "strict_whole_screen"), required=True)
    run_fold.add_argument("--heldout-column", help="override the grouping column used for leakage validation")
    run_fold.add_argument("--control-column", default="is_control",
        help="boolean column marking baseline rows; on a gene split these are exempt from the held-out-group check and excluded from accuracy metrics")
    run_fold.add_argument("--fold", required=True, type=int)
    run_fold.add_argument("--model", choices=("ridge", "gbdt", "mlp"), default="ridge")
    run_fold.add_argument("--task-mode", choices=("specialist", "shared_masked"), default="specialist")
    run_fold.add_argument("--features", help="comma-separated feature columns; by default every numeric column except the reserved identifier, partition and label names (observation_id, reporter_id, endpoint_id, screen_id, perturbation_id, guide_id, well_id, field_id, assay_id, is_control, is_observed, scored, fold, role, split_name, model_id, y_true, y_pred)")
    run_fold.add_argument("--seed", type=int, default=0)
    run_fold.add_argument("--output", required=True, help="canonical held-out prediction CSV")
    run_fold.add_argument("--model-output", help="optional local fitted artifact")
    run_fold.set_defaults(func=_run_fold)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
