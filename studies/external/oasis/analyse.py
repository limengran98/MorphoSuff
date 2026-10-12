#!/usr/bin/env python3
"""Run the full measurement-sufficiency chain on prepared cpg0037-oasis tables.

This exists to answer one question: does the toolkit that produced the OPS study
run, unchanged, on a dataset it was not built for and produce the same kind of
evidence?  It therefore calls the shared primitives rather than reimplementing
them.  The only work done here is the glue a new dataset needs:

* fit every fold of the frozen split with the built-in Ridge adapter;
* build replicate pairs so ``stability.within_screen_reliability`` can estimate how
  reproducible the *targeted measurement itself* is;
* assemble the five-column evidence table the tier rule consumes.

That third step is the gap this dataset exposed.  The repository has a producer for
every analysis except the evidence table: ``recoverability_r`` and ``reliability``
appear only in ``analysis/tiers.py`` and in test and config literals, and
``reproducibility/ops/source_map.tsv`` describes the input vaguely as a "reporter
evidence table" with no upstream stage declaring it.  Nothing forces the issue while
the evidence table arrives from a frozen figure package; preparing an external
dataset does.

Reliability is the piece that decides whether a tier can be assigned at all.  A low
prediction score has two incompatible explanations: the cheap measurement does not
carry the information, or the expensive measurement is not reproducible and nothing
could predict it.  The frozen rule refuses to call the first case without ruling out
the second, which is why a missing reliability yields ``not_identifiable`` rather
than ``measurement_required``.  Reporting a Pearson correlation alone would be
exactly the shortcut this study argues against.

Example:

    python studies/external/oasis/analyse.py \\
      --prepared ../external/oasis_canonical \\
      --output-dir ../external/oasis_analysis --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import reject_repo_paths  # noqa: E402

from measurement_sufficiency.adapters import LocalManifestAdapter  # noqa: E402
from measurement_sufficiency.analysis.response_fidelity import response_fidelity  # noqa: E402
from measurement_sufficiency.analysis.stability import within_screen_reliability  # noqa: E402
from measurement_sufficiency.analysis.tiers import tier_table  # noqa: E402
from measurement_sufficiency.metrics import ko_metrics  # noqa: E402
from measurement_sufficiency.splits import make_gene_splits  # noqa: E402
from measurement_sufficiency.training import fit_sparse_model, prediction_long_table  # noqa: E402

#: Endpoints every fidelity quantity is computed on. The raw readouts stay in the
#: canonical tables, but they must not enter the response magnitude, for a reason that
#: is easy to miss and changes the result.
#:
#: ``response_magnitudes`` takes the Euclidean norm of a perturbation's endpoint
#: response vector, which presumes the endpoints share a scale. In OPS they do: a
#: reporter block is standardised. Here they do not. Measured on the prepared 66-plate
#: tables, the median absolute response is 4,575 for ``mtt_lumi`` against 0.071 for
#: ``mtt_normalized``, so ``mtt_lumi`` accounts for **100.0%** of the squared norm and
#: the normalised endpoint contributes nothing; for LDH the split is 13.3% / 86.7%.
#: Two consequences, both fatal to a comparison: MTT's magnitude would be computed on
#: raw luminescence, which carries plate gain drift rather than biology, and the two
#: reporters' magnitudes would not be on comparable footings.
#:
#: Restricting to the plate-normalised readout fixes both. It also makes each reporter
#: single-endpoint, so the magnitude is an absolute value rather than a norm and
#: ``profile_fidelity`` is undefined. That is a real difference from OPS, whose
#: reporters carry 20 to 72 endpoints, and it is reported rather than papered over.
PRIMARY_ENDPOINTS = ("mtt_normalized", "ldh_normalized")


def load(prepared: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Read the prepared canonical tables and the declared input feature names."""
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["input_schema"]["feature_names"])

    def read(name: str) -> pd.DataFrame:
        path = prepared / manifest["tables"][name]
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    observations = read("observations")
    inputs = observations.merge(read("inputs"), on="input_cell_id", how="left")
    targets = read("targets")
    return observations, inputs, targets, features


def replicate_pairs(observations: pd.DataFrame, targets: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Pair independent measurements of the same compound for a reliability estimate.

    A compound is measured in several wells, within a batch and across the four
    production batches. Splitting those wells into two disjoint halves and correlating
    the half means over compounds estimates how reproducible the assay is, which is the
    quantity the tier rule gates on.

    Wells are assigned to halves by a seeded permutation within each compound, so the
    two halves are exchangeable and no batch is systematically on one side. Compounds
    measured only once contribute nothing and are excluded, with the count reported.

    **The dose axis is collapsed, and that is a limitation rather than a design.** The
    study images 1,085 compounds at eight concentrations, but ``prepare.py`` carries
    ``Metadata_compound_name`` into ``perturbation_id`` and does not carry
    ``compound_concentration_um`` into the canonical tables at all. So the grouping key
    here is the compound, a half is a random mix of concentrations, and both halves
    estimate the same dose-averaged quantity rather than a dose-matched one. An earlier
    version of this docstring said "a compound at one concentration", which the code
    has never done.

    This response-fidelity workflow retains compound-level grouping. The separate
    metabolic-loss prioritization workflow uses dose-matched aggregation; its
    relationship to this estimator is described in
    ``studies/external/README.md#interpreting-and-reusing-the-assessment``.
    """
    frame = targets.merge(
        observations[["observation_id", "reporter_id", "perturbation_id", "is_control"]],
        on="observation_id",
        how="inner",
    )
    frame = frame.loc[
        frame["endpoint_id"].isin(PRIMARY_ENDPOINTS) & ~frame["is_control"] & frame["y_true"].notna()
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for (reporter, endpoint, compound), group in frame.groupby(
        ["reporter_id", "endpoint_id", "perturbation_id"], sort=True
    ):
        if len(group) < 2:
            continue
        order = rng.permutation(len(group))
        half = len(group) // 2
        values = group["y_true"].to_numpy()
        rows.append(
            {
                "reporter_id": reporter,
                "endpoint_id": endpoint,
                "perturbation_id": compound,
                "value_a": float(values[order[:half]].mean()),
                "value_b": float(values[order[half:]].mean()),
                "n_measurements": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def fit_all_folds(
    inputs: pd.DataFrame, targets: pd.DataFrame, assignments: pd.DataFrame, features: list[str], model: str
) -> pd.DataFrame:
    """Fit every fold with the built-in adapter and return one long prediction table."""
    frames = []
    for fold in sorted(assignments["fold"].unique()):
        roles = assignments.loc[assignments["fold"].eq(fold)]
        train_ids = set(roles.loc[roles["role"].eq("train"), "observation_id"])
        test_ids = set(roles.loc[roles["role"].eq("test"), "observation_id"])
        train_inputs = inputs.loc[inputs["observation_id"].isin(train_ids)]
        test_inputs = inputs.loc[inputs["observation_id"].isin(test_ids)]
        train_targets = targets.loc[targets["observation_id"].isin(train_ids)]
        fitted = fit_sparse_model(
            train_inputs, train_targets, feature_columns=features, model_id=model
        )
        frames.append(
            prediction_long_table(
                fitted,
                test_inputs,
                metadata=test_inputs,
                split_name="gene",
                fold=int(fold),
                targets=targets.loc[targets["observation_id"].isin(test_ids)],
                # A reporter's endpoints are absent from a fold only if no training
                # well carried them; that is a data property, not a caller error.
                allow_unmodelled_endpoints=True,
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_evidence(
    predictions: pd.DataFrame, fidelity: dict[str, pd.DataFrame], reliability: pd.DataFrame
) -> pd.DataFrame:
    """Assemble the five columns the frozen tier rule consumes, one row per reporter.

    Every column comes from a shared primitive: ``recoverability_r`` from
    ``metrics.ko_metrics``, the three fidelity columns from
    ``analysis.response_fidelity``, and ``reliability`` from
    ``analysis.stability.within_screen_reliability``. A reporter with no reliability
    estimate keeps a null rather than a zero, because the rule distinguishes
    "unreliable" from "not established" and a zero would silently collapse them.
    """
    recoverability = (
        ko_metrics(predictions.loc[predictions["endpoint_id"].isin(PRIMARY_ENDPOINTS)])
        .groupby("reporter_id", as_index=False)["pearson_r"]
        .median()
        .rename(columns={"pearson_r": "recoverability_r"})
    )
    evidence = recoverability
    for key, column in (
        ("magnitude_spearman", "magnitude_spearman"),
        ("variance_ratio", "variance_ratio"),
        ("strong_hit_recall", "top5pct_recall"),
    ):
        table = fidelity[key]
        if column not in table.columns:
            raise KeyError(f"response_fidelity['{key}'] lacks {column}; got {list(table.columns)}")
        evidence = evidence.merge(
            table.groupby("reporter_id", as_index=False)[column].median(), on="reporter_id", how="left"
        )
    if not reliability.empty:
        evidence = evidence.merge(
            reliability.groupby("reporter_id", as_index=False)["within_screen_reliability"]
            .median()
            .rename(columns={"within_screen_reliability": "reliability"}),
            on="reporter_id",
            how="left",
        )
    else:
        evidence["reliability"] = np.nan
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True, help="directory written by prepare.py")
    parser.add_argument("--output-dir", required=True, help="destination outside the repository")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="ridge")
    parser.add_argument(
        "--batch", action="append", default=None, metavar="BATCH_ID",
        help="restrict to one production batch; repeatable. The four axiom batches are "
             "independent runs of the same design, so running each on its own asks "
             "whether the verdict is a property of the assay or of one production run. "
             "Folds, replicate pairs and reliability are then all inside that batch",
    )
    parser.add_argument(
        "--permute-inputs", choices=("none", "within_screen"), default="none",
        help="negative control. 'within_screen' gives every well another well's "
             "brightfield features inside the same plate, leaving the assay readings, "
             "the compound labels and the control flags where they are. Reliability "
             "reads only the assay side and must come back unchanged; recoverability "
             "must collapse. Output written under this flag is not a result about this "
             "dataset and provenance.json records it as a control",
    )
    parser.add_argument("--permutation-seed", type=int, default=None,
                        help="seed for --permute-inputs; defaults to --seed")
    return parser


def permute_inputs_within_screen(
    inputs: pd.DataFrame, features: list[str], *, seed: int
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Give every well another well's brightfield features, inside the same plate.

    This is the negative control the panel needs, and this dataset is where it can be
    read. Every evidence column is computed from a fitted model, so a panel of rows
    that all fail says nothing about whether the criteria can distinguish an
    informative input from an uninformative one. On cpg0021 the question cannot be put:
    the reliability gate fires first there and the verdict never consults the input at
    all. Here reliability is 0.72 and 0.64, comfortably above the floor, so the rule
    does reach the input and a destroyed input has somewhere to show.

    The shuffle is at the **well** and inside the plate. At the well, because the well
    is what carries one profile and one pair of assay readings, so shuffling per
    observation row would give a well two different sets of features for its two
    reporters. Inside the plate, because response is defined against that plate's DMSO
    controls, and a shuffle across plates would additionally destroy the baseline, after
    which a collapse could be blamed on the baseline rather than on the permutation.
    """
    rng = np.random.default_rng(seed)
    wells = inputs.drop_duplicates("input_cell_id")[["input_cell_id", "screen_id"]]
    mapping: dict[str, str] = {}
    moved = 0
    for _, block in wells.groupby("screen_id", sort=True):
        identifiers = block["input_cell_id"].to_numpy()
        shuffled = identifiers[rng.permutation(len(identifiers))]
        mapping.update(dict(zip(identifiers, shuffled)))
        moved += int((identifiers != shuffled).sum())
    donor = (
        inputs.drop_duplicates("input_cell_id")
        .set_index("input_cell_id")[features]
    )
    replacement = donor.reindex(inputs["input_cell_id"].map(mapping))
    if replacement[features[0]].isna().any():
        raise SystemExit("the permutation produced a well with no donor features")
    permuted = inputs.copy()
    permuted[features] = replacement.to_numpy()
    return permuted, {
        "n_screens": int(wells["screen_id"].nunique()),
        "n_wells": int(len(wells)),
        "n_wells_moved": moved,
        "fraction_moved": float(moved / len(wells)) if len(wells) else 0.0,
    }


def restrict_to_batches(
    observations: pd.DataFrame, inputs: pd.DataFrame, targets: pd.DataFrame, batches: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep only the named batches, in all three tables at once.

    Filtering the observations alone would leave the target rows of dropped wells in
    place, and ``build_evidence`` would then compute reliability over compounds the
    model never saw. The three tables are filtered from one identifier set.
    """
    available = sorted(observations["batch_id"].unique())
    unknown = sorted(set(batches).difference(available))
    if unknown:
        raise SystemExit(f"unknown batch(es) {unknown}; the prepared tables carry {available}")
    keep = observations["batch_id"].isin(batches)
    observations = observations.loc[keep].reset_index(drop=True)
    identifiers = set(observations["observation_id"])
    inputs = inputs.loc[inputs["observation_id"].isin(identifiers)].reset_index(drop=True)
    targets = targets.loc[targets["observation_id"].isin(identifiers)].reset_index(drop=True)
    return observations, inputs, targets


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = LocalManifestAdapter(args.prepared / "canonical_manifest.json")
    observations, inputs, targets, features = load(args.prepared)
    if args.batch:
        observations, inputs, targets = restrict_to_batches(
            observations, inputs, targets, args.batch
        )
    # Counted over wells, not observation rows: an observation is a (well, reporter)
    # pair, so every well appears twice and a compound in a single well would look
    # like a compound measured twice. Reliability pairs wells, so wells are the unit
    # that decides whether it is estimable at all.
    wells = observations.loc[~observations["is_control"]].drop_duplicates(
        ["perturbation_id", "input_cell_id"]
    )
    per_compound = wells["perturbation_id"].value_counts()
    summary = {
        "observations": len(observations),
        "features": len(features),
        "batches": sorted(observations["batch_id"].unique()),
        "screens": int(observations["screen_id"].nunique()),
        "compound_wells": int(len(wells)),
        "compounds": int(per_compound.size),
        "compounds_in_two_or_more_wells": int((per_compound >= 2).sum()),
        "control_observations": int(observations["is_control"].sum()),
        "capabilities": dataset.capabilities.__dict__,
    }
    print(json.dumps(summary, indent=2))

    permutation = None
    if args.permute_inputs == "within_screen":
        seed = args.seed if args.permutation_seed is None else args.permutation_seed
        inputs, permutation = permute_inputs_within_screen(inputs, features, seed=seed)
        permutation |= {"mode": args.permute_inputs, "seed": seed}
        print(
            "\n  NEGATIVE CONTROL: brightfield features were shuffled between wells "
            f"inside each plate.\n  {permutation['n_wells_moved']:,} of "
            f"{permutation['n_wells']:,} wells moved across {permutation['n_screens']} "
            "plates. Nothing written to this\n  directory describes cpg0037-oasis. "
            "Reliability reads only the assay side and must be\n  unchanged; "
            "recoverability should collapse.\n",
            flush=True,
        )

    run_kind = "all_batches" if not args.batch else "single_batch"
    if permutation:
        run_kind = "inputs_permuted_within_screen"
    (output / "provenance.json").write_text(json.dumps({
        "prepared": str(args.prepared), "seed": args.seed, "folds": args.folds,
        "model": args.model,
        "batch_restriction": sorted(args.batch) if args.batch else None,
        "run_kind": run_kind,
        "permutation": permutation,
        **{key: value for key, value in summary.items() if key != "capabilities"},
    }, indent=2) + "\n", encoding="utf-8")

    manifest = make_gene_splits(
        observations, control_column="is_control", n_folds=args.folds, seed=args.seed
    )
    manifest.assignments.to_csv(output / "split.csv", index=False)

    predictions = fit_all_folds(inputs, targets, manifest.assignments, features, args.model)
    predictions.to_csv(output / "predictions.csv", index=False)
    print(f"predictions: {len(predictions)} rows over {predictions.fold.nunique()} folds")

    # The capability declared by the manifest is enforced here, not assumed. Fidelity
    # is computed on the plate-normalised endpoints only; see PRIMARY_ENDPOINTS for
    # why mixing scales would silently decide the result.
    scored = predictions.loc[predictions["endpoint_id"].isin(PRIMARY_ENDPOINTS)]
    fidelity = response_fidelity(scored, capabilities=dataset.capabilities)
    for name, table in fidelity.items():
        table.to_csv(output / f"fidelity_{name}.csv", index=False)

    pairs = replicate_pairs(observations, targets, seed=args.seed)
    pairs.to_csv(output / "replicate_pairs.csv", index=False)
    reliability = (
        within_screen_reliability(pairs, group_columns=["reporter_id", "endpoint_id"])
        if not pairs.empty
        else pd.DataFrame()
    )
    reliability.to_csv(output / "reliability.csv", index=False)
    print(f"replicate pairs: {len(pairs)} compounds with at least two measurements")

    evidence = build_evidence(predictions, fidelity, reliability)
    evidence.to_csv(output / "evidence.csv", index=False)
    print("\nevidence table the frozen tier rule consumes:")
    print(evidence.to_string(index=False))

    tiers = tier_table(evidence, group_columns=["reporter_id"])
    tiers.to_csv(output / "tiers.csv", index=False)
    print("\ntier assignment under the frozen OPS rule, applied without tuning:")
    print(tiers.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
