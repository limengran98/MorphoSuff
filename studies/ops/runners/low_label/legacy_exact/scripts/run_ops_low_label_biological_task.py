#!/usr/bin/env python3
"""Run one biological OPS low-label task for MIDAS or scButterfly.

MIDAS trains on Donor40 full labels plus Target12 low labels.  scButterfly is
kept faithful to its pairwise/specialist role and trains only twelve independent
Target12 translators.  Both use a fresh post-selection process to load the
outer-test labels and emit the common low-label result marker.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

import ops_low_label_sampling_lib as sampling
import run_ops_biological_baseline_full52_training as bio


RESULT_SCHEMA = "ops-low-label-result-v1"
METHODS = ("scbutterfly_ops_b", "midas_ops")
OUTPUT_LOCK_TOKEN_ENV = "OPS_LOW_LABEL_OUTPUT_LOCK_TOKEN"
OUTPUT_LOCK_SCHEMA = "ops-low-label-output-root-lock-v1"


class OutputRootLockError(RuntimeError):
    """Raised when two independent jobs try to own the same output root."""


class OutputRootWriterLock:
    """Cross-node full-run writer exclusion for one low-label output root.

    Distributed filesystems do not always provide dependable shared advisory
    ``flock`` semantics. An atomic sibling ``.active.lockdir`` is therefore
    the primary guard. The
    training parent owns it throughout training and its evaluation child; the
    latter may write only through the parent-inherited opaque token.
    """

    def __init__(
        self,
        *,
        output_root: Path,
        lock_dir: Path,
        token: str,
        inherited: bool,
        previous_token: str | None,
    ) -> None:
        self.output_root = output_root
        self.lock_dir = lock_dir
        self.token = token
        self.inherited = inherited
        self.previous_token = previous_token
        self.released = False

    @classmethod
    def acquire_from_argv(cls, argv: list[str]) -> "OutputRootWriterLock":
        try:
            root_index = argv.index("--output-root")
            raw_root = argv[root_index + 1]
        except (ValueError, IndexError) as error:
            raise OutputRootLockError(
                "Biological low-label task lacks a valid --output-root"
            ) from error
        output_root = Path(raw_root).expanduser().resolve()
        lock_dir = output_root.with_name(f"{output_root.name}.active.lockdir")
        evaluation_only = "--evaluation-only" in argv
        inherited_token = os.environ.get(OUTPUT_LOCK_TOKEN_ENV)

        # The post-selection evaluation subprocess is intentionally a fresh
        # process. It may write the same root only while its training parent is
        # still alive and only with that parent's matching token.
        if evaluation_only and inherited_token and lock_dir.is_dir():
            owner_path = lock_dir / "owner.json"
            try:
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
            except Exception as error:
                raise OutputRootLockError(
                    f"Cannot verify inherited output-root ownership: {owner_path}"
                ) from error
            if owner.get("token") != inherited_token:
                raise OutputRootLockError(
                    "Evaluation child output-root token does not match its training parent"
                )
            return cls(
                output_root=output_root,
                lock_dir=lock_dir,
                token=inherited_token,
                inherited=True,
                previous_token=inherited_token,
            )

        lock_dir.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(24)
        try:
            lock_dir.mkdir()
        except FileExistsError as error:
            owner_path = lock_dir / "owner.json"
            try:
                owner_hint = owner_path.read_text(encoding="utf-8").strip()
            except OSError:
                owner_hint = "owner metadata unreadable"
            raise OutputRootLockError(
                "Refusing concurrent writes to one low-label output root: "
                f"{output_root}; active lock={lock_dir}; owner={owner_hint}"
            ) from error
        previous_token = os.environ.get(OUTPUT_LOCK_TOKEN_ENV)
        owner = {
            "schema_version": OUTPUT_LOCK_SCHEMA,
            "token": token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "started_unix": time.time(),
            "output_root": str(output_root),
        }
        try:
            (lock_dir / "owner.json").write_text(
                json.dumps(owner, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception:
            lock_dir.rmdir()
            raise
        os.environ[OUTPUT_LOCK_TOKEN_ENV] = token
        return cls(
            output_root=output_root,
            lock_dir=lock_dir,
            token=token,
            inherited=False,
            previous_token=previous_token,
        )

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        if self.inherited:
            return
        try:
            owner_path = self.lock_dir / "owner.json"
            if owner_path.is_file():
                owner = json.loads(owner_path.read_text(encoding="utf-8"))
                if owner.get("token") != self.token:
                    raise OutputRootLockError(
                        "Output-root lock ownership changed unexpectedly; refusing removal"
                    )
                owner_path.unlink()
            self.lock_dir.rmdir()
        finally:
            if os.environ.get(OUTPUT_LOCK_TOKEN_ENV) == self.token:
                if self.previous_token is None:
                    os.environ.pop(OUTPUT_LOCK_TOKEN_ENV, None)
                else:
                    os.environ[OUTPUT_LOCK_TOKEN_ENV] = self.previous_token


def _pop(argv: list[str], option: str, default: str | None = None) -> str | None:
    if option not in argv:
        return default
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {option}")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def _peek(argv: list[str], option: str) -> str | None:
    if option not in argv:
        return None
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {option}")
    return argv[index + 1]


def _configure_argv() -> tuple[str, float, Path, bool]:
    argv = list(sys.argv)
    method = _pop(argv, "--method") or os.environ.get("OPS_LOW_LABEL_BIO_METHOD")
    if method not in METHODS:
        raise SystemExit(f"--method must be one of {METHODS}")
    raw_fraction = _pop(argv, "--label-fraction") or os.environ.get(
        "OPS_LOW_LABEL_FRACTION"
    )
    if raw_fraction is None:
        raise SystemExit("Missing --label-fraction")
    fraction = float(raw_fraction)
    sampling_root = Path(
        _pop(argv, "--sampling-manifest-root")
        or os.environ.get("OPS_LOW_LABEL_SAMPLING_ROOT", str(sampling.DEFAULT_ROOT))
    ).expanduser().resolve()
    end_to_end_smoke = "--end-to-end-smoke" in argv
    if end_to_end_smoke:
        argv.remove("--end-to-end-smoke")
    # The fixed scButterfly source anchor is FP32/no-TF32.
    if method == "scbutterfly_ops_b":
        if "--amp" in argv or "--tf32" in argv:
            raise SystemExit("scButterfly low-label requires --no-amp --no-tf32")
        if "--no-amp" not in argv:
            argv.append("--no-amp")
        if "--no-tf32" not in argv:
            argv.append("--no-tf32")
    # ``_pop`` removes the wrapper option, but the reused formal parser also
    # requires ``--method``.  Put the validated value back before delegating.
    argv.extend(["--method", method])
    sys.argv = argv
    os.environ["OPS_LOW_LABEL_BIO_METHOD"] = method
    os.environ["OPS_LOW_LABEL_FRACTION"] = sampling.fraction_token(fraction)
    os.environ["OPS_LOW_LABEL_SAMPLING_ROOT"] = str(sampling_root)
    return method, fraction, sampling_root, end_to_end_smoke


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "usage: run_ops_low_label_biological_task.py --method "
            "{scbutterfly_ops_b,midas_ops} --label-fraction FRACTION "
            "--split gene_holdout_main --fold {0,1,2,3,4} "
            "--campaign-config JSON --output-root PATH "
            "--sampling-manifest-root PATH [delegated OPS asset options]\n\n"
            + (__doc__ or "")
        )
        return
    method, fraction, sampling_root, end_to_end_smoke = _configure_argv()
    target_table = Path(
        _peek(sys.argv, "--target-table")
        or os.environ.get(
            "OPS_LOW_LABEL_TARGET_TABLE", bio.common_runner.DEFAULT_TARGET_TABLE
        )
    ).expanduser().resolve()
    frozen_table = pd.read_csv(target_table)
    target12 = sampling.canonical_target12(frozen_table)
    full52 = tuple(frozen_table["reporter_slug"].astype(str))
    training_registry = target12 if method == "scbutterfly_ops_b" else full52
    bio.FULL52 = training_registry

    original_bind = bio.bind_fixed_trial
    original_prepare = bio.prepare_training_data
    original_prepare_eval = bio.prepare_evaluation_data
    original_write = bio.write_formal_result

    def bind(args: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        campaign, base, base_path = bio.load_campaign(args.campaign_config)
        trial_index = 0
        trial = json.loads(json.dumps(base["methods"][method]["trial_configs"][0]))
        binding = {
            "schema_version": RESULT_SCHEMA,
            "campaign_path": str(args.campaign_config.resolve()),
            "campaign_sha256": bio.common_runner.file_sha256(args.campaign_config),
            "base_method_config": str(base_path),
            "base_method_config_sha256": bio.common_runner.file_sha256(base_path),
            "method": method,
            "fixed_trial_index": trial_index,
            "fixed_trial_sha256": bio.common_runner.json_sha256(trial),
            "reporters": list(training_registry),
            "split": args.split,
            "fold": args.fold,
            "seed": args.seed,
            "mode": "formal",
            "label_fraction": fraction,
            "target_reporters": list(target12),
            "target_reporter_policy": "panel12_low_label",
            "donor_reporter_policy": (
                "absent"
                if method == "scbutterfly_ops_b"
                else "donor40_full"
            ),
            "experiment_arm": (
                "target_only"
                if method == "scbutterfly_ops_b"
                else "atlas_transfer"
            ),
            "central_sampling_manifest_required": True,
            "hyperparameter_search": False,
            "outer_test_used_for_training_or_checkpoint_selection": False,
        }
        return base, trial, binding

    def prepare(args: Any) -> Any:
        # The full52 source runner deliberately rejects a target table whose
        # complete order is not identical to ``FULL52``.  For scButterfly the
        # faithful low-label adaptation is Target12 specialist training, while
        # MIDAS remains a full52 mosaic model.  Build the sealed subset here
        # instead of weakening the source runner's full52 assertion.
        reporter_table = pd.read_csv(args.target_table)
        if tuple(reporter_table["reporter_slug"].astype(str)) != full52:
            raise RuntimeError("Frozen target table order changed")
        phase_cache = bio.PhaseCache.open(args.phase_cache)
        bio.common_runner.validate_frozen_assets(
            phase_cache, reporter_table, args.exact_root
        )
        technical = bio.frozen_engine.load_technical_core_features(
            args.target_feature_dictionary, reporter_table
        )
        reporter_data = {
            slug: bio.common_runner.load_sealed_training_data(
                bio.common_runner.exact_cache_path(args.exact_root, slug),
                slug,
                technical[slug],
                phase_cache,
                args.split,
                args.fold,
            )
            for slug in training_registry
        }
        indexed = reporter_table.set_index("reporter_slug", drop=False)
        heads = [
            bio.common_runner.build_sealed_training_partition(
                args, phase_cache, indexed.loc[slug], reporter_data[slug]
            )
            for slug in training_registry
        ]
        reference_path = (
            args.preprocessing_reference_root
            / args.split
            / f"fold_{args.fold}"
            / "train"
            / "preprocessing.json"
        )
        x_state = bio.common_runner.load_reference_preprocessing(
            reference_path, heads
        )
        bio.frozen_engine.global_partition_integrity(heads)
        comparability = {
            head.slug: {
                "strict_specialist_comparable": True,
                "reference": str(
                    bio.common_runner.frozen_test_reference(args, head.slug)[
                        2
                    ].resolve()
                ),
                "outer_test_y_physically_loaded": False,
            }
            for head in heads
        }
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
        )
        sampling.apply_manifest_to_heads(
            heads=heads,
            manifest_file=manifest_file,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
            seed=args.seed,
            target_reporters=target12,
        )
        reference_path = (
            args.preprocessing_reference_root
            / args.split
            / f"fold_{args.fold}"
            / "train"
            / "preprocessing.json"
        )
        preprocessing = bio.common_runner.preprocessing_manifest(
            reference_path, x_state, heads
        )
        preprocessing.update(
            {
                "central_sampling_manifest": str(manifest_file),
                "central_sampling_manifest_sha256": sampling.file_sha256(
                    manifest_file
                ),
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": (
                    "absent"
                    if method == "scbutterfly_ops_b"
                    else "donor40_full"
                ),
            }
        )
        preprocessing_path = args.output_root / "preprocessing.json"
        bio.atomic_json(preprocessing_path, preprocessing)
        split_contract = bio.common_runner.build_split_contract(args, heads)
        return (
            phase_cache,
            reporter_table,
            heads,
            x_state,
            comparability,
            split_contract,
            bio.common_runner.file_sha256(preprocessing_path),
        )

    def prepare_eval(args: Any) -> Any:
        table = pd.read_csv(args.target_table)
        indexed = table.set_index("reporter_slug", drop=False)
        phase_cache = bio.PhaseCache.open(args.phase_cache)
        bio.common_runner.validate_frozen_assets(
            phase_cache, table, args.exact_root
        )
        technical = bio.frozen_engine.load_technical_core_features(
            args.target_feature_dictionary, table
        )
        heads = []
        for slug in target12:
            reference_rows, reference_controls, _ = (
                bio.common_runner.frozen_test_reference(args, slug)
            )
            data = bio.common_runner.load_test_only_data(
                bio.common_runner.exact_cache_path(args.exact_root, slug),
                slug,
                technical[slug],
                reference_rows,
                reference_controls,
            )
            heads.append(
                bio.common_runner.test_only_partition(
                    indexed.loc[slug],
                    data,
                    (args.fold + 1) % int(phase_cache.manifest["n_folds"]),
                )
            )
        preprocessing_path = args.output_root / "preprocessing.json"
        payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        x_state = bio.frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
        for head in heads:
            head.y_preprocessing = bio.frozen_engine.HeadYPreprocessing.from_json(
                payload["heads"][head.slug]
            )
        comparability = {
            head.slug: bio.frozen_engine.assert_specialist_comparability(
                args, head, args.split, args.fold
            )
            for head in heads
        }
        staging = bio.common_runner.build_phase_only_gpu_staging(
            args, phase_cache, x_state
        )
        return (
            phase_cache,
            heads,
            x_state,
            comparability,
            staging,
            bio.common_runner.file_sha256(preprocessing_path),
        )

    def write(args: Any, complete: Mapping[str, Any]) -> None:
        # The training registry is full52 for MIDAS, but every low-label outer
        # evaluation is intentionally Target12.  The reused evaluator records
        # ``len(FULL52)``; correct that run-scope field before the source writer
        # hashes the evaluation marker into ``training_result.json``.
        evaluation_path = args.output_root / "evaluation_complete.json"
        evaluation = bio.common_runner.read_json_without_duplicate_keys(
            evaluation_path
        )
        if (
            not (
                evaluation.get("status") == "complete"
                or evaluation.get("completed") is True
            )
            or evaluation.get("n_evaluation_states") != 3
            or evaluation.get(
                "outer_test_y_loaded_after_checkpoint_selection"
            )
            is not True
            or evaluation.get("test_may_select_state") is not False
        ):
            raise RuntimeError("Biological low-label evaluation is incomplete")
        evaluation["n_reporters"] = len(target12)
        evaluation["evaluation_scope"] = "canonical_target12_outer_test"
        bio.atomic_json(evaluation_path, evaluation)
        original_write(args, complete)
        # The reused source writer reports the immutable 52-reporter registry
        # size. That is correct for MIDAS but not for the faithful Target12
        # scButterfly specialist adaptation. Make the run-scope fields truthful
        # before emitting the strict low-label completion marker.
        training_result_path = args.output_root / "training_result.json"
        training_result = bio.common_runner.read_json_without_duplicate_keys(
            training_result_path
        )
        head_schema = bio.common_runner.read_json_without_duplicate_keys(
            args.output_root / "head_schema.json"
        )
        training_result["n_reporters"] = len(training_registry)
        training_result["n_endpoints"] = sum(
            int(value) for value in head_schema["head_dimensions"].values()
        )
        bio.atomic_json(training_result_path, training_result)
        summary_path = args.output_root / "evaluation_states_summary.csv"
        rows = pd.read_csv(summary_path)
        if (
            len(rows) != 3 * len(target12)
            or set(rows["evaluation_state"])
            != {"single", "ema", "checkpoint_average"}
        ):
            raise RuntimeError(
                "Biological low-label evaluation state matrix is incomplete"
            )
        primary = rows.loc[rows["evaluation_state"] == "single"].copy()
        if len(primary) != 12 or set(primary["reporter_slug"]) != set(target12):
            raise RuntimeError("Biological low-label evaluation is not Target12")
        preferred = (
            "cell_macro_feature_pearson",
            "cell_standardized_gain_vs_train_mean",
            "gene_macro_feature_pearson",
            "gene_gain_vs_train_mean",
            "gene_mean_profile_cosine",
            "gene_response_magnitude_spearman",
        )
        missing_metrics = [name for name in preferred if name not in primary]
        if missing_metrics:
            raise RuntimeError(
                f"Biological low-label summary lacks metrics: {missing_metrics}"
            )
        macro_metrics = {
            name: float(primary[name].mean()) for name in preferred
        }
        if not all(math.isfinite(value) for value in macro_metrics.values()):
            raise RuntimeError("Biological low-label macro metrics are nonfinite")
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
        )
        bio.atomic_json(
            args.output_root / "low_label_result.json",
            {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "completed": True,
                "mode": "formal",
                "method_id": method,
                "experiment_arm": (
                    "target_only"
                    if method == "scbutterfly_ops_b"
                    else "atlas_transfer"
                ),
                "label_fraction": fraction,
                "split": args.split,
                "fold": args.fold,
                "seed": args.seed,
                "n_target_reporters": 12,
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": (
                    "absent"
                    if method == "scbutterfly_ops_b"
                    else "donor40_full"
                ),
                "cohort_matches_frozen_reference": True,
                "sampling_manifest": str(manifest_file),
                "sampling_manifest_sha256": sampling.file_sha256(manifest_file),
                "execution_device": "cuda",
                "cpu_fallback_allowed": False,
                "outer_test_used_for_selection": False,
                "outer_test_y_loaded_after_checkpoint_selection": True,
                "test_evaluated": True,
                "macro_metrics": macro_metrics,
                "reporter_metrics": str(summary_path.resolve()),
                "reporter_metrics_sha256": bio.common_runner.file_sha256(
                    summary_path
                ),
            },
        )

    bio.bind_fixed_trial = bind
    bio.prepare_training_data = prepare
    bio.prepare_evaluation_data = prepare_eval
    bio.write_formal_result = write
    # Preserve the wrapper in the fresh post-selection child.
    bio.__file__ = str(Path(__file__).resolve())
    if end_to_end_smoke:
        # The method-native loop reads the frozen trial mapping.  Use a local
        # bind wrapper with one epoch and one reporter round; production config
        # and source files remain unchanged.
        previous_bind = bio.bind_fixed_trial

        def smoke_bind(args: Any) -> Any:
            base, trial, binding = previous_bind(args)
            # Two trained states are the minimum needed by the immutable
            # checkpoint-averaging contract exercised after training.
            trial["training"]["joint_epochs"] = 2
            for key in (
                "phase_pretrain_epochs",
                "phenotype_pretrain_epochs",
                "adversarial_epochs",
            ):
                if key in trial["training"]:
                    trial["training"][key] = 1
            args.reporter_rounds_per_epoch_override = 1
            binding["end_to_end_smoke"] = True
            return base, trial, binding

        bio.bind_fixed_trial = smoke_bind
    writer_lock = OutputRootWriterLock.acquire_from_argv(sys.argv)
    print(
        json.dumps(
            {
                "status": (
                    "OUTPUT_ROOT_LOCK_INHERITED"
                    if writer_lock.inherited
                    else "OUTPUT_ROOT_LOCK_ACQUIRED"
                ),
                "output_root": str(writer_lock.output_root),
                "lock_dir": str(writer_lock.lock_dir),
                "method": method,
                "label_fraction": fraction,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        bio.main()
    finally:
        writer_lock.release()
        bio.bind_fixed_trial = original_bind
        bio.prepare_training_data = original_prepare
        bio.prepare_evaluation_data = original_prepare_eval
        bio.write_formal_result = original_write


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
