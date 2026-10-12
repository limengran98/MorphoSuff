#!/usr/bin/env python3
"""Build deterministic global strict whole-screen folds from canonical OPS bundles.

Screen assignment uses only assay topology and cell counts.  Target values are
never opened.  Every held physical screen is removed from all reporters before
source-screen train/validation roles are formed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def table(bundle: Path, manifest: dict[str, Any], key: str) -> Path:
    return bundle / str(manifest["tables"][key])


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release(root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], pd.DataFrame]:
    release = json.loads((root / "canonical_release.json").read_text(encoding="utf-8"))
    bundles = []
    for relative in release["reporter_manifests"]:
        path = root / relative
        bundles.append((path.parent, json.loads(path.read_text(encoding="utf-8"))))
    tasks = pd.read_csv(root / "atlas" / "strict_screen_tasks.csv")
    required = {"reporter_id", "destination_screen"}
    if not required.issubset(tasks):
        raise ValueError(f"strict_screen_tasks lacks {sorted(required - set(tasks))}")
    if tasks.duplicated(["reporter_id", "destination_screen"]).any():
        raise ValueError("strict-screen directions are duplicated")
    return bundles, tasks


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.canonical_root.resolve()
    bundles, tasks = load_release(root)
    counts: dict[str, int] = {}
    reporter_screens: dict[str, set[str]] = {}
    for bundle, manifest in bundles:
        reporter = str(manifest["reporter_id"])
        frame = read(table(bundle, manifest, "inputs"), ["screen_id"])
        local = frame.screen_id.astype(str).value_counts()
        reporter_screens[reporter] = set(local.index)
        for screen, value in local.items():
            counts[str(screen)] = counts.get(str(screen), 0) + int(value)
    directions = tasks.destination_screen.astype(str).value_counts().to_dict()
    screens = sorted(counts, key=lambda value: (-int(directions.get(value, 0)), -counts[value], value))
    folds = [{"screens": [], "directions": 0, "cells": 0} for _ in range(args.n_folds)]
    for screen in screens:
        selected = min(
            range(args.n_folds),
            key=lambda index: (folds[index]["directions"], folds[index]["cells"], index),
        )
        folds[selected]["screens"].append(screen)
        folds[selected]["directions"] += int(directions.get(screen, 0))
        folds[selected]["cells"] += int(counts[screen])
    screen_to_fold = {
        screen: index for index, fold in enumerate(folds) for screen in fold["screens"]
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignment_paths = []
    audits = []
    for fold, record in enumerate(folds):
        held = set(map(str, record["screens"]))
        pieces = []
        eligible = []
        for bundle, manifest in bundles:
            reporter = str(manifest["reporter_id"])
            inputs = read(table(bundle, manifest, "inputs"), ["observation_id", "screen_id"])
            inputs["observation_id"] = inputs.observation_id.astype(str)
            inputs["screen_id"] = inputs.screen_id.astype(str)
            source_screens = reporter_screens[reporter] - held
            destination_screens = reporter_screens[reporter] & held
            field_path = bundle / str(manifest["splits"]["field"])
            field_all = read(field_path)
            field_all["observation_id"] = field_all.observation_id.astype(str)
            # Select the first frozen field fold whose test group survives in
            # the source screens.  This preserves group-held validation even
            # when field fold 0 happens to lie entirely in the destination
            # screen removed by the outer split.
            source_ids = set(inputs.loc[~inputs.screen_id.isin(held), "observation_id"])
            validation_fold = None
            for candidate in sorted(field_all.fold.astype(int).unique()):
                candidate_rows = field_all.loc[field_all.fold.astype(int).eq(candidate)]
                candidate_test = set(
                    candidate_rows.loc[candidate_rows.role.astype(str).eq("test"), "observation_id"]
                )
                if source_ids.intersection(candidate_test):
                    validation_fold = int(candidate)
                    break
            if source_screens and validation_fold is None:
                raise RuntimeError(
                    f"no source-screen field validation group for reporter={reporter}, fold={fold}"
                )
            field = field_all.loc[
                field_all.fold.astype(int).eq(validation_fold if validation_fold is not None else 0),
                ["observation_id", "role"],
            ].copy()
            rows = inputs.merge(field, on="observation_id", how="left", validate="one_to_one")
            rows = rows.loc[~rows.screen_id.isin(held) | bool(source_screens)].copy()
            rows["role"] = np.where(
                rows.screen_id.isin(held),
                "test",
                np.where(rows.role.astype(str).eq("test"), "validation", "train"),
            )
            # A reporter with no remaining source cannot be evaluated and its
            # held rows are omitted rather than leaking into training.
            if not source_screens:
                rows = rows.loc[~rows.screen_id.isin(held)].copy()
            if source_screens and destination_screens:
                eligible.extend(
                    {"reporter_id": reporter, "destination_screen": screen}
                    for screen in sorted(destination_screens)
                )
            rows["fold"] = fold
            rows["split_name"] = "strict_whole_screen"
            rows["reporter_id"] = reporter
            if source_screens and destination_screens:
                if not rows.role.eq("train").any() or not rows.role.eq("validation").any() or not rows.role.eq("test").any():
                    raise RuntimeError(
                        f"incomplete train/validation/test roles for reporter={reporter}, fold={fold}"
                    )
            pieces.append(rows[["observation_id", "fold", "role", "split_name", "reporter_id", "screen_id"]])
        assignments = pd.concat(pieces, ignore_index=True)
        if assignments.duplicated("observation_id").any():
            raise RuntimeError(f"fold {fold} duplicates observation IDs")
        if not set(assignments.role).issubset({"train", "validation", "test"}):
            raise RuntimeError("invalid role")
        path = output / f"assignments_fold_{fold}.csv"
        assignments.to_csv(path, index=False)
        assignment_paths.append({"fold": fold, "path": path.name, "sha256": sha(path)})
        audits.append(
            {
                "fold": fold,
                "held_screens": sorted(held),
                "n_directions": len(eligible),
                "directions": eligible,
                "n_train": int(assignments.role.eq("train").sum()),
                "n_validation": int(assignments.role.eq("validation").sum()),
                "n_test": int(assignments.role.eq("test").sum()),
            }
        )
    observed = {(row["reporter_id"], row["destination_screen"]) for fold in audits for row in fold["directions"]}
    expected = set(map(tuple, tasks[["reporter_id", "destination_screen"]].astype(str).itertuples(index=False, name=None)))
    if observed != expected:
        raise RuntimeError(f"direction coverage differs: missing={len(expected-observed)}, extra={len(observed-expected)}")
    payload = {
        "schema_version": "measurement-sufficiency-strict-screen-plan-v2",
        "randomness": "none",
        "determinism": "greedy screen balance over direction count, then cell count, then screen name",
        "n_folds": int(args.n_folds),
        "n_screens": len(screens),
        "n_directions": len(expected),
        "whole_screen_removed_from_every_reporter": True,
        "target_values_accessed": False,
        "validation_policy": "first frozen field fold with a held group remaining in source screens",
        "screen_to_fold": screen_to_fold,
        "folds": audits,
        "assignments": assignment_paths,
    }
    (output / "strict_screen_plan.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--canonical-root", required=True, type=Path)
    output.add_argument("--output-dir", required=True, type=Path)
    output.add_argument("--n-folds", type=int, default=5)
    output.add_argument(
        "--seed",
        type=int,
        default=None,
        help="refused: the plan is deterministic (greedy screen balance with a stable "
             "tiebreak), so recording a seed would claim a randomness it never used",
    )
    return output


if __name__ == "__main__":
    _args = parser().parse_args()
    if _args.seed is not None:
        raise SystemExit(
            "build_strict_screen is deterministic; --seed has no effect and is refused so "
            "a frozen plan cannot record a seed it never used"
        )
    value = build(_args)
    print(json.dumps({key: value[key] for key in ("schema_version", "n_folds", "n_screens", "n_directions")}, indent=2))
