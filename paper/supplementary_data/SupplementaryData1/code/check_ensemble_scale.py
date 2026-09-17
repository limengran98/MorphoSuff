#!/usr/bin/env python3
"""Recompute the response-scale sensitivity of the fixed median ensemble.

Requires the separately supplied frozen prediction archive. No predictor is fitted.
The main common-scale values must reproduce evidence.csv before results are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import replay_frozen_predictions as replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mm-root', type=Path, required=True)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'scale_sensitivity.csv')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    replay.MM = args.mm_root.resolve()
    spec = json.loads((root / 'analysis_spec.json').read_text())
    methods = pd.read_csv(root / 'method_roster.csv').sort_values('method_order').method_id.tolist()
    reference = pd.read_csv(root / 'evidence.csv')
    reference = reference.loc[reference.run_state.eq('checkpoint_average') &
                              reference.method_id.eq('coherent_median_ensemble')].set_index('reporter_slug')
    scales = pd.read_csv(root / 'provenance/endpoint_scales.csv')
    records = []
    for reporter in spec['reporter_order']:
        genes, y_fold, p_fold, y_common, p_common, recoveries = [], [], [], [], [], []
        seen_genes = set()
        for fold in range(5):
            paths = replay.profile_paths('ridge', fold, reporter, 'checkpoint_average')
            gene, screen, _, truth_path, names_path, _ = paths
            gene, screen = replay.load(gene), replay.load(screen)
            truth = replay.load(truth_path).astype(float)
            names = names_path.read_text().splitlines()
            keys = list(zip(gene.tolist(), screen.tolist()))
            assert len(keys) == len(set(keys))
            fold_genes = set(gene.tolist())
            assert not seen_genes.intersection(fold_genes)
            seen_genes.update(fold_genes)
            factors = scales.loc[scales.reporter_slug.eq(reporter) & scales.fold.eq(fold)].sort_values('endpoint_index')
            assert factors.endpoint_name.tolist() == names
            factor = factors.fold_to_common_factor.to_numpy(float)
            assert np.isfinite(factor).all() and (factor > 0).all()
            predictions = []
            for method in methods:
                gp, sp, pp, yp, npth, _ = replay.profile_paths(method, fold, reporter, 'checkpoint_average')
                mg, ms = replay.load(gp), replay.load(sp)
                mkeys = list(zip(mg.tolist(), ms.tolist()))
                assert len(mkeys) == len(set(mkeys)) and set(mkeys) == set(keys)
                lookup = {key: i for i, key in enumerate(mkeys)}
                order = [lookup[key] for key in keys]
                pred, target = replay.load(pp)[order].astype(float), replay.load(yp)[order].astype(float)
                assert npth.read_text().splitlines() == names
                assert np.array_equal(target, truth) and pred.shape == truth.shape
                predictions.append(pred)
            median = np.median(np.stack(predictions), axis=0)
            recoveries.append(replay.macro_r(truth, median))
            genes.append(gene)
            y_fold.append(truth)
            p_fold.append(median)
            y_common.append(truth * factor)
            p_common.append(median * factor)
        assert len(seen_genes) == 1000
        all_genes = np.concatenate(genes)
        recovery = float(np.mean(recoveries))
        ref = reference.loc[reporter]
        for name, ys, ps in [('common_control', y_common, p_common),
                             ('fold_standardized', y_fold, p_fold)]:
            fidelity = replay.fidelity(replay.by_gene(np.concatenate(ys), all_genes),
                                       replay.by_gene(np.concatenate(ps), all_genes))
            row = replay.decorate(dict(reporter_slug=reporter, short_name=ref.short_name,
                                       prediction_state='checkpoint_average',
                                       predictor='coherent_median_ensemble', response_scale=name,
                                       recoverability_r=recovery, reliability=float(ref.reliability),
                                       **fidelity))
            if name == 'common_control':
                for metric in replay.METRICS:
                    assert abs(row[metric] - float(ref[metric])) < 1e-10, (reporter, metric)
                assert row['tier'] == ref.tier
            records.append(row)
        print(f'{reporter}: common-scale reference reproduced', flush=True)
    output = pd.DataFrame(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.groupby(['response_scale', 'tier']).size().to_string())
    matrix = output.pivot(index='reporter_slug', columns='response_scale', values='tier')
    print(matrix.loc[matrix.common_control.ne(matrix.fold_standardized)].to_string())


if __name__ == '__main__':
    main()
