#!/usr/bin/env python3
"""Optional replay from a separately supplied archive of frozen OPS predictions.

All input paths are resolved beneath --mm-root. No fitting, model selection or
prediction writes occur. Full scale-audit replay requires saved raw controls,
not microscopy images; the table-only rebuild needs no original prediction files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MM = None
from build_tables import tier
from common_scale import control_scale

METRICS = ['recoverability_r', 'magnitude_spearman', 'variance_ratio', 'top5pct_recall', 'reliability']
TIERS = ['quantitative_proxy', 'ranking_proxy', 'measurement_required', 'not_identifiable', 'unresolved']
PROXY = {'quantitative_proxy', 'ranking_proxy'}
SOURCES: dict[str, dict] = {}


def source(path: Path, *, hash_content: bool = True) -> Path:
    key = str(path)
    locator = path.relative_to(MM).as_posix() if MM is not None and path.is_relative_to(MM) else path.relative_to(ROOT).as_posix()
    if key not in SOURCES:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if hash_content else None
        SOURCES[key] = dict(path=locator, bytes=stat.st_size, sha256=digest)
    return path


def load(path: Path) -> np.ndarray:
    value = np.load(source(path), allow_pickle=False)
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite array: {path}')
    return value


def profile_paths(method: str, fold: int, reporter: str, state: str):
    specialist = {'ridge': 'ops_reporter_specialists_v1', 'mlp': 'ops_reporter_specialists_v1',
                  'gbdt': 'ops_reporter_specialists_gbdt_v1',
                  'catboost_multirmse': 'ops_reporter_specialists_catboost_multirmse_v1'}
    if method in specialist:
        root = MM / 'results' / specialist[method] / 'gene_holdout_main' / f'fold_{fold}' / reporter
        shared = root / 'shared'
        gp = shared / 'gene_profiles'
        return (gp/'gene_code.npy', gp/'screen_code.npy',
                root/'models'/method/'gene_prediction_control_relative_scaled.npy',
                gp/'truth_control_relative_scaled.npy', shared/'target_feature_names.txt',
                shared/'manifest.json')
    if method == 'scpair_ops':
        root = MM/'results/ops_recent_bio_full_label_acp4_v1'/method/'formal/gene_holdout_main'/f'fold_{fold}'/reporter
    else:
        root = MM/'results/ops_eight_comparator_full52_defaults_acp4_v1'/method/'formal/gene_holdout_main'/f'fold_{fold}'
    block = root/'evaluation_states'/state/'reporters'/reporter
    return (block/'gene_gene_code.npy', block/'gene_screen_code.npy',
            block/'gene_prediction_control_relative_scaled.npy',
            block/'gene_truth_control_relative_scaled.npy', block/'target_feature_names.txt', None)


def macro_r(y: np.ndarray, p: np.ndarray) -> float:
    yc, pc = y-y.mean(axis=0), p-p.mean(axis=0)
    den = np.sqrt(np.square(yc).sum(axis=0)*np.square(pc).sum(axis=0))
    valid = den > 1e-12
    if not valid.all():
        raise ValueError('Constant endpoint in prediction/truth; audit aggregation before proceeding')
    return float(np.mean(np.sum(yc*pc, axis=0)/den))


def by_gene(values: np.ndarray, genes: np.ndarray) -> np.ndarray:
    # Input keys are gene×screen with each gene confined to one outer test fold.
    unique, inverse = np.unique(genes, return_inverse=True)
    output = np.zeros((len(unique), values.shape[1]), dtype=np.float64)
    np.add.at(output, inverse, values)
    return output / np.bincount(inverse)[:, None]


def fidelity(y: np.ndarray, p: np.ndarray) -> dict:
    ym, pm = np.linalg.norm(y, axis=1), np.linalg.norm(p, axis=1)
    if len(ym) != 1000:
        raise ValueError(f'Expected 1000 held-out genes, got {len(ym)}')
    # np.unique in by_gene sorts numeric gene codes; stable sorting keeps this
    # fixed order when magnitudes tie. Boundary ties are counted explicitly.
    top_y = np.argsort(-ym, kind='stable')[:50]
    top_p = np.argsort(-pm, kind='stable')[:50]
    return dict(magnitude_spearman=float(spearmanr(ym, pm).statistic),
                variance_ratio=float(np.var(pm)/np.var(ym)),
                top5pct_recall=float(np.intersect1d(top_y, top_p).size/50),
                n_genes=len(ym),
                observed_boundary_ties=int(np.sum(ym == ym[top_y[-1]])),
                predicted_boundary_ties=int(np.sum(pm == pm[top_p[-1]])))


def decorate(record: dict) -> dict:
    record['tier'] = tier(pd.Series(record))
    record['proxy_supported'] = record['tier'] in PROXY
    gates = dict(reliability=record['reliability'] >= .30,
                 recovery=record['recoverability_r'] >= .70,
                 rank=record['magnitude_spearman'] >= .70,
                 variance_lower=record['variance_ratio'] >= .50,
                 variance_upper=record['variance_ratio'] <= 1.50,
                 quantitative_hit=record['top5pct_recall'] >= .60,
                 ranking_hit=record['top5pct_recall'] >= .50,
                 measurement_recovery=record['recoverability_r'] < .60)
    record.update({f'pass_{k}': v for k, v in gates.items()})
    record['failed_quantitative_gates'] = ';'.join(k for k in
        ['reliability','recovery','rank','variance_lower','variance_upper','quantitative_hit'] if not gates[k])
    record['failed_ranking_gates'] = ';'.join(k for k in
        ['reliability','rank','ranking_hit'] if not gates[k])
    return record


def main():
    global MM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mm-root', type=Path, required=True, help='Local archive root containing the frozen results/ trees; these predictions are not included in Supplementary Data 1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reporters', help='Comma-separated subset for provenance preflight only')
    parser.add_argument('--state', default='single', choices=['single','checkpoint_average'])
    args = parser.parse_args()
    MM = args.mm_root.resolve()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    deposited = pd.read_csv(ROOT/'evidence.csv')
    definitions = pd.read_csv(ROOT/'definition_sensitivity.csv')
    order = json.loads((ROOT/'analysis_spec.json').read_text())['reporter_order']
    ledger = definitions.loc[definitions.run_state.eq('checkpoint_average') & definitions.method_id.eq('hybrid_median_before_screen_mean')].copy()
    ledger = ledger.set_index('reporter_slug').loc[order].reset_index().rename(columns={'historical_reference_tier':'substitutability_tier', 'recoverability_r':'consensus10_gene_pearson', 'variance_ratio':'magnitude_variance_ratio'})
    roster = pd.read_csv(ROOT/'method_roster.csv').sort_values('method_order')
    methods = roster.method_id.tolist()
    method_names = dict(zip(roster.method_id, roster.method))
    audit = deposited.loc[deposited.run_state.eq('single') & deposited.method_id.isin(methods)].rename(columns={'recoverability_r':'gene_macro_feature_pearson'}).set_index(['reporter_slug','method_id'])
    if args.reporters:
        ledger = ledger.loc[ledger.reporter_slug.isin(args.reporters.split(','))]
    records, fold_records, scale_records, verification, alignment = [], [], [], [], []
    for idx, row in enumerate(ledger.itertuples(index=False), 1):
        reporter = row.reporter_slug
        scale, scale_audit = control_scale(reporter, mm_root=MM)
        scale_audit['endpoint_scales'] = scale.tolist()
        for path in scale_audit['input_paths']:
            source(Path(path), hash_content=not path.endswith('truth_raw.npy'))
        scale_audit['input_paths'] = [Path(p).relative_to(MM).as_posix() for p in scale_audit['input_paths']]
        scale_audit.pop('elapsed_seconds', None)
        scale_records.append(scale_audit)
        all_genes, all_screens, all_truth, fold_stacks = [], [], [], []
        fold_r = {m: [] for m in methods}
        ensemble_fold_r = []
        names_reference = None
        gene_fold_membership = {}
        for fold in range(5):
            base = MM/'results/ops_reporter_specialists_v1/gene_holdout_main'/f'fold_{fold}'/reporter/'shared'
            manifest = json.loads(source(base/'manifest.json').read_text())
            y_scale = np.asarray(manifest['preprocessing']['y_scale'], dtype=float)
            names = source(base/'target_feature_names.txt').read_text().splitlines()
            if names_reference is None:
                names_reference = names
            assert names == names_reference and len(names) == len(scale) == len(y_scale)
            assert np.isfinite(y_scale).all() and (y_scale > 0).all()
            transform = y_scale/scale
            gene = load(base/'gene_profiles/gene_code.npy')
            screen = load(base/'gene_profiles/screen_code.npy')
            truth = load(base/'gene_profiles/truth_control_relative_scaled.npy').astype(float)
            keys = list(zip(gene.tolist(), screen.tolist()))
            assert len(set(keys)) == len(keys)
            for g in np.unique(gene):
                assert int(g) not in gene_fold_membership
                gene_fold_membership[int(g)] = fold
            preds = []
            for method in methods:
                gp, sp, pp, yp, npth, mp = profile_paths(method, fold, reporter, args.state)
                mg, ms = load(gp), load(sp)
                pred, my = load(pp).astype(float), load(yp).astype(float)
                mnames = source(npth).read_text().splitlines()
                assert mnames == names_reference, (reporter, method, fold, 'endpoint order')
                mkeys = list(zip(mg.tolist(), ms.tolist()))
                assert len(set(mkeys)) == len(mkeys) and set(keys) == set(mkeys)
                index = {key: j for j, key in enumerate(mkeys)}
                order = np.array([index[key] for key in keys])
                pred, my = pred[order], my[order]
                assert pred.shape == truth.shape == my.shape
                max_truth_delta = float(np.max(np.abs(my-truth)))
                if not np.allclose(my, truth, rtol=2e-6, atol=2e-7):
                    raise ValueError(f'Target-transform/truth mismatch {reporter} {method} {fold}: {max_truth_delta}')
                if mp:
                    mm = json.loads(source(mp).read_text())
                    assert np.array_equal(y_scale, mm['preprocessing']['y_scale'])
                    assert manifest['split_fingerprint_sha256'] == mm['split_fingerprint_sha256']
                r = macro_r(truth, pred)
                fold_r[method].append(r)
                fold_records.append(dict(reporter_slug=reporter,method_id=method,fold=fold,
                                         recoverability_r=r,n_gene_screen_pairs=len(gene),n_endpoints=len(scale)))
                alignment.append(dict(reporter_slug=reporter,method_id=method,fold=fold,
                                      n_keys=len(keys),max_truth_delta=max_truth_delta,
                                      reordered=not np.array_equal(order,np.arange(len(order)))))
                preds.append(pred*transform)
            stack = np.stack(preds)
            ensemble_fold_r.append(macro_r(truth*transform, np.median(stack, axis=0)))
            all_genes.append(gene)
            all_screens.append(screen)
            all_truth.append(truth*transform)
            fold_stacks.append(stack)
        genes = np.concatenate(all_genes)
        y = np.concatenate(all_truth)
        stack = np.concatenate(fold_stacks, axis=1)
        assert len(gene_fold_membership) == 1000
        yg = by_gene(y, genes)
        pg = np.stack([by_gene(p, genes) for p in stack])
        before_screen_mean = by_gene(np.median(stack,axis=0), genes)
        common = dict(reporter_slug=reporter,short_name=row.short_name,
                      run_state=args.state,
                      biological_category=row.biological_category,reliability=row.reliability,
                      published_tier=row.substitutability_tier)
        for m, p in zip(methods, pg):
            calculated = float(np.mean(fold_r[m]))
            reference = float(audit.loc[(reporter,m),'gene_macro_feature_pearson'])
            delta = calculated-reference
            if args.state == 'single' and abs(delta)>5e-6:
                raise ValueError(f'Recovery replay mismatch {reporter} {m}: {calculated} != {reference}')
            verification.append(dict(reporter_slug=reporter,method_id=m,check='recovery_vs_saved_single_state',
                                     calculated=calculated,reference=reference,delta=delta))
            records.append(decorate(dict(**common,method_id=m,method=method_names[m],
                                         recoverability_r=calculated,**fidelity(yg,p))))
        median_method_r=float(np.median([float(audit.loc[(reporter,m),'gene_macro_feature_pearson']) for m in methods]))
        verification.append(dict(reporter_slug=reporter,method_id='published_hybrid',check='ledger_recovery',
            calculated=median_method_r,reference=row.consensus10_gene_pearson,
            delta=median_method_r-row.consensus10_gene_pearson))
        if abs(median_method_r-row.consensus10_gene_pearson)>5e-6:
            raise ValueError(f'Published ledger recovery not reproduced: {reporter}')
        for key, p, recovery in [
            ('hybrid_median_before_screen_mean',before_screen_mean,median_method_r),
            ('state_matched_hybrid',before_screen_mean,float(np.median([np.mean(fold_r[m]) for m in methods]))),
            ('coherent_median_ensemble',before_screen_mean,float(np.mean(ensemble_fold_r)))]:
            result=decorate(dict(**common,method_id=key,method=key,recoverability_r=recovery,**fidelity(yg,p)))
            records.append(result)
            if key.startswith('hybrid_'):
                for metric, field in [('magnitude_spearman','magnitude_spearman'),
                                      ('variance_ratio','magnitude_variance_ratio'),
                                      ('top5pct_recall','top5pct_recall')]:
                    reference=float(getattr(row,field))
                    verification.append(dict(reporter_slug=reporter,method_id=key,check='ledger_'+metric,
                        calculated=result[metric],reference=reference,delta=result[metric]-reference))
        print(f'{idx}/{len(ledger)} {reporter}: aligned 10 x 5 folds, common scale restored',flush=True)
    frame=pd.DataFrame(records)
    # The optional replay must reproduce the actual same-state main evidence,
    # not merely agree with the historical mixed-state reference construction.
    expected = deposited.loc[deposited.run_state.eq(args.state)].set_index(['reporter_slug', 'method_id'])
    main_rows = frame.loc[frame.method_id.isin(methods + ['coherent_median_ensemble'])]
    for result in main_rows.itertuples(index=False):
        saved = expected.loc[(result.reporter_slug, result.method_id)]
        assert result.tier == saved.tier
        for metric in METRICS:
            delta = float(getattr(result, metric)) - float(saved[metric])
            assert abs(delta) < 1e-10, (result.reporter_slug, result.method_id, metric, delta)
    print('Same-state primary evidence replay: all metrics <1e-10 and exact tiers match', flush=True)
    frame.to_csv(out/'all_evidence_candidates.csv',index=False)
    pd.DataFrame(fold_records).to_csv(out/'fold_recoverability.csv',index=False)
    pd.DataFrame(verification).to_csv(out/'reference_replay.csv',index=False)
    pd.DataFrame(alignment).to_csv(out/'alignment_audit.csv',index=False)
    (out/'control_scale_audit.json').write_text(json.dumps(scale_records,indent=2)+'\n')
    pd.DataFrame(SOURCES.values()).to_csv(out/'source_inventory.csv',index=False)
    spec=dict(rule=json.loads((ROOT/'analysis_spec.json').read_text())['rule'],methods=methods,reporters=ledger.reporter_slug.tolist(),
              state=args.state,analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scale_script_sha256=hashlib.sha256(Path(__file__).with_name('common_scale.py').read_bytes()).hexdigest(),
              tier_script_sha256=hashlib.sha256(Path(__file__).with_name('build_tables.py').read_bytes()).hexdigest())
    (out/'run_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    errors=pd.DataFrame(verification).groupby(['method_id','check']).delta.agg(lambda v:float(np.max(np.abs(v))))
    print(errors.to_string(),flush=True)


if __name__ == '__main__':
    main()
