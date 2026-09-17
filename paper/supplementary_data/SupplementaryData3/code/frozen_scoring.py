"""Score frozen held-out metabolic-loss priorities; never fit or issue Q labels."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FRACTIONS = [.05,.10,.20]
REPEATS = 2000
SEED = 20260911

def top_indices(values,k,ids):
    return np.lexsort((np.asarray(ids),-np.asarray(values)))[:k]

def recall(actual,guess,k,ids):
    a=top_indices(actual,k,ids)
    b=top_indices(guess,k,ids)
    return np.intersect1d(a,b).size/k

def contexts(predictions):
    rows=[]
    for model,p in predictions.groupby("model"):
        runs=p.groupby(["Metadata_Compound","Metadata_source","Metadata_Concentration"])[["response_true","response_pred"]].mean().reset_index()
        sizes=runs.groupby(["Metadata_Compound","Metadata_source"]).size()
        complete=sizes.eq(8).groupby(level=0).sum()
        paired=set(complete[complete.eq(2)].index)
        assert len(paired)==94
        dosemeans=runs.groupby(["Metadata_Compound","Metadata_Concentration"])[["response_true","response_pred"]].mean().reset_index()
        assert dosemeans.groupby("Metadata_Compound").size().eq(8).all()
        means=dosemeans.groupby("Metadata_Compound")[["response_true","response_pred"]].mean()
        pairs=runs.groupby("Metadata_Compound").Metadata_source.agg(lambda x:"|".join(sorted(set(x))))
        for context,ids in [("A_all_heldout",means.index),("B_complete_paired",sorted(paired))]:
            for compound,row in means.loc[ids].iterrows():
                rows.append({"context":context,"compound":compound,"model":model,
                             "source_pair":pairs[compound],"reference_loss":-row.response_true,
                             "prediction_loss":-row.response_pred,"repeat_loss":np.nan})
        pr=runs[runs.Metadata_Compound.isin(paired)]
        source_means=pr.groupby(["Metadata_Compound","Metadata_source"])[["response_true","response_pred"]].mean()
        for compound in sorted(paired):
            group=source_means.loc[compound].sort_index()
            assert len(group)==2
            # Pair actual concentration values rather than ordinal dose counts alone.
            block=pr[pr.Metadata_Compound.eq(compound)]
            concentrations=[sorted(g.Metadata_Concentration) for _,g in block.groupby("Metadata_source")]
            assert concentrations[0]==concentrations[1]
            for source in [0,1]:
                r={"context":f"C_source{source}_to_source{1-source}","compound":compound,"model":model,
                   "source_pair":pairs[compound],"reference_loss":-group.iloc[1-source].response_true,
                   "prediction_loss":-group.iloc[source].response_pred,
                   "repeat_loss":-group.iloc[source].response_true}
                rows.append(r)
                rows.append({**r,"context":r["context"]+"__"+pairs[compound]})
    frame=pd.DataFrame(rows)
    assert not frame.duplicated(["context","compound","model"]).any()
    return frame

def evaluate(frame):
    scores,selections=[],[]
    for (context,model),g in frame.groupby(["context","model"]):
        g=g.sort_values("compound").reset_index(drop=True)
        ids=g.compound.to_numpy()
        actual=g.reference_loss.to_numpy()
        predicted=g.prediction_loss.to_numpy()
        repeated=g.repeat_loss.to_numpy()
        n=len(g)
        has_repeat=np.isfinite(repeated).all()
        rank=float(spearmanr(actual,predicted).statistic) if np.unique(predicted).size>1 else np.nan
        r=float(np.corrcoef(actual,predicted)[0,1]) if np.std(predicted)>0 else np.nan
        ks=[math.ceil(n*f) for f in FRACTIONS]
        # Same bootstrap draws across predictors and budgets within a context.
        rng=np.random.default_rng(SEED)
        boot=np.empty((REPEATS,len(ks)))
        delta=np.full_like(boot,np.nan)
        bootstrap_ids=np.arange(n)
        for iteration in range(REPEATS):
            ix=rng.integers(0,n,n)
            for j,k in enumerate(ks):
                boot[iteration,j]=recall(actual[ix],predicted[ix],k,bootstrap_ids)
                if has_repeat:
                    delta[iteration,j]=boot[iteration,j]-recall(actual[ix],repeated[ix],k,bootstrap_ids)
        for j,(fraction,k) in enumerate(zip(FRACTIONS,ks)):
            actual_top=top_indices(actual,k,ids)
            predicted_top=top_indices(predicted,k,ids)
            overlap=int(np.intersect1d(actual_top,predicted_top).size)
            recall_value=overlap/k
            null=hypergeom(n,k,k)
            lo,hi=np.quantile(boot[:,j],[.025,.975])
            row={"context":context,"model":model,"fraction":fraction,"n_compounds":n,"k":k,
                 "overlap":overlap,"recall":recall_value,"recall_lower95":lo,"recall_upper95":hi,
                 "random_expected_overlap":k*k/n,"random_expected_recall":k/n,
                 "enrichment_over_random":recall_value/(k/n),
                 "random_overlap_lower95":float(null.ppf(.025)),"random_overlap_upper95":float(null.ppf(.975)),
                 "random_selection_tail_probability":float(null.sf(overlap-1)),
                 "signed_spearman":rank,"signed_pearson":r,
                 "selected_reference_loss_mean":float(actual[predicted_top].mean()),
                 "selected_reference_loss_min":float(actual[predicted_top].min()),
                 "selected_reference_loss_max":float(actual[predicted_top].max()),
                 "true_top_loss_mean":float(actual[actual_top].mean()),
                 "n_bootstrap":REPEATS,"interpretation":"use_specific_retention_not_substitution_certificate"}
            if has_repeat:
                ref_recall=recall(actual,repeated,k,ids)
                dl,du=np.quantile(delta[:,j],[.025,.975])
                row.update({"measured_repeat_recall":ref_recall,"measured_repeat_overlap":int(round(ref_recall*k)),
                            "recall_difference_from_repeat":recall_value-ref_recall,
                            "difference_lower95":dl,"difference_upper95":du})
            scores.append(row)
            actual_set=set(actual_top)
            predicted_set=set(predicted_top)
            actual_order=np.lexsort((ids,-actual))
            predicted_order=np.lexsort((ids,-predicted))
            ranks_a=np.empty(n,int); ranks_a[actual_order]=np.arange(1,n+1)
            ranks_p=np.empty(n,int); ranks_p[predicted_order]=np.arange(1,n+1)
            for i in range(n):
                selections.append({"context":context,"model":model,"fraction":fraction,
                    "compound":ids[i],"source_pair":g.source_pair.iloc[i],
                    "reference_loss":actual[i],"predicted_loss":predicted[i],
                    "reference_rank":ranks_a[i],"predicted_rank":ranks_p[i],
                    "reference_top":i in actual_set,"predicted_top":i in predicted_set,
                    "recovered":i in actual_set and i in predicted_set})
        print(f"Scored {context}: {model}, N={n}",flush=True)
    return pd.DataFrame(scores),pd.DataFrame(selections)

def correspondence_null(predictions,frame,scores):
    output,distributions=[],[]
    for model in ["BF_Ridge100","BF_HGB100"]:
        p=predictions[predictions.model.eq(model)].sort_values("Metadata_well_id").reset_index(drop=True)
        ids=sorted(p.Metadata_Compound.unique())
        lookup={c:i for i,c in enumerate(ids)}
        comp=p.Metadata_Compound.map(lookup).to_numpy()
        source_keys=["Metadata_Compound","Metadata_Concentration","Metadata_source"]
        within_source=p.groupby(source_keys).Metadata_well_id.transform("size").to_numpy()
        sources_at_dose=p.groupby(source_keys[:-1]).Metadata_source.transform("nunique").to_numpy()
        n_doses=p.groupby("Metadata_Compound").Metadata_Concentration.transform("nunique").to_numpy()
        weights=1/within_source/sources_at_dose/n_doses
        assert np.allclose(np.bincount(comp,weights=weights),1)
        targets=frame[frame.context.eq("A_all_heldout") & frame.model.eq(model)].set_index("compound").loc[ids]
        actual=targets.reference_loss.to_numpy()
        raw=p.response_pred.to_numpy()
        real=-np.bincount(comp,weights=raw*weights,minlength=len(ids))
        assert np.allclose(real,targets.prediction_loss,rtol=1e-12,atol=1e-12)
        strata=[np.asarray(ix) for ix in p.groupby(["Metadata_Plate","dose_position"]).indices.values()]
        movable=sum(len(ix) for ix in strata if len(ix)>1)/len(p)
        rng=np.random.default_rng(SEED)
        null_recall=np.empty((REPEATS,len(FRACTIONS)))
        for iteration in range(REPEATS):
            shuffled=raw.copy()
            for ix in strata:
                shuffled[ix]=raw[rng.permutation(ix)]
            loss=-np.bincount(comp,weights=shuffled*weights,minlength=len(ids))
            for j,fraction in enumerate(FRACTIONS):
                k=math.ceil(len(ids)*fraction)
                null_recall[iteration,j]=recall(actual,loss,k,ids)
                distributions.append({"model":model,"iteration":iteration,"fraction":fraction,
                                      "recall":null_recall[iteration,j]})
        for j,fraction in enumerate(FRACTIONS):
            observed=float(scores.loc[scores.context.eq("A_all_heldout") & scores.model.eq(model) & scores.fraction.eq(fraction),"recall"].iloc[0])
            lo,hi=np.quantile(null_recall[:,j],[.025,.975])
            output.append({"context":"A_all_heldout","model":model,"fraction":fraction,
                           "observed_recall":observed,"null_mean_recall":float(null_recall[:,j].mean()),
                           "null_lower95":lo,"null_upper95":hi,"n_permutations":REPEATS,
                           "upper_tail_fraction_plus_one":float((1+(null_recall[:,j]>=observed).sum())/(REPEATS+1)),
                           "movable_well_fraction":movable,"n_strata":len(strata),
                           "null_type":"fixed_model_within_plate_correspondence_not_refit"})
    return pd.DataFrame(output),pd.DataFrame(distributions)

def main():
    freeze=json.loads((ROOT/"scoring_freeze.json").read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==freeze["scoring_sha256"]
    assert hashlib.sha256((ROOT/"PROTOCOL.md").read_bytes()).hexdigest()==freeze["protocol_sha256"]
    if (OUT/"utility_scores.csv").exists():
        raise FileExistsError("Do not silently overwrite the first final-test score record")
    pred=pd.read_parquet(OUT/"heldout_predictions.parquet")
    assert pred.Metadata_Compound.nunique()==217
    frame=contexts(pred)
    frame.to_csv(OUT/"context_compound_scores.csv",index=False)
    scores,selections=evaluate(frame)
    scores.to_csv(OUT/"utility_scores.csv",index=False)
    selections.to_csv(OUT/"selection_ledger.csv",index=False)
    null,distribution=correspondence_null(pred,frame,scores)
    null.to_csv(OUT/"correspondence_null_summary.csv",index=False)
    distribution.to_csv(OUT/"correspondence_null_distribution.csv.gz",index=False,compression="gzip")
    print(scores.loc[scores.model.eq("BF_Ridge100") & scores.fraction.eq(.05),
        ["context","n_compounds","k","overlap","recall","recall_lower95","recall_upper95","measured_repeat_recall","recall_difference_from_repeat"]].to_string(index=False),flush=True)
    print(null.to_string(index=False),flush=True)

if __name__ == "__main__":
    main()
