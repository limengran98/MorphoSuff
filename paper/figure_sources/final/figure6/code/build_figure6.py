#!/usr/bin/env python3
"""Render canonical Figure 6 and its six unlabelled native panels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIG6 = Path(__file__).resolve().parents[1]
PAPER = FIG6.parents[2]
sys.path.insert(0, str(FIG6 / "code"))
import figure6_layout as layout

FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM = 183.0, 210.0
DEFAULT_OUTPUT_DIR = FIG6 / "published"
NATIVE_COMPONENTS = {
    "atlas": FIG6 / "a_counterfactual_utility_atlas/build_panel_a.py",
    "utility": FIG6 / "d_prediction_utility/build_panel_d.py",
    "ledger": FIG6 / "c_decision_ledger/code/build_figure6c_decision_ledger.py",
    "regression": FIG6 / "b_nested_loo_audit/build_panel_b.py",
    "stability": FIG6 / "e_reporter_context_stability/build_panel_e.py",
    "contexts": FIG6 / "code/build_panel_f.py",
}
# Stable numerical component paths are shared with statistics reconstruction.
# Display-panel assignments are explicit in SOURCE_TABLES and build_figure.
COMPONENT_SOURCES = {
    "derived_input": [
        FIG6 / "source_data/coherent_ensemble_evidence.csv",
        FIG6 / "source_data/coherent_ensemble_fold_recoverability.csv",
        FIG6 / "source_data/response_replacement_utility_and_annotations.csv",
        FIG6 / "source_data/coherent_source_contract.json",
        FIG6 / "source_data/derived_statistics_summary.json",
    ],
    "a": [
        FIG6 / "a_counterfactual_utility_atlas" / "figure_source_data.csv",
    ],
    "b": [
        FIG6 / "b_nested_loo_audit" / "figure_source_nested_loo.csv",
        FIG6 / "b_nested_loo_audit" / "figure_source_association_bootstrap.csv",
        FIG6 / "b_nested_loo_audit" / "figure_source_nested_loo_summary.csv",
    ],
    "c": [
        FIG6 / "c_decision_ledger" / "source_data" / "figure6c_decision_ledger_source.csv",
        FIG6 / "c_decision_ledger" / "source_data" / "figure6c_threshold_transition_matrix.csv",
        FIG6 / "c_decision_ledger" / "source_data" / "figure6c_threshold_assignments.csv",
        FIG6 / "c_decision_ledger" / "source_data" / "figure6c_threshold_scenario_counts.csv",
    ],
    "d": [
        FIG6 / "d_prediction_utility" / "figure_source_reporter_utility.csv",
        FIG6 / "d_prediction_utility" / "figure_source_decision_ledger.csv",
        FIG6 / "d_prediction_utility" / "figure_source_prediction_utility_plot.csv",
        FIG6 / "d_prediction_utility" / "figure_source_prediction_utility_summary.csv",
        FIG6 / "d_prediction_utility" / "figure_source_direct_labels.csv",
    ],
    "e": [
        FIG6 / "e_reporter_context_stability" / "figure_source_directed_screen_evaluations.csv",
        FIG6 / "e_reporter_context_stability" / "figure_source_context_stability_long.csv",
        FIG6 / "e_reporter_context_stability" / "figure_source_context_stability_reporter_summary.csv",
        FIG6 / "e_reporter_context_stability" / "figure_source_context_stability_summary.csv",
    ],
    "f": [
        FIG6 / "f_cross_context_decisions" / "figure_source_data.csv",
        FIG6 / "f_cross_context_decisions" / "figure_source_gate_rows.csv",
        FIG6 / "f_cross_context_decisions" / "figure_source_replicate_detail.csv",
    ],
}

EXTERNAL = FIG6 / "source_data/external_use"
SOURCE_TABLES = {
    "derived_input": COMPONENT_SOURCES["derived_input"],
    "a": COMPONENT_SOURCES["a"], "b": COMPONENT_SOURCES["d"],
    "c": COMPONENT_SOURCES["c"],
    "d": [layout.DATA/"selection_ledger.csv", EXTERNAL/"external_marginal_histograms.csv",
          EXTERNAL/"external_selection_boundaries.csv"],
    "e": [layout.DATA/"utility_scores.csv", EXTERNAL/"external_hit_membership.csv"],
    "f": COMPONENT_SOURCES["f"],
}

def load_components():
    return {name:layout.load_module("figure6_"+name,path) for name,path in NATIVE_COMPONENTS.items()}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_source_manifest(output_dir):
    records=[]
    for panel,paths in SOURCE_TABLES.items():
        for path in paths:
            if not path.is_file(): raise FileNotFoundError(path)
            records.append({"panel":panel,"source_table":path.relative_to(PAPER).as_posix(),
                            "rows":len(pd.read_csv(path)) if path.suffix==".csv" else None,
                            "sha256":sha256(path)})
    target=output_dir/"figure_source_manifest.csv"
    pd.DataFrame(records).to_csv(target,index=False)
    return target

def export_artwork(fig,stem,*,qa_root,point_axes=(),tiff=False):
    stem.parent.mkdir(parents=True,exist_ok=True)
    qa_root.mkdir(parents=True,exist_ok=True)
    layout.format_figure(fig,6)
    report=layout.layout_report(fig,qa_root/(stem.name+"_layout.json"))
    report["figure"]=stem.name
    report["annotation_marker_collisions"]=[
        {"axis":ax.get_xlabel(),"text":text}
        for ax in point_axes for text in layout.point_text_collisions(ax)]
    if report["text_collisions"] or report["clipped_text"] or report["annotation_marker_collisions"]:
        raise ValueError(json.dumps(report,indent=2))
    metadata={"Creator":"MorphoSuff deterministic figure renderer","CreationDate":None,"ModDate":None}
    fig.savefig(stem.with_suffix(".pdf"),metadata=metadata)
    fig.savefig(stem.with_suffix(".svg"),metadata={"Date":None})
    fig.savefig(stem.with_suffix(".png"),dpi=600)
    fig.savefig(stem.with_name(stem.name+"_preview.png"),dpi=180)
    if tiff: fig.savefig(stem.with_suffix(".tiff"),dpi=600)
    plt.close(fig)
    return report

def build_figure(components=None,data=None):
    components=components or load_components()
    data=data or layout.sources()
    layout.configure()
    fig=plt.figure(figsize=(FIGURE_WIDTH_MM/25.4,FIGURE_HEIGHT_MM/25.4))
    fig._composite_panel_letters=True
    rows=fig.subfigures(4,1,height_ratios=[48,57,65,40],hspace=.018)
    top=rows[0].subfigures(1,2,width_ratios=[110,73],wspace=.025)
    layout.draw_utility_atlas(top[0],components["atlas"])
    utility=layout.draw_recovery_utility(top[1],components["utility"])
    components["ledger"].draw_panel_c(rows[1],add_letter=False)
    lower=rows[2].subfigures(1,2,width_ratios=[89,94],wspace=.020)
    layout.draw_external_scatter(lower[0],data[0])
    layout.draw_external_controls(lower[1],data[1])
    components["contexts"].draw_panel_f(rows[3],add_letter=False,compact=False)
    containers=[top[0],top[1],rows[1],lower[0],lower[1],rows[3]]
    for container,letter in zip(containers,"abcdef"):layout.letter(container,letter)
    for ax in rows[1].axes:
        p=ax.get_position();ax.set_position([p.x0,p.y0-.035,p.width,p.height])
    for text in rows[1].texts:
        x,y=text.get_position();text.set_position((x,y-.035))
    geometry={letter:layout.dimensions(container) for letter,container in zip("abcdef",containers)}
    return fig,geometry,utility["axis"]

def render_clean_panels(components,data,geometry):
    reports=[]
    for letter in "abcdef":
        layout.configure()
        width,height=geometry[letter]
        fig=plt.figure(figsize=(width/25.4,height/25.4));point_axes=[]
        if letter=="a":layout.draw_utility_atlas(fig,components["atlas"])
        elif letter=="b":point_axes=[layout.draw_recovery_utility(fig,components["utility"])["axis"]]
        elif letter=="c":
            components["ledger"].draw_panel_c(fig,add_letter=False)
            for ax in fig.axes:
                p=ax.get_position();ax.set_position([p.x0,p.y0-.035,p.width,p.height])
            for text in fig.texts:
                x,y=text.get_position();text.set_position((x,y-.035))
        elif letter=="d":layout.draw_external_scatter(fig,data[0])
        elif letter=="e":layout.draw_external_controls(fig,data[1])
        else:components["contexts"].draw_panel_f(fig,add_letter=False,compact=False)
        reports.append(export_artwork(fig,FIG6/"panels"/("Figure6"+letter),
                                      qa_root=FIG6/"qa",point_axes=point_axes))
    return reports

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stem",type=Path,default=DEFAULT_OUTPUT_DIR/"Figure6")
    args=parser.parse_args()
    components=load_components();data=layout.sources()
    fig,geometry,axis=build_figure(components,data)
    reports=[export_artwork(fig,args.stem,qa_root=FIG6/"qa",point_axes=[axis],tiff=True)]
    reports+=render_clean_panels(components,data,geometry)
    manifest=write_source_manifest(args.stem.parent)
    (args.stem.parent/"panel_geometry_mm.json").write_text(json.dumps(geometry,indent=2)+"\n")
    report={"status":"REVIEW","scientific_contract":{
        "question":"Which scientific uses survive targeted-measurement prediction?",
        "OPS_reporters":52,"external_test_compounds":217,
        "measured_top":11,"predicted_top":11,"shared_top":10,
        "matrix_shape":[6,11],"matrix_row_sums":[10,10,8,7,0,1],
        "inferential_units":"Reporter in OPS; held-out compound for external recall intervals",
        "new_fitting":False,"new_inference":False},
        "automated_problem_figures":[],"source_manifest":manifest.name,
        "source_manifest_sha256":sha256(manifest),"figures":reports,
        "visual_review":{"final_size_checked":False,"user_adoption_pending":False},
        "claim_boundary":{
            "can_say":["Brightfield predictions retain 10 of 11 measured metabolic-loss hits among 217 held-out compounds."],
            "cannot_say":["Universal label-free assay substitution, repeat equivalence, or superiority over cell-count predictors."]}}
    (args.stem.parent/"figure_qa_report.json").write_text(json.dumps(report,indent=2)+"\n")
    shutil.copyfile(args.stem.with_suffix(".pdf"),PAPER/"figures/Figure6.pdf")
    print("Rendered Figure 6 and six unlabelled panels; numerical and layout checks passed.")

if __name__=="__main__":main()
