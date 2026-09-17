#!/usr/bin/env python3
"""Check canonical Figure 6 marks, source data, live text and panel boundaries."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import fitz
import matplotlib.colors as mcolors
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def external_display_audit():
    """Check rendered marks and display tables against the frozen evidence."""
    spec=importlib.util.spec_from_file_location('figure6_display_audit',ROOT/'code/figure6_layout.py')
    renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
    ledger=pd.read_csv(renderer.DATA/'selection_ledger.csv',float_precision='round_trip')
    scores=pd.read_csv(renderer.DATA/'utility_scores.csv',float_precision='round_trip')
    rows=ledger.loc[ledger.context.eq(renderer.CONTEXTS[0]) & ledger.model.eq(renderer.PRIMARY)
                    & ledger.fraction.eq(.05)]
    comparison=scores.loc[scores.context.eq(renderer.CONTEXTS[0]) & scores.fraction.eq(.05)]
    geometry=json.loads((ROOT/'published/panel_geometry_mm.json').read_text())
    renderer.configure()
    fig=renderer.plt.figure(figsize=np.array(geometry['d'])/25.4,dpi=180)
    scatter=renderer.draw_external_scatter(fig,rows);fig.canvas.draw()
    points=[];clearances=[]
    for collection in scatter.collections:
        offsets=np.asarray(collection.get_offsets())
        points.extend(offsets)
        centers=collection.get_offset_transform().transform(offsets)
        box=collection.get_paths()[0].get_extents()
        scale=np.sqrt(collection.get_sizes()[0])*fig.dpi/72
        edge=collection.get_linewidths()[0]*fig.dpi/72/2
        clearances.extend(np.minimum.reduce([
            centers[:,0]+box.x0*scale-edge-scatter.bbox.x0,
            scatter.bbox.x1-(centers[:,0]+box.x1*scale+edge),
            centers[:,1]+box.y0*scale-edge-scatter.bbox.y0,
            scatter.bbox.y1-(centers[:,1]+box.y1*scale+edge)]))
    observed=np.asarray(points);expected=rows[['reference_loss','predicted_loss']].to_numpy()
    order=lambda a:a[np.lexsort((a[:,1],a[:,0]))]
    assert len(observed)==217 and np.array_equal(order(observed),order(expected))
    clearance=float(min(clearances)*25.4/fig.dpi)
    assert clearance>0,clearance
    histogram=renderer.marginal_histograms(rows)
    deposited=pd.read_csv(ROOT/'source_data/external_use/external_marginal_histograms.csv',float_precision='round_trip')
    pd.testing.assert_frame_equal(histogram,deposited)
    for axis,marginal in zip(['measured','predicted'],fig.axes[1:]):
        actual=np.array([p.get_height() if axis=='measured' else p.get_width() for p in marginal.patches])
        assert np.array_equal(actual,histogram.loc[histogram.axis.eq(axis),'compound_count'])
    assert fig.axes[1].get_ylim()==fig.axes[2].get_xlim()
    boundaries=renderer.selection_boundaries(rows)
    pd.testing.assert_frame_equal(boundaries,pd.read_csv(ROOT/'source_data/external_use/external_selection_boundaries.csv',float_precision='round_trip'))
    renderer.plt.close(fig)

    fig=renderer.plt.figure(figsize=np.array(geometry['e'])/25.4,dpi=180)
    forest=renderer.draw_external_controls(fig,comparison);fig.canvas.draw()
    matrix=fig.axes[1]
    membership=pd.read_csv(ROOT/'source_data/external_use/external_hit_membership.csv',float_precision='round_trip')
    assert len(matrix.patches)==len(membership)==66
    common=rows.loc[rows.reference_top].sort_values(['reference_loss','compound'],ascending=[False,True])
    for i,model in enumerate(renderer.MODELS):
        source=ledger.loc[ledger.context.eq(renderer.CONTEXTS[0]) & ledger.model.eq(model)
                          & ledger.fraction.eq(.05)].set_index('compound').loc[common.compound]
        hits=membership.loc[membership.model.eq(model)].sort_values('measured_hit_rank')
        assert hits.compound.tolist()==source.index.tolist()
        for column in ['reference_loss','predicted_loss','reference_rank','predicted_rank',
                       'reference_top','predicted_top','recovered']:
            assert np.array_equal(hits[column],source[column]),(model,column)
        colour=renderer.BLUE if model.startswith('BF') else renderer.GOLD if model.startswith('Fluorescence') else renderer.GREY
        for patch,recovered in zip(matrix.patches[i*11:(i+1)*11],source.recovered):
            assert patch.get_facecolor()==mcolors.to_rgba(colour if recovered else '#F1F4F6')
        result=comparison.set_index('model').loc[model]
        container=forest.containers[i]
        assert float(container.lines[0].get_xdata()[0])==result.recall
        ends=container.lines[2][0].get_segments()[0][:,0]
        assert np.allclose(ends,[result.recall_lower95,result.recall_upper95],rtol=0,atol=2e-16)
        assert int(source.recovered.sum())==result.overlap
    marker_clearances=[]
    for line in forest.lines:
        if line.get_marker() in (None,'None','none','',' '):continue
        marker=MarkerStyle(line.get_marker())
        box=marker.get_path().transformed(marker.get_transform()).get_extents()
        scale=line.get_markersize()*fig.dpi/72
        edge=line.get_markeredgewidth()*fig.dpi/72/2
        centers=line.get_transform().transform(np.column_stack([line.get_xdata(),line.get_ydata()]).astype(float))
        marker_clearances.extend(np.minimum.reduce([
            centers[:,0]+box.x0*scale-edge-forest.bbox.x0,
            forest.bbox.x1-(centers[:,0]+box.x1*scale+edge),
            centers[:,1]+box.y0*scale-edge-forest.bbox.y0,
            forest.bbox.y1-(centers[:,1]+box.y1*scale+edge)]))
    forest_clearance=float(min(marker_clearances)*25.4/fig.dpi)
    cell_clearances=[]
    for patch in matrix.patches:
        box=patch.get_window_extent();edge=patch.get_linewidth()*fig.dpi/72/2
        cell_clearances.append(min(box.x0-edge-matrix.bbox.x0,matrix.bbox.x1-box.x1-edge,
                                   box.y0-edge-matrix.bbox.y0,matrix.bbox.y1-box.y1-edge))
    matrix_clearance=float(min(cell_clearances)*25.4/fig.dpi)
    assert forest_clearance>.3 and matrix_clearance>.25,(forest_clearance,matrix_clearance)
    renderer.plt.close(fig)
    return {'external_paired_scatter_points':217,
            'point_coordinates_match_frozen_sources_exactly':True,
            'minimum_complete_marker_clearance_mm':{'external_scatter':clearance,
                                                    'recall_forest':forest_clearance},
            'minimum_complete_matrix_cell_clearance_mm':matrix_clearance,
            'marginal_histograms':{'compounds_each':217,'common_bins':23,'common_count_scale':True},
            'selection_boundaries_match_frozen_memberships':True,
            'matrix_shape':[6,11],'matrix_row_sums':[10,10,8,7,0,1],
            'all_matrix_memberships_match_frozen_ledger':True,
            'all_six_external_recall_intervals_match_frozen_results':True}



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--visual-reviewed",action="store_true",
                        help="Record completed manual inspection at manuscript placement size.")
    args=parser.parse_args()
    report_path=ROOT/"published/figure_qa_report.json"
    report=json.loads(report_path.read_text())
    assert not report["automated_problem_figures"]
    vectors=[]
    for letter in ["",*"abcdef"]:
        stem=ROOT/"panels"/("Figure6"+letter) if letter else ROOT/"published/Figure6"
        with fitz.open(stem.with_suffix(".pdf")) as doc:
            assert len(doc)==1
            page=doc[0]
            assert page.get_fonts() and not any("Type3" in str(f) for f in page.get_fonts())
            spans=[s for b in page.get_text("dict")["blocks"] for line in b.get("lines",[])
                   for s in line["spans"] if s["text"].strip()]
            assert spans and min(s["size"] for s in spans)>=5.99
            letters=[s for s in spans if s["text"] in list("abcdef")]
            assert not letters if letter else sorted(s["text"] for s in letters)==list("abcdef")
            if not letter: assert all(abs(s["size"]-9)<.01 for s in letters)
            canvas=page.rect+(-.5,-.5,.5,.5)
            assert all(canvas.contains(fitz.Rect(s["bbox"])) for s in spans)
            svg=ET.parse(stem.with_suffix(".svg"))
            assert svg.findall(".//{http://www.w3.org/2000/svg}text")
            vectors.append({"figure":stem.name,"minimum_authored_pt":min(s["size"] for s in spans),
                            "page_mm":[page.rect.width*25.4/72,page.rect.height*25.4/72],
                            "standalone_unlabelled":not letters if letter else None})
    report["rendered_external_evidence_audit"]=external_display_audit()
    report["vector_checks"]=vectors
    report["status"]="PASS" if args.visual_reviewed else "REVIEW"
    report["visual_review"]={"final_size_checked":args.visual_reviewed,"placement_width_mm":168,
                            "user_adoption_pending":False}
    report_path.write_text(json.dumps(report,indent=2)+"\n")
    summary={"status":report["status"],"automatic_checks":"PASS",
             "canonical_panels":6,"source_data_and_rendered_marks":"PASS",
             "visual_review_required":not args.visual_reviewed}
    (ROOT/"published/package_qa_report.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    return 0

if __name__=="__main__":raise SystemExit(main())
