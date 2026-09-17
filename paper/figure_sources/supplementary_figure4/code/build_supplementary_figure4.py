#!/usr/bin/env python3
"""Render supporting utility, context and external-prioritization analyses."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import sys
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT.parents[1]
FIG6=PAPER/"figure_sources/final/figure6"
sys.path.insert(0,str(FIG6/"code"))
import build_figure6 as native
import figure6_layout as layout

WIDTH,HEIGHT=183.0,130.0

def source_tables(data):
    parts=[];manifest=[]
    for panel,paths in [("a",native.COMPONENT_SOURCES["b"]),("b",native.COMPONENT_SOURCES["e"])]:
        for path in paths:
            rows=pd.read_csv(path,float_precision="round_trip")
            rows.insert(0,"source_table",path.name)
            rows.insert(0,"panel",panel)
            rows.insert(2,"mark_id",[panel+":"+path.stem+":"+str(i) for i in range(len(rows))])
            parts.append(rows)
            manifest.append({"panel":panel,"source_table":path.relative_to(PAPER).as_posix(),
                             "sha256":native.sha256(path)})
    for panel,rows in [("c",data[2]),("d",data[3])]:
        rows=rows.copy();rows.insert(0,"panel",panel)
        rows.insert(1,"source_table","utility_scores.csv")
        rows.insert(2,"mark_id",panel+":"+rows.context+":"+rows.fraction.astype(str))
        parts.append(rows)
        manifest.append({"panel":panel,"source_table":(layout.DATA/"utility_scores.csv").relative_to(PAPER).as_posix(),
                         "sha256":native.sha256(layout.DATA/"utility_scores.csv")})
    pd.concat(parts,ignore_index=True).to_csv(ROOT/"figure_source_data.csv",index=False)
    pd.DataFrame(manifest).to_csv(ROOT/"source_manifest.csv",index=False)

def draw_panel(container,letter,components,data):
    layout.configure()
    if letter=="a":
        components["regression"].configure()
        components["regression"].draw_panel_b(container,add_letter=False,left_margin=.155)
    elif letter=="b":
        components["stability"].draw_panel_e(container,add_letter=False)
    elif letter=="c":layout.draw_repeat(container,data[2])
    else:layout.draw_budget(container,data[3])

def main():
    components=native.load_components();data=layout.sources()
    source_tables(data)
    layout.configure()
    fig=plt.figure(figsize=(WIDTH/25.4,HEIGHT/25.4))
    fig._composite_panel_letters=True
    rows=fig.subfigures(2,1,height_ratios=[72,56],hspace=.025)
    top=rows[0].subfigures(1,2,width_ratios=[77,106],wspace=.04)
    lower=rows[1].subfigures(1,2,width_ratios=[89,94],wspace=.02)
    containers=[*top,*lower]
    for container,letter in zip(containers,"abcd"):
        draw_panel(container,letter,components,data);layout.letter(container,letter)
    geometry={letter:layout.dimensions(container) for letter,container in zip("abcd",containers)}
    reports=[native.export_artwork(fig,ROOT/"SupplementaryFigure4",qa_root=ROOT/"qa")]
    for letter in "abcd":
        width,height=geometry[letter];layout.configure()
        fig=plt.figure(figsize=(width/25.4,height/25.4))
        draw_panel(fig,letter,components,data)
        reports.append(native.export_artwork(fig,ROOT/"panels"/("panel_"+letter),qa_root=ROOT/"qa"))
    (ROOT/"panel_geometry_mm.json").write_text(json.dumps(geometry,indent=2)+"\n")
    report={"status":"REVIEW","scientific_contract":{
        "a":"Frozen nested leave-one-reporter-out utility prediction; reporter unit",
        "b":"81 directed screen evaluations for 34 reporters; reporter/destination unit",
        "c":"Paired source-direction repeat comparisons in 94 complete compounds",
        "d":"Frozen5/10/20-percent selection budgets in four declared compound contexts",
        "no_refitting":True,"no_new_inference":True},
        "automated_problem_figures":[],"figures":reports,
        "visual_review":{"final_size_checked":False}}
    (ROOT/"figure_qa_report.json").write_text(json.dumps(report,indent=2)+"\n")
    paths=sorted(p for p in ROOT.rglob("*") if p.is_file()
                 and p.name!="manifest_sha256.csv" and "__pycache__" not in p.parts)
    pd.DataFrame([{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,
                   "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]).to_csv(ROOT/"manifest_sha256.csv",index=False)
    shutil.copyfile(ROOT/"SupplementaryFigure4.pdf",PAPER/"figures/SupplementaryFigure4.pdf")
    print("Rendered Supplementary Figure 4 and four unlabelled panels; layout checks passed.")

if __name__=="__main__":main()
