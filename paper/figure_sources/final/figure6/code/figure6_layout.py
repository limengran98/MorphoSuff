#!/usr/bin/env python3
"""Shared deterministic drawing functions for Figure 6 and Supplementary Figure 4.

All plotted values are read from deposited evidence. Panel letters are assigned
by the figure compositors, never embedded in the standalone panel exports.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

FIG6 = Path(__file__).resolve().parents[1]
PAPER = FIG6.parents[2]
ROOT = FIG6 / 'source_data/external_use'
DATA = PAPER / 'supplementary_data/SupplementaryData3/data'
sys.path.insert(0, str(PAPER / 'figure_sources'))
from publication_style import FONT, LETTER_PT, format_figure, layout_report

INK, MUTED, GRID = '#202A33', '#6F7E88', '#DCE3E7'
BLUE, GOLD, ROSE = '#3E6F9C', '#DDA06A', '#D56C75'
GREY, PALE, TEAL = '#7E8D96', '#C8D0D4', '#477F79'
W, H = 183.0, 210.0
PRIMARY = 'BF_Ridge100'
CONTEXTS = ['A_all_heldout', 'B_complete_paired', 'C_source0_to_source1', 'C_source1_to_source0']
MODELS = ['BF_Ridge100', 'BF_HGB100', 'FluorescenceCount_Ridge100',
          'FluorescenceCount_HGB100', 'AcquisitionMetadata_Ridge100', 'AcquisitionMetadata_HGB100']
LOSS_LIMITS = (-.15, 1.0)
LOSS_BINS = np.linspace(*LOSS_LIMITS, 24)
MANIFEST = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    MANIFEST[str(path.relative_to(PAPER))] = sha(path)
    return pd.read_csv(path, float_precision='round_trip')


def configure():
    mpl.rcParams.update({
        'font.family': FONT, 'font.size': 6.5, 'text.color': INK,
        'axes.labelsize': 6.5, 'axes.labelcolor': INK, 'axes.linewidth': .65,
        'xtick.labelsize': 6, 'ytick.labelsize': 6, 'xtick.color': INK,
        'ytick.color': INK, 'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 2.4, 'ytick.major.size': 2.4, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'MorphoSuff-publication',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def dimensions(container):
    return container.bbox.width / container.dpi * 25.4, container.bbox.height / container.dpi * 25.4


def axmm(container, x, y, w, h):
    fw, fh = dimensions(container)
    ax = container.add_axes([x/fw, y/fh, w/fw, h/fh])
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(pad=2)
    return ax


def txt(container, x, y, value, **kw):
    fw, fh = dimensions(container)
    return container.text(x/fw, y/fh, value, **kw)


def letter(container, value):
    _, fh = dimensions(container)
    txt(container, .7, fh-.7, value, ha='left', va='top', fontsize=LETTER_PT,
        fontweight='bold', color=INK)


def selection_boundaries(rows):
    """Display separators between ranks 11/12, not sufficiency thresholds."""
    boundaries = []
    for axis, column, selected in [('measured', 'reference_loss', 'reference_top'),
                                    ('predicted', 'predicted_loss', 'predicted_top')]:
        ordered = rows.sort_values([column, 'compound'], ascending=[False, True])
        last, next_value = ordered[column].iloc[10:12]
        assert last > next_value, 'A tied boundary requires an explicit display rule'
        boundary = float((last + next_value) / 2)
        assert np.array_equal(rows[column].ge(boundary), rows[selected])
        boundaries.append({'axis': axis, 'last_selected_loss': last,
                           'first_unselected_loss': next_value,
                           'display_separator': boundary, 'selected_compounds': 11})
    return pd.DataFrame(boundaries)


def marginal_histograms(rows):
    parts = []
    for axis, column in [('measured', 'reference_loss'), ('predicted', 'predicted_loss')]:
        counts, edges = np.histogram(rows[column], bins=LOSS_BINS)
        assert counts.sum() == len(rows) == 217
        parts.append(pd.DataFrame({'axis': axis, 'bin_left': edges[:-1],
                                   'bin_right': edges[1:], 'compound_count': counts}))
    return pd.concat(parts, ignore_index=True)


def prepare_external_display(ledger, scatter, scores):
    """Expose existing selection membership and descriptive bin counts."""
    common = scatter.loc[scatter.reference_top].sort_values(
        ['reference_loss', 'compound'], ascending=[False, True])
    assert common.reference_rank.tolist() == list(range(1, 12))
    hit_rows = []
    reference = scatter.set_index('compound').sort_index()
    for model in MODELS:
        part = ledger.loc[ledger.context.eq(CONTEXTS[0]) & ledger.model.eq(model)
                          & ledger.fraction.eq(.05)].set_index('compound').sort_index()
        assert part.index.equals(reference.index)
        for column in ['reference_top', 'reference_loss', 'reference_rank']:
            assert part[column].equals(reference[column]), (model, column)
        assert part.predicted_top.sum() == 11
        hits = part.loc[common.compound].reset_index()
        hits.insert(0, 'measured_hit_rank', range(1, 12))
        assert hits.recovered.equals(hits.predicted_top)
        assert hits.recovered.sum() == scores.set_index('model').loc[model, 'overlap']
        hit_rows.append(hits[['model', 'measured_hit_rank', 'compound', 'reference_loss',
                             'predicted_loss', 'reference_rank', 'predicted_rank',
                             'reference_top', 'predicted_top', 'recovered']])
    membership = pd.concat(hit_rows, ignore_index=True)
    membership.to_csv(ROOT / 'external_hit_membership.csv', index=False)
    marginal_histograms(scatter).to_csv(ROOT / 'external_marginal_histograms.csv', index=False)
    selection_boundaries(scatter).to_csv(ROOT / 'external_selection_boundaries.csv', index=False)


def sources():
    ROOT.mkdir(parents=True, exist_ok=True)
    ledger = read(DATA / 'selection_ledger.csv')
    score = read(DATA / 'utility_scores.csv')
    a = ledger.loc[ledger.context.eq(CONTEXTS[0]) & ledger.model.eq(PRIMARY) & ledger.fraction.eq(.05)].copy()
    b = score.loc[score.context.eq(CONTEXTS[0]) & score.fraction.eq(.05)].set_index('model').loc[MODELS].reset_index()
    c = score.loc[score.context.isin(CONTEXTS[2:]) & score.model.eq(PRIMARY) & score.fraction.eq(.05)].copy()
    d = score.loc[score.context.isin(CONTEXTS) & score.model.eq(PRIMARY)].copy()
    assert len(a) == 217 and a.compound.nunique() == 217
    assert a.reference_top.sum() == a.predicted_top.sum() == 11 and a.recovered.sum() == 10
    assert (a.reference_top | a.predicted_top).sum() == 12
    assert len(b) == 6 and b.overlap.tolist() == [10,10,8,7,0,1]
    assert len(c) == 2 and c.overlap.eq(4).all() and c.measured_repeat_overlap.eq(4).all()
    assert len(d) == 12 and not a[['reference_loss', 'predicted_loss']].isna().any().any()
    prepare_external_display(ledger, a, b)
    rows = []
    for key, part in [('Figure6d',a),('Figure6e',b),('SupplementaryFigure4c',c),('SupplementaryFigure4d',d)]:
        part = part.copy(); part.insert(0, 'panel', key); rows.append(part)
    pd.concat(rows, ignore_index=True).to_csv(ROOT / 'figure_source_data.tsv', sep='\t', index=False)
    return a,b,c,d


def draw_external_scatter(container, rows):
    configure()
    _, fh = dimensions(container)
    side = min(44.5, fh-19.5)
    ax = axmm(container, 12.7, 12, side, side)
    boundaries = selection_boundaries(rows).set_index('axis').display_separator
    measured, predicted = boundaries['measured'], boundaries['predicted']
    ax.add_patch(Rectangle((measured, predicted), LOSS_LIMITS[1]-measured,
                           LOSS_LIMITS[1]-predicted, facecolor='#EDF3F7',
                           edgecolor='none', zorder=0))
    ax.plot(LOSS_LIMITS, LOSS_LIMITS, color='#A5B5C0', lw=.75, zorder=1)
    ax.axvline(measured, color='#738D9F', lw=.85, ls=(0,(3,2.5)), zorder=1)
    ax.axhline(predicted, color='#738D9F', lw=.85, ls=(0,(3,2.5)), zorder=1)
    layers = [
        (~rows.reference_top & ~rows.predicted_top, '#91A8B8', 'o', 13, True),
        (rows.recovered, BLUE, 'o', 34, True),
        (rows.predicted_top & ~rows.reference_top, GOLD, '^', 42, True),
        (rows.reference_top & ~rows.predicted_top, ROSE, 's', 39, False),
    ]
    for mask,colour,marker,size,filled in layers:
        part = rows.loc[mask]
        ax.scatter(part.reference_loss, part.predicted_loss, s=size, marker=marker,
                   facecolors=colour if filled else 'none', edgecolors='white' if filled else colour,
                   linewidths=.32 if filled else 1.05, alpha=.85 if size==13 else 1,
                   zorder=3 if size==13 else 4)
    ax.set(xlim=LOSS_LIMITS, ylim=LOSS_LIMITS, xticks=[0,.3,.6,.9], yticks=[0,.3,.6,.9],
           xlabel='Measured metabolic loss', ylabel='Predicted metabolic loss')
    ax.set_aspect('equal', adjustable='box')
    ax.xaxis.labelpad=3; ax.yaxis.labelpad=3
    # The two marginals contain all 217 compounds, with common loss bins and
    # a common count scale. They are descriptive histograms, not fitted curves.
    hist = marginal_histograms(rows)
    count_limit = hist.compound_count.max() * 1.10
    top = axmm(container, 12.7, 12+side+.8, side, 4.8)
    right = axmm(container, 12.7+side+.8, 12, 4.8, side)
    for axis, marginal in [('measured', top), ('predicted', right)]:
        part = hist.loc[hist.axis.eq(axis)]
        centers = (part.bin_left+part.bin_right)/2
        widths = (part.bin_right-part.bin_left)*.90
        if axis == 'measured':
            marginal.bar(centers, part.compound_count, width=widths,
                         color='#7897AC', linewidth=0)
            marginal.set(xlim=LOSS_LIMITS, ylim=(0,count_limit))
        else:
            marginal.barh(centers, part.compound_count, height=widths,
                          color='#7897AC', linewidth=0)
            marginal.set(ylim=LOSS_LIMITS, xlim=(0,count_limit))
        marginal.set_axis_off()
    x=64.0
    txt(container,x,fh-16,'OASIS · n = 217',fontsize=6.0,ha='left',va='top',color=MUTED)
    txt(container,x,fh-20.5,'10/11 retained',fontsize=6.8,fontweight='semibold',
        ha='left',va='top',color=INK)
    handles=[
        Line2D([],[],marker='o',ls='',ms=4.8,mfc=BLUE,mec='white',mew=.35,label='Both sets'),
        Line2D([],[],marker='^',ls='',ms=4.8,mfc=GOLD,mec='white',mew=.3,label='Predicted only'),
        Line2D([],[],marker='s',ls='',ms=4.6,mfc='none',mec=ROSE,mew=1.0,label='Measured only'),
    ]
    fw,fh=dimensions(container)
    legend=container.legend(handles=handles,loc='upper left',bbox_to_anchor=((x-.8)/fw,(fh-27)/fh),
                     bbox_transform=container.transSubfigure if hasattr(container,'transSubfigure') else container.transFigure,
                     fontsize=6.0,handletextpad=.35,handlelength=.8,labelspacing=.65,borderaxespad=0)
    legend._legend_box.sep=3.8
    for column in legend._legend_handle_box.get_children():
        column.sep=3.6
    return ax


def draw_external_controls(container, rows):
    configure()
    fw,fh=dimensions(container)
    ax=axmm(container,56.4,13,34.6,fh-24)
    matrix=axmm(container,18.5,13,34.4,fh-24)
    membership=pd.read_csv(ROOT/'external_hit_membership.csv',float_precision='round_trip')
    ys=[0,.8,2,2.8,4,4.8]
    ax.set(xlim=(-.08,1.09),ylim=(5.35,-.55),xticks=[0,.5,1],yticks=[])
    ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.spines['left'].set_visible(False)
    ax.axvline(11/217,color=MUTED,lw=.8,ls=(0,(2.5,2.5)),zorder=0)
    matrix.set(xlim=(.4,11.6),ylim=(5.35,-.55),xticks=[1,6,11],
               xticklabels=['1','6','11'],yticks=[])
    matrix.spines[['left','bottom']].set_visible(False)
    matrix.tick_params(axis='x',length=0,pad=3)
    matrix.xaxis.labelpad=3
    indexed=rows.set_index('model')
    for model,y in zip(MODELS,ys):
        r=indexed.loc[model];colour=BLUE if model.startswith('BF') else GOLD if model.startswith('Fluorescence') else GREY
        marker='o' if 'Ridge' in model else 's'
        ax.errorbar(r.recall,y,xerr=[[r.recall-r.recall_lower95],[r.recall_upper95-r.recall]],
                    fmt=marker,color=colour,ecolor=colour,markersize=5.3,elinewidth=1.4,
                    capsize=2.4,capthick=.9,markeredgecolor='white',markeredgewidth=.45,zorder=3)
        hits=membership.loc[membership.model.eq(model)].sort_values('measured_hit_rank')
        assert len(hits)==11 and hits.recovered.sum()==int(r.overlap)
        for hit in hits.itertuples():
            matrix.add_patch(Rectangle((hit.measured_hit_rank-.44,y-.26),.88,.52,
                                       facecolor=colour if hit.recovered else '#F1F4F6',
                                       edgecolor=colour if hit.recovered else '#CBD5DC',
                                       linewidth=.45,zorder=2))
        ypos=13+(fh-24)*(5.35-y)/5.9
        txt(container,15.3,ypos,'Ridge' if 'Ridge' in model else 'HGB',ha='right',va='center',fontsize=6.2)
    # Group labels sit above their two model rows; row labels align to the matrix.
    for label,y in [('Brightfield',0),('Cell count',2),('Acquisition metadata',4)]:
        ypos=13+(fh-24)*(5.35-y)/5.9+3.8
        txt(container,1.8,ypos,label,ha='left',va='center',fontsize=6.3,
            fontweight='semibold',linespacing=1.1)
    for x,label in [(35.7,'Measured top 11'),(73.7,'Recall (95% CI)')]:
        txt(container,x,fh-7,label,ha='center',va='center',fontsize=6.2,fontweight='semibold')
    handles=[Rectangle((0,0),1,1,facecolor=GREY,edgecolor=GREY,lw=.45,label='Recovered'),
             Rectangle((0,0),1,1,facecolor='#F1F4F6',edgecolor='#CBD5DC',lw=.45,label='Missed'),
             Line2D([],[],color=MUTED,lw=.8,ls=(0,(2.5,2.5)),label='Random')]
    container.legend(handles=handles,loc='lower center',bbox_to_anchor=(55/fw,4/fh),
                     bbox_transform=container.transSubfigure if hasattr(container,'transSubfigure') else container.transFigure,
                     fontsize=6,handlelength=1.1,handletextpad=.4,ncol=3,columnspacing=1.1,borderaxespad=0)
    return ax


def draw_recovery_utility(container, module):
    result=module.draw_panel_d(container,add_letter=False)
    ax=result['axis']
    ax.set_position([.16,.18,.80,.72])
    # Reserve physical space for the complete highest marker. Re-express the
    # four background regions in data coordinates so the median split stays
    # exact when the display range gains padding above one.
    median=float(result['summary'].descriptive_utility_split.iloc[0])
    for patch in list(ax.patches):patch.remove()
    ax.set_ylim(0,1.04)
    for x0,x1,y0,y1,colour in [
        (.1,.7,0,median,'#F5F6F7'),(.7,1,0,median,'#FBF2F0'),
        (.1,.7,median,1.04,'#F6F2E9'),(.7,1,median,1.04,'#EEF4F8')]:
        ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,facecolor=colour,edgecolor='none',zorder=0))
    ax.set_xlabel('Ensemble recovery (Pearson r)',labelpad=2)
    ax.set_ylabel('Retained utility',labelpad=2)
    # Keep the six frozen examples, labels and summary lines, with typography
    # positions optimized for the shorter top row only.
    positions={'VAPA':(.50,.95),'VPS35':(.81,.17),'5xUPRE':(.34,.71),
               'Hoechst':(.35,.075),'CellROX':(.28,.88),'NCLN':(.30,.23)}
    for t in ax.texts:
        name=t.get_text()
        if name in positions:t.set_position(positions[name])
        elif name.startswith('cohort median utility'):
            t.set_text(f'cohort median {result["summary"].descriptive_utility_split.iloc[0]:.2f}')
            t.set_position((.105,result['summary'].descriptive_utility_split.iloc[0]+.025))
        if name=='recovery gate 0.70':t.set_text('recovery gate 0.70');t.set_color(INK)
    return result


def draw_utility_atlas(container, module):
    module.configure()
    result=module.draw_panel_a(container,add_letter=False)
    # Keep lower axis labels inside the standalone panel, not in the inter-row
    # gap of the composite. The same physical translation is used in both.
    _,fh=dimensions(container)
    for ax in container.axes:
        p=ax.get_position();ax.set_position([p.x0,p.y0+2.8/fh,p.width,p.height])
    return result


def draw_repeat(container, rows):
    configure();fw,fh=dimensions(container)
    ax=axmm(container,26,15,34,fh-29)
    ax.axvline(0,color=MUTED,lw=.8,ls=(0,(2.5,2.5)))
    indexed=rows.set_index('context')
    for i,context in enumerate(CONTEXTS[2:]):
        r=indexed.loc[context]
        ax.errorbar(r.recall_difference_from_repeat,i,
                    xerr=[[r.recall_difference_from_repeat-r.difference_lower95],
                          [r.difference_upper95-r.recall_difference_from_repeat]],
                    fmt='o',color=BLUE,markersize=4.6,elinewidth=1.05,capsize=2.3,
                    markeredgecolor='white',markeredgewidth=.4)
        ypos=15+(fh-29)*(1.6-i)/2.2
        txt(container,70,ypos,f'{int(r.overlap)}/5',ha='center',va='center',fontsize=6.5)
        txt(container,82,ypos,f'{int(r.measured_repeat_overlap)}/5',ha='center',va='center',fontsize=6.5)
    ax.set(xlim=(-.72,.52),ylim=(1.6,-.6),xticks=[-.6,-.3,0,.3],
           yticks=[0,1],yticklabels=['26 → 27/30','27/30 → 26'],xlabel='Recall difference\n(BF − measured repeat)')
    ax.tick_params(axis='y',length=0,pad=4)
    ax.spines['left'].set_visible(False)
    txt(container,26,fh-6,'94 compounds · top 5%',ha='left',va='center',fontsize=6.3)
    txt(container,70,fh-12,'BF',ha='center',va='center',fontsize=6.1,fontweight='bold')
    txt(container,82,fh-12,'Repeat',ha='center',va='center',fontsize=6.1,fontweight='bold')


def draw_budget(container, rows):
    configure();fw,fh=dimensions(container)
    ax=axmm(container,13,12,69,fh-29)
    labels=['All held-out (217)','Complete pairs (94)','26 → 27/30 (94)','27/30 → 26 (94)']
    for context,label,colour,marker,style in zip(CONTEXTS,labels,[BLUE,GREY,GOLD,TEAL],['o','s','^','D'],
                                                ['-',(0,(3,2)),(0,(1,1)),(0,(5,2,1,2))]):
        r=rows.loc[rows.context.eq(context)].sort_values('fraction')
        ax.plot(r.fraction*100,r.recall,color=colour,marker=marker,ls=style,lw=1.1,
                markersize=4.7 if marker in ('o','s') else 4,markeredgewidth=.85,
                markerfacecolor='white' if marker in ('s','^') else colour,label=label,zorder=3)
    ax.set(xlim=(3.7,21.3),ylim=(0,1.05),xticks=[5,10,20],yticks=[0,.25,.5,.75,1],
           xlabel='Selection budget (%)',ylabel='Hit recall')
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    handles,labels=ax.get_legend_handles_labels()
    container.legend(handles,labels,loc='upper center',bbox_to_anchor=(.55,(fh-4)/fh),
                     bbox_transform=container.transSubfigure if hasattr(container,'transSubfigure') else container.transFigure,
                     ncol=2,fontsize=6,handlelength=2,handletextpad=.4,columnspacing=.7,labelspacing=.5,borderaxespad=0)


def point_text_collisions(ax):
    fig=ax.get_figure()
    while isinstance(fig, mpl.figure.SubFigure):fig=fig.figure
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    points=[]
    for col in ax.collections:
        sizes=col.get_sizes() if hasattr(col,'get_sizes') else []
        if not len(sizes):continue
        coords=col.get_offset_transform().transform(col.get_offsets())
        radius=np.sqrt(max(sizes))*.5*fig.dpi/72+.4*fig.dpi/72
        points.extend((x,y,radius) for x,y in coords)
    problems=[]
    for t in ax.texts:
        if not t.get_visible():continue
        box=Text.get_window_extent(t,renderer)
        for x,y,radius in points:
            dx=max(box.x0-x,0,x-box.x1);dy=max(box.y0-y,0,y-box.y1)
            if dx*dx+dy*dy <= radius*radius:problems.append(t.get_text());break
    return sorted(set(problems))
