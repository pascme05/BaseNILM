#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         workflow.py
# Date:         01.11.2026
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function draws the workflow of every operating mode (and sub-mode) as a flow chart (Graphviz, rendered in the
browser) and explains the steps. The nodes show the settings of the current configuration that control the step;
steps that are skipped with the current settings are drawn dashed. Add one function per mode to WORKFLOWS.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms

import streamlit as st

#######################################################################################################################
# Style
#######################################################################################################################
COL = {'inp': ('#e8f1fc', '#2a78d6'), 'step': ('#ffffff', '#52514e'), 'out': ('#e7f6ef', '#1baf7a'),
       'opt': ('#fdf1e8', '#eb6834'), 'off': ('#f6f6f3', '#c3c2b7')}


def opt(sec, cat, var):
    """'value) label' of a parameter with options."""
    return f"{forms.value(sec, cat, var)}) {forms.option_label(sec, cat, var).split(' (')[0]}"


def val(sec, cat, var, unit=''):
    v = forms.value(sec, cat, var)
    if isinstance(v, float) and forms.find(sec, cat, var)['percent']:
        return f"{100 * v:g} %"
    return f"{v:g}{(' ' + unit) if unit else ''}" if isinstance(v, (int, float)) else str(v)


class Flow:
    """Small builder of a Graphviz flow chart."""

    def __init__(self, rankdir='TB'):
        self.nodes, self.edges, self.ranks = [], [], []
        self.rankdir = rankdir

    def node(self, nid, title, lines=(), kind='step', active=True):
        fill, line = COL[kind if active else 'off']
        txt = f"<<B>{_esc(title)}</B>" + ''.join(f"<BR/><FONT POINT-SIZE='10' COLOR='#52514e'>{_esc(l)}</FONT>"
                                                  for l in lines if l) + ">"
        shape = {'inp': 'note', 'out': 'box', 'opt': 'box', 'step': 'box'}[kind]
        style = 'filled,rounded' + (',dashed' if not active else '') + (',bold' if kind == 'out' else '')
        self.nodes.append(f'"{nid}" [label={txt}, shape={shape}, style="{style}", fillcolor="{fill}", '
                          f'color="{line}", fontcolor="{"#0b0b0b" if active else "#898781"}"];')
        return self

    def edge(self, a, b, label='', active=True):
        lab = f', label="{label}", fontsize=9, fontcolor="#52514e"' if label else ''
        self.edges.append(f'"{a}" -> "{b}" [color="{"#52514e" if active else "#c3c2b7"}"'
                          f'{", style=dashed" if not active else ""}{lab}];')
        return self

    def same(self, *ids):
        self.ranks.append('{rank=same; ' + ' '.join(f'"{i}";' for i in ids) + '}')
        return self

    def dot(self):
        return ('digraph G {\n'
                f'  rankdir={self.rankdir}; bgcolor="transparent"; nodesep=0.35; ranksep=0.4;\n'
                '  node [fontname="Helvetica", fontsize=12, margin="0.18,0.08"];\n'
                '  edge [arrowsize=0.7, penwidth=1.2];\n  ' + '\n  '.join(self.nodes + self.edges + self.ranks) +
                '\n}')


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _legend():
    st.caption(":blue[■] inputs (dataset, trained model) · ■ processing step · :orange[■] option / loop · "
               ":green[■] results · dashed: skipped with the current settings")


#######################################################################################################################
# Workflows: mode -> function(cfg, sub) returning (Flow, [(step title, explanation), ...])
#######################################################################################################################
def _example(cfg, sub):
    f = Flow()
    f.node('cfg', 'Configuration', [A.NAME], kind='inp')
    f.node('sim', 'Simulation', ['main(setup, path)'])
    f.node('res', 'Results', ['figures, data'], kind='out')
    f.edge('cfg', 'sim').edge('sim', 'res')
    return f, [('Configuration', "parameters of the GUI"), ('Simulation', "toolkit"), ('Results', "outputs")]


_NORM = {0: 'none', 1: '−1/+1', 2: '0/1', 3: 'mean/std', 4: 'Q1/Q3', 5: 'quantile'}
_FEAT = {0: 'raw windows', 1: 'statistical (frame)', 2: 'rolling (input)', 3: 'rolling + statistical', 4: '2D images'}


def _pipeline(f, c, start, train_on=True):
    """Pre-processing → framing → features → sequence nodes (shared by training, testing and the optimisation)."""
    D, P = c['Dat'], c['Par']
    filt = [x for x, on in (('median X', D.get('inpFil')), ('median y', D.get('outFil'))) if on]
    noise = [x for x, on in (('noise X', D.get('inpNoise')), ('noise y', D.get('outNoise'))) if on]
    f.node('pre', 'Pre-processing', [f"resample to fs = {D.get('fs'):.4g} Hz",
                                     f"norm X {_NORM.get(D.get('inpNorm'))} · y {_NORM.get(D.get('outNorm'))}",
                                     ', '.join(filt + noise) or None,
                                     'ghost power' if D.get('ghost') else None], active=train_on)
    fr = P.get('frame') == 1 and D.get('freq') == 'LF'
    f.node('frm', 'Framing', [f"window {P.get('window')} · overlap {P.get('overlap')}"] if fr else
           ['HF windows (adaptDimHF)' if D.get('freq') == 'HF' else 'off: one sample per step'],
           active=fr or D.get('freq') == 'HF')
    f.node('fea', 'Features', [_FEAT.get(P.get('feat'), P.get('feat'))], active=P.get('feat', 0) != 0)
    seq = 'seq2point' + (f" (sample {P.get('yFocus')})" if P.get('yFocus', 0) >= 0 else ' (mean)') \
        if not P.get('outseq') else f"seq2subseq {P['outseq']}" if P['outseq'] < P.get('window', 0) else 'seq2seq'
    f.node('seq', 'Target sequence', [seq, f"input dim {P.get('nDim')}D"], active=P.get('frame') == 1)
    f.edge(start, 'pre', active=train_on).edge('pre', 'frm').edge('frm', 'fea').edge('fea', 'seq')
    return f


def _sim(c, sub):
    E, D, P, M = c['Exp'], c['Dat'], c['Par'], c['Mdl']
    m = E.get('method', 0) if sub is None else sub
    tr, te = E.get('train') == 1, E.get('test') == 1
    folder = D.get('folder') or ''
    f = Flow()
    f.node('data', 'Dataset', [f"data/{folder + '/' if folder else ''}{', '.join(map(str, D.get('train') or []))}."
                               f"{D.get('type')}", f"{D.get('freq')} · {D.get('dim')}D"], kind='inp')
    apps = f"{len(D.get('out') or [])} appliances" if not D.get('outEnergy') else f"top {100 * D['outEnergy']:g} % energy"
    f.node('load', 'Load & select', [f"inputs: {', '.join(map(str, D.get('inp') or [])) or 'all'}", apps,
                                     f"limit {D.get('lim'):,} samples" if D.get('lim') else 'no limit'])
    split = {0: f"1-fold: train {D.get('rT', 0):.2f} · test {1 - D.get('rT', 0):.2f}",
             1: f"k-fold: {E.get('kfold')} folds", 2: f"transfer: test {D.get('test')}, val {D.get('val')}",
             3: f"id: test {D.get('idT')} · val {D.get('idV')}"}[m]
    f.node('split', 'Split', [split], kind='opt' if m == 1 else 'step')
    f.edge('data', 'load').edge('load', 'split')
    _pipeline(f, c, 'split')
    f.node('train', 'Training', [f"{P.get('solver')} · {P.get('model')}", f"≤ {M.get('epoch')} epochs · "
                                 f"{M.get('opt')} lr {M.get('lr'):g}" if P.get('solver') in ('TF', 'PT') else None,
                                 f"→ mdl/mdl_{P.get('model')}_{E.get('name')}"], active=tr)
    f.node('model', 'Trained model', [f"mdl/mdl_{P.get('model')}_{E.get('name')}"], kind='inp', active=te and not tr)
    f.node('test', 'Testing', ['prediction on the test data'], active=te)
    f.node('post', 'Post-processing', ['de-normalisation', f"clip [{P.get('outMin')}, {P.get('outMax')}]",
                                       f"ON threshold {D.get('threshold')}"], active=te)
    f.node('res', 'Results', ['ACC · F1 · TECA · RMSE · MAE', 'feature ranking' if P.get('ranking') else None,
                              'figures' if E.get('plot') else None, 'saved to results/' if E.get('save') else None],
           kind='out', active=te)
    f.edge('seq', 'train', active=tr).edge('train', 'test', active=tr and te).edge('model', 'test', active=te and not tr)
    f.edge('seq', 'test', active=te).edge('test', 'post', active=te).edge('post', 'res', active=te)
    if m == 1:
        f.edge('res', 'split', label=f"next fold (×{E.get('kfold')})")
    steps = [('Dataset', "the file is read with the loader of its format (loadData.py)."),
             ('Load & select', "input features and appliances are selected, the samples limited, constant columns "
                               "dropped, NaNs interpolated."),
             ('Split', {0: "the first rT of the samples train the model, the rest tests it (validation: first rV of "
                           "the training part).", 1: "KFold: every fold is tested once by a model trained on the "
                           "others; scores are averaged.", 2: "whole datasets for training, validation and testing.",
                        3: "the column 'id' decides training, validation and test data."}[m]),
             ('Pre-processing', "resampling (zero-order hold), filtering, noise and normalisation (preprocess.py)."),
             ('Framing', "sliding windows of the time series (framing.py)."),
             ('Features', "optional statistical, rolling or 2D features of the windows."),
             ('Target sequence', "the target value(s) of each window (createSeq.py) and the input dimension "
                                 "(adaptDim.py)."),
             ('Training', "the model is built (models.py) and trained with early stopping; skipped if Training is "
                          "off, then the stored model is loaded."),
             ('Testing / Post-processing', "prediction, de-normalisation and clipping; the ON threshold defines the "
                                           "states for accuracy and F1."),
             ('Results', "performance.py computes the scores, shown under Results.")]
    return f, steps


def _hyper(c, sub):
    D, P = c['Dat'], c['Par']
    f = Flow()
    f.node('data', 'Dataset', [', '.join(map(str, D.get('train') or []))], kind='inp')
    f.node('load', 'Load', ['training data (train = 1)', 'validation data (train = 2)'])
    f.edge('data', 'load')
    _pipeline(f, c, 'load')
    tuner = {'TF': 'Keras-Tuner (trainMdlTFopti)', 'SK': 'GridSearchCV (trainMdlSKopti)'}.get(P.get('solver'),
                                                                                             'not supported')
    f.node('opt', 'Optimisation', [tuner, P.get('model')], kind='opt', active=P.get('solver') in ('TF', 'SK'))
    f.node('res', 'Best hyperparameters', ['console output'], kind='out')
    f.edge('seq', 'opt').edge('opt', 'res')
    return f, [('Load', "training and validation data of the first training dataset (transfer split)."),
               ('Pipeline', "same pre-processing, framing and features as the simulation."),
               ('Optimisation', "search over the hyperparameter space defined in models.py (tfMdlopti*) or "
                                "trainMdlSKopti.py."),
               ('Result', "best parameters / epoch are printed; copy them into the Model parameters.")]


def _grid(c, sub):
    f = Flow()
    f.node('cfg', 'Configuration', ['all parameters of the GUI'], kind='inp')
    f.node('grid', 'Grid loop', ['window ∈ {4, 8, 16, 32, 64}', 'fs ∈ {1/60 … 1/3600} Hz', 'overlap = window − 1,'
                                 ' yFocus = window/2'], kind='opt')
    f.node('main', 'Simulation', ['main(): train / test as configured'])
    f.node('save', 'Saved results', ['results/<name>/result_Grid_*.csv'], kind='out')
    f.node('res', 'Heatmap', ['average score per grid point'], kind='out')
    f.edge('cfg', 'grid').edge('grid', 'main').edge('main', 'save').edge('main', 'res')
    f.edge('main', 'grid', label='next grid point (30)')
    return f, [('Grid loop', "optiGrid.py overwrites window, overlap, yFocus and fs for every grid point."),
               ('Simulation', "a complete run of main() per point (name <name>_Grid<i><j>, saving on)."),
               ('Results', "the GUI collects the average scores of all points into a heatmap.")]


WORKFLOWS = {0: _sim, 1: _hyper, 2: _grid}


#######################################################################################################################
# View
#######################################################################################################################
def view(m):
    cfg = forms.cfg()
    sub = None
    spec = A.MODES[m].get('sub')
    if spec:
        p = forms.find(*spec)
        cur = forms.value(*spec)
        if p['options']:
            names = p['options']
            sub = st.segmented_control("Sub-mode", list(names), default=cur, key=f"wf|{m}",
                                       format_func=lambda k: f"{names[k]}" + (' (selected)' if k == cur else ''))
            sub = cur if sub is None else sub
            if sub != cur:
                st.caption(f"Showing sub-mode {sub}; the configuration runs sub-mode {cur} — change it in "
                           f"*Parameters*.")
    f, steps = WORKFLOWS[m](cfg, sub)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.graphviz_chart(f.dot(), width='content')
        _legend()
    with c2:
        st.markdown("##### What happens")
        for i, (t, txt) in enumerate(steps, start=1):
            st.markdown(f"**{i}. {t}** — {txt}")
        st.caption("The boxes show the settings of the current configuration; change them in *Parameters*, the "
                   "diagram follows.")
