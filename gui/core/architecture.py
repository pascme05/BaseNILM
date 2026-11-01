#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         architecture.py
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
View 'Model' of the operating modes: architecture of the neural network the toolkit builds for the current
configuration (read-only). The model is built by the training pipeline of the toolkit on a short slice of the data
(nilm.probe), so input shape, output dimension and layers are those of a real run. Non-TF solvers show their
parameters instead.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import forms, nilm
from gui.core.style import table, fmt, status_badge

import os

import pandas as pd
import streamlit as st

MODES = {0, 1, 2}
KIND = {'Conv1D': ('#e8f1fc', '#2a78d6'), 'Conv2D': ('#e8f1fc', '#2a78d6'), 'MaxPooling1D': ('#f6f6f3', '#898781'),
        'MaxPooling2D': ('#f6f6f3', '#898781'), 'Dense': ('#e7f6ef', '#1baf7a'), 'LSTM': ('#fdf1e8', '#eb6834'),
        'GRU': ('#fdf1e8', '#eb6834'), 'Bidirectional': ('#fdf1e8', '#eb6834'), 'Flatten': ('#ffffff', '#c3c2b7'),
        'Dropout': ('#ffffff', '#c3c2b7'), 'InputLayer': ('#ffffff', '#52514e')}
OTHER = ('#f3effc', '#6250d6')                                        # attention / custom blocks
SOLVER_PARAMS = {'SK': ('Sklearn', 'SK_'), 'PM': ('Pattern matching', 'PM_'), 'SS': ('Source separation', 'SS_')}


#######################################################################################################################
# Helper
#######################################################################################################################
def _shape(s):
    return '–' if s is None else '×'.join('N' if v is None else str(v) for v in s)


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def dot(res, cfg):
    """Graphviz diagram of the layers (functional models with their connections)."""
    lines = ['digraph G {', '  rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.22;',
             '  node [fontname="Helvetica", fontsize=11, shape=box, style="filled,rounded", margin="0.2,0.06"];',
             '  edge [arrowsize=0.6, penwidth=1.1, color="#898781"];']
    W = cfg['Par'].get('window')
    lines.append(f'  "in" [label=<<B>Input</B><BR/><FONT POINT-SIZE="10" COLOR="#52514e">{_shape(res["input"])}'
                 f'  (window {W} × features)</FONT>>, fillcolor="#ffffff", color="#52514e", penwidth=1.6];')
    names = [l['name'] for l in res['layers']]
    prev = 'in'
    for l in res['layers']:
        if l['type'] == 'InputLayer':
            continue
        fill, col = KIND.get(l['type'], OTHER)
        sub = l['info'].replace('kernel_size=', 'k=').replace('filters=', 'f=').replace('units=', 'u=')
        act = f" · {l['activation']}" if l['activation'] and l['activation'] != 'linear' else ''
        txt = (f"<<B>{_esc(l['type'])}</B>  <FONT POINT-SIZE='10' COLOR='#52514e'>{_esc(sub)}{_esc(act)}</FONT>"
               f"<BR/><FONT POINT-SIZE='10' COLOR='#52514e'>{_shape(l['shape'])}  ·  {l['params']:,} params</FONT>>")
        lines.append(f'  "{l["name"]}" [label={txt}, fillcolor="{fill}", color="{col}"];')
        src = [i for i in l['inbound'] if i in names and i != l['name']]
        src = ['in' if next((x for x in res['layers'] if x['name'] == s and x['type'] == 'InputLayer'), None)
               else s for s in src] or [prev]
        for s in dict.fromkeys(src):
            lines.append(f'  "{s}" -> "{l["name"]}";')
        prev = l['name']
    lines.append(f'  "out" [label=<<B>Output</B><BR/><FONT POINT-SIZE="10" COLOR="#52514e">{_shape(res["output"])}'
                 f'  ({_out_text(cfg)})</FONT>>, fillcolor="#e7f6ef", color="#1baf7a", penwidth=1.6];')
    lines.append(f'  "{prev}" -> "out";')
    lines.append('}')
    return '\n'.join(lines)


def _out_text(cfg):
    P, D = cfg['Par'], cfg['Dat']
    n = len(D.get('out') or []) or 'all'
    kind = 'power' if P.get('method') == 0 else 'ON/OFF probability'
    seq = 'seq2point' if not P.get('outseq') else f"{P['outseq']} samples"
    return f"{n} appliance(s), {kind}, {seq}"


#######################################################################################################################
# View
#######################################################################################################################
def view(m):
    cfg = forms.cfg()
    P, E = cfg['Par'], cfg['Exp']
    if P.get('solver') != 'TF':
        _non_tf(cfg)
        return
    if not nilm.exists(cfg):
        st.warning("The training dataset of the configuration was not found — the model is built from the data "
                   "shapes, choose a dataset on the Setup page.")
        return
    D = cfg['Dat']
    name = nilm.first_train(cfg)
    if nilm.is_big(D.get('folder', ''), name, D['type']) and not st.session_state.get(f"big|{name}"):
        st.info(f"{name}.{D['type']} is a large file — load it under *Inputs* first, the model is then built on a "
                f"slice of it.")
        return

    c1, c2 = st.columns([3, 2])
    with c2:
        st.caption("Built by the training pipeline of the toolkit (load → pre-process → framing → features → "
                   "trainMdlTF) on the first 20,000 samples, stopped before fit() — no training, no files written. "
                   "Read-only: the layers are defined in src/model/models.py.")
    try:
        res = nilm.probe(nilm.probe_key(cfg))
    except Exception as e:
        st.error(f"The model could not be built with this configuration:\n\n```\n{str(e)[-2500:]}\n```")
        return

    with c1:
        k = st.columns(4)
        k[0].metric("Model", P.get('model'), help=f"Keras {res['cls']} model built by models.py")
        k[1].metric("Parameters", f"{res['params']:,}", help=f"trainable: {res['trainable']:,}")
        k[2].metric("Input", _shape(res['input']), help="batch × window × features (× channels)")
        k[3].metric("Output", _shape(res['output']), help=_out_text(cfg))
    wf = nilm.weights_file(cfg)
    if os.path.isfile(wf):
        status_badge(f"trained weights mdl/{os.path.basename(wf)} exist (used when Training is off)", 'good')
    elif E.get('train') != 1:
        status_badge(f"no trained weights mdl/{os.path.basename(wf)} — switch on Training", 'warning')

    c1, c2 = st.columns([2, 3])
    with c1:
        st.graphviz_chart(dot(res, cfg), width='content')
        st.caption(":blue[■] convolution · :green[■] dense · :orange[■] recurrent · :violet[■] attention / custom "
                   "block · grey: reshaping / pooling. N = batch dimension.")
    with c2:
        rows = [{'layer': l['name'], 'type': l['type'], 'output shape': _shape(l['shape']), 'parameters': l['params'],
                 'activation': l['activation'] or '', 'settings': l['info']} for l in res['layers']
                if l['type'] != 'InputLayer']
        table(pd.DataFrame(rows), hide_index=True, height=min(560, 38 + 35 * len(rows)))
        M = cfg['Mdl']
        st.markdown("**Training setup** (setupMdl)")
        st.markdown(f"Loss **{M.get('loss')}** · metric **{M.get('metric')}** · optimiser **{M.get('opt')}** "
                    f"(lr {fmt(M.get('lr'))}) · batch {M.get('batch')} · ≤ {M.get('epoch')} epochs · early stopping "
                    f"after {M.get('patience')} epochs · output activation "
                    f"**{'linear' if P.get('method') == 0 else 'sigmoid'}**")
        if res.get('batch'):
            st.caption("Training batch from the pipeline: X " + _shape(res['batch'][0]) + ", y " +
                       _shape(res['batch'][1]))
        with st.expander("Pipeline log"):
            st.code(res['log'][-6000:], language=None)


def _non_tf(cfg):
    P = cfg['Par']
    sol = P.get('solver')
    if sol == 'PT':
        st.info("PyTorch models (trainMdlPT.py) are built inside the training loop; the architecture view supports "
                "the TensorFlow models. Switch the solver to TF to see the corresponding Keras model.")
        return
    if sol == 'CU':
        st.info("Custom solver: the model is defined in src/model/trainMdlCU.py.")
        return
    title, prefix = SOLVER_PARAMS.get(sol, (sol, ''))
    st.markdown(f"#### {title} · {P.get('model')}")
    st.caption("No neural network: the model is defined by these parameters (setupMdl).")
    rows = [{'parameter': p['name'], 'variable': p['var'], 'value': str(forms.value('Mdl', p['cat'], p['var']))}
            for p in forms.schema()['Mdl'] if not p['nested'] and p['var'].startswith(prefix)
            and (f"_{P.get('model')}_" in p['var'] or '_Gen_' in p['var'])]
    if rows:
        table(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("No parameters for this model in mdlPara.py.")
