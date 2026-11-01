#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         theory.py
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
Theory of the operating modes with interactive examples (view 'Theory' of a mode page, shown when the mode has an
entry in THEORY). Each function gets the current config and renders equations (st.latex), a short explanation and
explorers: widgets with keys 'th<mode>|…' (not bound to the config), key-figure tiles and charts via style.show.
Use the functions of the toolkit (pre-processing, models, loaders) instead of re-implementing them, cache expensive
calls (st.cache_resource / st.cache_data on a JSON dump of the config) and state units in every label and table.
Reference: PyPowerSim gui/core/theory.py (+ gui/core/toolkit.py), OPTELIA gui/core/theory.py.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import streamlit as st


def _intro(text):
    st.caption(text + " Values start from the current configuration; changes here do not modify it.")


#######################################################################################################################
# Theory per mode: mode -> function(cfg)
#######################################################################################################################
def _metrics_explorer(cfg):
    """Synthetic appliance + disturbed prediction scored with performance() of the toolkit."""
    import contextlib
    import io

    import numpy as np
    import plotly.graph_objects as go
    from gui.core.style import show, line, fmt, MUTED
    from src.general.performance import performance

    thr = float(cfg['Dat'].get('threshold') or 50)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        miss = st.slider("Missed activations [%]", 0, 100, 10, key='th0|miss')
    with c2:
        false_on = st.slider("False ON events", 0, 20, 2, key='th0|false')
    with c3:
        amp = st.slider("Amplitude error [%]", -50, 50, 10, key='th0|amp')
    with c4:
        noise = st.slider("Noise [W]", 0, 200, 20, key='th0|noise')

    rng = np.random.default_rng(1)
    n, p_on = 1440 * 3, 2000.0
    y = np.zeros(n)
    starts = np.arange(60, n - 120, 240)
    for s in starts:
        y[s:s + 60] = p_on
    yp = y * (1 + amp / 100)
    for s in rng.choice(starts, int(round(len(starts) * miss / 100)), replace=False):
        yp[s:s + 60] = 0
    for s in rng.choice(np.arange(0, n - 40), false_on, replace=False):
        yp[s:s + 30] = p_on
    yp = np.clip(yp + rng.normal(0, noise, n), 0, None)
    lab_t, lab_p = (y > thr).astype(int)[:, None], (yp > thr).astype(int)[:, None]
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = performance(yp[:, None], y[:, None], lab_p, lab_t, {'numOut': 1})
    k = st.columns(5)
    for col, (name, v, u) in zip(k, [('Accuracy', 100 * res[0, 0], '%'), ('F1 (weighted)', 100 * res[0, 1], '%'),
                                     ('TECA', 100 * res[0, 3], '%'), ('RMSE', res[0, 4], 'W'),
                                     ('MAE', res[0, 5], 'W')]):
        col.metric(name, fmt(v, u))
    t = np.arange(n) / 60
    fig = go.Figure([line(t, y, 'true', c=MUTED, width=1.2), line(t, yp, 'predicted', 0, width=1.2)])
    fig.update_layout(xaxis_title="Time [h]", yaxis_title="Power [W]", hovermode='x unified')
    show(fig, height=300, key='th0|fig', info="Synthetic appliance (2 kW, one hour every four hours, 1 min "
         "resolution) and a prediction with the chosen errors. The scores are computed by performance() of the "
         f"toolkit with the ON threshold of the configuration ({fmt(thr)} W).")


def _framing_explorer(cfg):
    """Toy aggregate: how window, overlap and the output type turn a series into model samples (framing.py,
    createSeq.py of the toolkit)."""
    import contextlib
    import io

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from gui.core.style import show, line, SERIES, MUTED
    from src.general.framing import framing
    from src.general.createSeq import createSeq

    P = cfg['Par']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        W = st.slider("Window W [samples]", 4, 40, int(min(max(P.get('window', 20), 4), 40)), key='th0|W')
    with c2:
        step = st.slider("Step W − overlap", 1, W, 1 if P.get('overlap', W - 1) >= W - 1 else
                         max(1, min(W, W - int(P.get('overlap', 0)))), key='th0|step')
    with c3:
        kind = st.selectbox("Output", ['seq2point', 'seq2subseq', 'seq2seq'],
                            index=0 if not P.get('outseq') else 1 if P.get('outseq') < P.get('window', 20) else 2,
                            key='th0|kind')
    with c4:
        if kind == 'seq2point':
            foc = st.slider("Focus point", 0, W - 1, min(W // 2, W - 1), key='th0|foc')
            L = 0
        elif kind == 'seq2subseq':
            L = st.slider("Sub-sequence length", 1, W, max(1, W // 2), key='th0|L')
            foc = W // 2
        else:
            L, foc = W, W // 2

    rng = np.random.default_rng(3)
    n = 120
    t = np.arange(n)
    fridge = np.where((t % 30) < 12, 120.0, 0.0)
    kettle = np.zeros(n)
    kettle[50:56] = 2000.0
    agg = fridge + kettle + 60 + rng.normal(0, 10, n)
    X = pd.DataFrame({'P-agg': agg})
    y = pd.DataFrame({'Fridge': fridge, 'Kettle': kettle})
    with contextlib.redirect_stdout(io.StringIO()):
        fx = framing(X, W, W - step)
        fy = framing(y, W, W - step)
        tgt = createSeq(fy, {'window': W, 'outseq': L, 'yFocus': foc})
    nf = len(fx)
    show_n = min(nf, 12)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.04)
    fig.add_trace(line(t, agg, 'Aggregate P-agg', 0, width=1.4), row=1, col=1)
    fig.add_trace(line(t, kettle, 'Kettle (target)', 1, width=1.1), row=1, col=1)
    delta = int((W - L) / 2) if L else 0
    for k in range(show_n):
        s0 = k * step
        fig.add_trace(go.Scatter(x=[s0, s0 + W - 1], y=[k, k], mode='lines', line=dict(color=SERIES[2], width=7),
                                 opacity=0.55, showlegend=k == 0, name='input window', hoverinfo='skip'), row=2, col=1)
        if kind == 'seq2point':
            fig.add_trace(go.Scatter(x=[s0 + foc], y=[k], mode='markers', marker=dict(color=SERIES[7], size=9,
                                     symbol='diamond'), showlegend=k == 0, name='target sample', hoverinfo='skip'),
                          row=2, col=1)
        else:
            fig.add_trace(go.Scatter(x=[s0 + delta, s0 + W - 1 - delta], y=[k, k], mode='lines',
                                     line=dict(color=SERIES[7], width=3), showlegend=k == 0, name='target sequence',
                                     hoverinfo='skip'), row=2, col=1)
    fig.update_yaxes(title_text="Power [W]", row=1, col=1)
    fig.update_yaxes(title_text="Frame", autorange='reversed', row=2, col=1)
    fig.update_xaxes(title_text="Sample", row=2, col=1)
    fig.update_layout(title=f"{nf} frames of {W} samples, step {step} (first {show_n} shown)")
    show(fig, height=520, key='th0|frames', info="Top: a toy aggregate (fridge cycles plus one kettle event). "
         "Bottom: every bar is one input sample of the model (framing.py with overlap = W − step); the red marks are "
         "the targets createSeq.py takes from the appliance power in the same window. Step 1 (overlap W − 1) gives "
         "one frame per time step — the usual seq2point setting of BaseNILM.")
    shape_t = tuple(np.shape(tgt))
    st.markdown(f"Input array **{tuple(np.shape(fx))}** (frames × W × features) → target array **{shape_t}**"
                f" (frames{' × length' if kind != 'seq2point' else ''} × appliances). "
                + ("For testing, seq2subseq/seq2seq use the step = sub-sequence length so that the predicted "
                   "pieces line up without overlap (test.py)." if kind != 'seq2point' else
                   "Each frame predicts one sample; with step 1 the predictions form a continuous series again."))


def _simulation(cfg):
    _intro("Energy disaggregation estimates the power of single appliances from the aggregate measurement.")
    t1, t2, t3, t4 = st.tabs([":material/function: Problem", ":material/view_week: Framing",
                              ":material/tune: Pre-processing", ":material/rule: Scores"])
    with t1:
        st.latex(r"P_{agg}(t) = \sum_{i=1}^{N} p_i(t) + e(t)")
        st.markdown("The aggregate $P_{agg}$ is the sum of the appliance powers $p_i$ and of loads that are not "
                    "modelled ($e$, ghost power). The model learns a mapping from a window of the aggregate to the "
                    "appliance powers:")
        st.latex(r"\hat{\mathbf{p}}(t_k) = f_\theta\left(P_{agg}(t_{k-W/2}), \dots, P_{agg}(t_{k+W/2})\right)")
        W, fs = cfg['Par'].get('window', 20), cfg['Dat'].get('fs') or 1
        st.caption(f"Current configuration: W = {W} samples at fs = {fs:.4g} Hz → one window covers "
                   f"{W / fs / 60:.4g} min of the aggregate.")
        st.markdown("**Pipeline of one run** (main.py → train.py / test.py)")
        st.markdown(
            "1. `loadData` — read the dataset, select inputs/appliances, limit, split (1-fold, k-fold, transfer, "
            "ids), normalisation statistics of the **training** data\n"
            "2. `preprocess` — resampling to fs, ghost power, filters, noise, normalisation (`normData`)\n"
            "3. `framing` — windows of W samples (step W − overlap)\n"
            "4. `features1D/2D` — optional statistical or 2D features per window\n"
            "5. `createSeq` — target per window (seq2point / seq2subseq / seq2seq)\n"
            "6. `adaptDim` — array shape for the model (2D / 3D)\n"
            "7. training (`trainMdlTF`, …) or prediction (`testMdlTF`, …)\n"
            "8. `postprocess` — inverse normalisation, clipping to [outMin, outMax], ON/OFF states\n"
            "9. `performance` — scores per appliance and average")
    with t2:
        _framing_explorer(cfg)
    with t3:
        D = cfg['Dat']
        st.markdown("**Normalisation** (`normData.py`, statistics from the training data, inverted after the "
                    "prediction). Options of inputs (inpNorm) and outputs (outNorm):")
        st.markdown("| Option | Transformation | Use |\n|---|---|---|\n"
                    "| 0 | none | data already scaled |\n"
                    "| 1 | $(x - \\mu)/(x_{max} - x_{min})$ | centred, range about ±1 |\n"
                    "| 2 | $(x - x_{min})/(x_{max} - x_{min})$ | 0 … 1, keeps 0 W at 0 |\n"
                    "| 3 | $(x - \\mu)/\\sigma$ | centred, unit standard deviation |\n"
                    "| 4 | 0/1 scaling, then $(x - Q_1)/(Q_3 - Q_1)$ | robust against peaks |\n"
                    "| 5 | 0/1 scaling, then quantile transformation to a normal distribution | skewed data |")
        st.caption(f"Current configuration: inputs {D.get('inpNorm')}, outputs {D.get('outNorm')}. The *Inputs* view "
                   f"(Pre-processing & framing) shows the frames before and after the normalisation.")
        st.markdown("**Resampling** (`resampleData.py`): downsampling averages all samples of each new interval "
                    "(energy is kept, short peaks are flattened), upsampling holds the last value (zero-order hold). "
                    "**Filters** (`filterData.py`): median or mean filter of the given length. **Ghost power** "
                    "(`addGhost.py`): the remainder aggregate − Σ appliances is 1) an extra output 'Ghost' or "
                    "2) subtracted from the aggregate (ideal data, X = Σ y).")
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"\mathrm{TECA}_i = 1 - \frac{\sum_t |\hat{p}_i(t) - p_i(t)|}{2 \sum_t |p_i(t)|}")
            st.latex(r"\mathrm{TECA}_{avg} = 1 - \frac{\sum_i \sum_t |\hat{p}_i(t) - p_i(t)|}"
                     r"{2 \sum_i \sum_t |p_i(t)|}")
        with c2:
            st.latex(r"\mathrm{ACC}_i = 1 - \frac{1}{T}\sum_t |\hat{s}_i(t) - s_i(t)|, \quad s_i = [p_i > p_{thr}]")
            st.latex(r"\mathrm{EST}_i = \frac{\sum_t |\hat{p}_i|}{\sum_i \sum_t |p_i|}, \quad "
                     r"\mathrm{TRUTH}_i = \frac{\sum_t |p_i|}{\sum_i \sum_t |p_i|}")
        st.caption("The average TECA weights the appliances by their energy (one sum over all appliances); accuracy, "
                   "F1 (sklearn, weighted over ON and OFF), RMSE and MAE are plain means over the appliances. F1 "
                   "uses the ON/OFF states of the threshold.")
        st.markdown("#### Explorer: how errors change the scores")
        _metrics_explorer(cfg)


THEORY = {0: _simulation}
