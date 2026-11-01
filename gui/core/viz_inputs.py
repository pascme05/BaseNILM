#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         viz_inputs.py
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
Data explorer of the BaseNILM GUI: the dataset is read with the loaders of the toolkit (nilm.load) and shown before any
training — overview, aggregate and appliance time series, energy shares and ON times, the train / validation / test
split of loadData() and a preview of resampling, normalisation and framing with the toolkit functions.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from gui.core import nilm
from gui.core.style import show, line, table, fmt, color, SERIES, INK2, MUTED

# ==============================================================================
# External
# ==============================================================================
import contextlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MAX_POINTS = 6000
ROLE_COL = {'Training': SERIES[0], 'Validation': SERIES[3], 'Testing': SERIES[1], 'Fold': SERIES[2]}


#######################################################################################################################
# Generic viewer (kept from the scaffold for tables)
#######################################################################################################################
def arrow_safe(df):
    """Mixed text/number columns as text, so the table renders without warnings."""
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and not all(isinstance(v, str) for v in df[c].dropna()):
            df[c] = df[c].map(lambda v: '' if v is None or (isinstance(v, float) and v != v) else str(v))
    return df


#######################################################################################################################
# Helper
#######################################################################################################################
def _time_axis(t):
    """x values and title: dates for unix time, else days since the start."""
    t = np.asarray(t, dtype=float)
    if t.size and np.nanmin(t) > 9e8:
        return pd.to_datetime(t, unit='s'), 'Time'
    return (t - (t[0] if t.size else 0)) / 86400, 'Time [days]'


def _window(n, key, ts):
    """Start / length selection of the shown samples (keeps the charts responsive)."""
    days = n * ts / 86400 if np.isfinite(ts) and ts > 0 else None
    c1, c2 = st.columns([3, 1])
    if days and days > 2:
        with c2:
            span = st.selectbox("Span", [1, 3, 7, 14, 30, 90, 365, 'all'], index=2, key=f"{key}|span",
                                format_func=lambda d: 'all' if d == 'all' else f"{d} days")
        span_n = n if span == 'all' else min(n, int(span * 86400 / ts))
        with c1:
            start = st.slider("Start [day]", 0.0, max(days - span_n * ts / 86400, 0.0), 0.0, key=f"{key}|start")
        i0 = min(int(start * 86400 / ts), max(n - span_n, 0))
    else:
        with c2:
            span_n = st.number_input("Samples", 100, n, min(n, 20000), step=1000, key=f"{key}|span")
        with c1:
            i0 = st.slider("Start [sample]", 0, max(n - span_n, 0), 0, key=f"{key}|start")
    return i0, i0 + int(span_n)


def _thin(i0, i1):
    return max(1, (i1 - i0) // MAX_POINTS)


def _units(d, side, cols):
    return {c: nilm.unit_of(d['units'], 'Input' if side == 'X' else 'Output', c) for c in cols}


def _limited(cfg, d):
    """X, y after the sample limit of the configuration (as loadData)."""
    lim = int(cfg['Dat'].get('lim') or 0)
    if lim:
        return d['X'].iloc[:lim], d['y'].iloc[:lim]
    return d['X'], d['y']


#######################################################################################################################
# Sections
#######################################################################################################################
def overview(cfg, d, name):
    X, y = d['X'], d['y']
    ts = nilm.sample_time(X)
    xin, yout = nilm.channels(X), nilm.channels(y)
    sel = nilm.selected_outputs(cfg, y)
    lim = int(cfg['Dat'].get('lim') or 0)
    n_used = min(len(X), lim) if lim else len(X)
    ids = sorted(pd.unique(X['id'])) if 'id' in X else []
    c = st.columns(6)
    c[0].metric("Samples", f"{len(X):,}", help="Rows of the dataset file.")
    c[1].metric("Used", f"{n_used:,}", help="After the sample limit 'lim' (before resampling).")
    c[2].metric("Duration", fmt(len(X) * ts / 86400, 'days') if np.isfinite(ts) else '–')
    c[3].metric("Sampling time", fmt(ts, 's'), help=f"time[1] − time[0] of the file; the model uses "
                f"fs = {fmt(cfg['Dat'].get('fs'))} Hz → {fmt(1 / cfg['Dat']['fs'] if cfg['Dat'].get('fs') else 0, 's')}")
    c[4].metric("Inputs", len(xin), help=", ".join(map(str, xin)))
    c[5].metric("Appliances", f"{len(sel)} / {len(yout)}", help="selected / in the file")

    if len(ids) > 1:
        txt = ', '.join(str(int(i)) if float(i).is_integer() else str(i) for i in ids[:30])
        st.caption(f"Ids (column 'id', e.g. houses): {txt}")
    if d.get('HF_W'):
        st.caption(f"High-frequency data: {d['HF_F']} features × {d['HF_W']} samples per time step.")

    # Channel table
    un_y = _units(d, 'y', yout)
    energy = y[yout].clip(lower=0).sum(axis=0)
    thr = float(cfg['Dat'].get('threshold') or 0)
    rows = [{'channel': c_, 'side': 'input', 'unit': nilm.unit_of(d['units'], 'Input', c_), 'mean': X[c_].mean(),
             'max': X[c_].max(), 'energy share [%]': np.nan, 'ON time [%]': np.nan, 'selected': c_ in
             (cfg['Dat'].get('inp') or xin)} for c_ in xin]
    rows += [{'channel': c_, 'side': 'appliance', 'unit': un_y[c_], 'mean': y[c_].mean(), 'max': y[c_].max(),
              'energy share [%]': 100 * energy[c_] / max(energy.sum(), 1e-12),
              'ON time [%]': 100 * float((y[c_] > thr).mean()) if thr else np.nan, 'selected': c_ in sel}
             for c_ in yout]
    st.markdown("**Channels**")
    table(pd.DataFrame(rows), hide_index=True, height=min(460, 38 + 35 * len(rows)),
          column_config={'selected': st.column_config.CheckboxColumn(help="Used by the configuration (inp / out).")})
    with st.expander("Loader output"):
        st.code(d.get('log') or '(no output)', language=None)


def timeseries(cfg, d, key):
    X, y = d['X'], d['y']
    ts = nilm.sample_time(X)
    xin, yout = nilm.channels(X), nilm.channels(y)
    sel = nilm.selected_outputs(cfg, y)
    c1, c2 = st.columns([1, 2])
    with c1:
        inp = st.selectbox("Aggregate input", xin, key=f"{key}|inp")
    with c2:
        apps = st.multiselect("Appliances", yout, default=sel[:8], key=f"{key}|apps")
    i0, i1 = _window(len(X), key, ts)
    s = _thin(i0, i1)
    xs, xt = _time_axis(X['time'].iloc[i0:i1:s])
    stack = st.toggle("Stack appliances", value=True, key=f"{key}|stack",
                      help="Stacked areas show how the appliances add up to the aggregate (the gap is the ghost / "
                           "unmetered power).")
    fig = go.Figure()
    for k, a in enumerate(apps):
        fig.add_trace(go.Scatter(x=xs, y=y[a].iloc[i0:i1:s], name=str(a), mode='lines', line=dict(width=0.8,
                                 color=color(k)), stackgroup='a' if stack else None))
    fig.add_trace(line(xs, X[inp].iloc[i0:i1:s], f"{inp} (aggregate)", c=INK2, width=1.2))
    u = nilm.unit_of(d['units'], 'Input', inp)
    fig.update_layout(xaxis_title=xt, yaxis_title=f"{inp}" + (f" [{u}]" if u else ''), hovermode='x unified')
    show(fig, height=460, key=f"{key}|ts")
    if s > 1:
        st.caption(f"Every {s}th sample shown ({(i1 - i0):,} samples in the window).")


def appliances(cfg, d, key):
    X, y = _limited(cfg, d)
    yout = nilm.channels(y)
    sel = nilm.selected_outputs(cfg, y)
    thr = float(cfg['Dat'].get('threshold') or 0)
    energy = y[yout].clip(lower=0).sum(axis=0).sort_values(ascending=False)
    share = 100 * energy / max(energy.sum(), 1e-12)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=share.index.astype(str), y=share.values,
                               marker_color=[SERIES[0] if a in sel else '#c3c2b7' for a in share.index],
                               hovertemplate='%{x}: %{y:.2f} %<extra></extra>'))
        fig.update_layout(title="Energy share of the appliances", yaxis_title="Energy share [%]")
        show(fig, key=f"{key}|share", info="Share of each appliance in the energy of all appliances (within the "
             "sample limit). Blue: selected by the configuration ('out' or 'outEnergy'), grey: not used. Rarely "
             "used appliances with small shares are hard to disaggregate and lower the average scores.")
    with c2:
        if thr:
            on = 100 * (y[yout] > thr).mean(axis=0).reindex(share.index)
            fig = go.Figure(go.Bar(x=on.index.astype(str), y=on.values,
                                   marker_color=[SERIES[1] if a in sel else '#c3c2b7' for a in on.index],
                                   hovertemplate='%{x}: %{y:.1f} %<extra></extra>'))
            fig.update_layout(title=f"ON time (power > {fmt(thr)})", yaxis_title="ON time [%]")
            show(fig, key=f"{key}|on", info="Share of the time the appliance is above the ON threshold. This "
                 "threshold also defines the states of the accuracy and F1 score. Appliances that are almost always "
                 "OFF give high accuracy even for a poor model — look at F1 and TECA.")
        else:
            st.info("Set an ON threshold (Data › Pre-processing) to see the ON times.")
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Distribution of", yout, index=yout.index(sel[0]) if sel and sel[0] in yout else 0,
                         key=f"{key}|dist")
        v = y[a].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        fig = go.Figure(go.Histogram(x=v, nbinsx=80, marker_color=SERIES[0]))
        if thr:
            fig.add_vline(x=thr, line_dash='dash', line_color=SERIES[1], annotation_text='threshold')
        u = nilm.unit_of(d['units'], 'Output', a)
        fig.update_layout(title=f"Distribution of {a}", xaxis_title=f"{a}" + (f" [{u}]" if u else ''),
                          yaxis_title="Samples", yaxis_type='log', bargap=0.02)
        show(fig, key=f"{key}|hist", info="Histogram of the appliance power (log scale). Distinct peaks are the "
             "operating states (OFF, standby, ON levels). Multi-state appliances need more model capacity.")
    with c2:
        xin = nilm.channels(X)
        cols = xin + [c_ for c_ in sel if c_ in y]
        s = max(1, len(X) // 200000)
        corr = pd.concat([X[xin].iloc[::s], y[sel].iloc[::s]], axis=1).corr().loc[xin, sel] if sel and xin else None
        if corr is not None and not corr.empty:
            fig = go.Figure(go.Heatmap(z=corr.values, x=[str(c_) for c_ in corr.columns],
                                       y=[str(r) for r in corr.index], zmin=-1, zmax=1, colorscale='RdBu',
                                       reversescale=True, text=np.round(corr.values, 2), texttemplate='%{text}'))
            fig.update_layout(title="Correlation inputs ↔ appliances")
            show(fig, key=f"{key}|corr", info="Pearson correlation of every input feature with every selected "
                 "appliance. Large appliances (heat pump, dryer) correlate strongly with the aggregate; small ones "
                 "hardly — they need temporal context (windows) to be detected.")
        del cols


def split(cfg, d, key):
    X, _ = _limited(cfg, d)
    n = len(X)
    m = int(cfg['Exp'].get('method', 0))
    ts = nilm.sample_time(X)
    if m == 2:
        D = cfg['Dat']
        st.markdown("**Transfer**: whole datasets are used for each role.")
        table(pd.DataFrame([{'role': 'Training', 'datasets': ', '.join(map(str, D.get('train') or []))},
                            {'role': 'Validation', 'datasets': str(D.get('val'))},
                            {'role': 'Testing', 'datasets': str(D.get('test'))}]), hide_index=True)
        return
    if m == 3:
        if 'id' not in X:
            st.info("The dataset has no column 'id'.")
            return
        cnt = X['id'].value_counts().sort_index()
        idT = [float(i) for i in cfg['Dat'].get('idT') or []]
        idV = [float(i) for i in cfg['Dat'].get('idV') or []]
        role = ['Testing' if float(i) in idT else 'Validation' if float(i) in idV else 'Training' for i in cnt.index]
        fig = go.Figure(go.Bar(x=[str(i) for i in cnt.index], y=cnt.values, marker_color=[ROLE_COL[r] for r in role],
                               text=role, hovertemplate='id %{x}: %{y:,} samples (%{text})<extra></extra>'))
        fig.update_layout(title="Samples per id and role", xaxis_title="id", yaxis_title="Samples")
        show(fig, key=f"{key}|ids", info="Id-based split: the test ids are held out for testing, the validation ids "
             "for validation (taken from the training data, as loadData() does for train = 3).")
        return
    seg = nilm.split_segments(cfg, n)
    # thumbnail of the aggregate with the segments
    inp = nilm.channels(X)[0]
    s = max(1, n // MAX_POINTS)
    xs, xt = _time_axis(X['time'].iloc[::s])
    fig = go.Figure()
    t = np.asarray(xs) if not isinstance(xs, pd.DatetimeIndex) else xs
    rows = []
    if m == 0:
        lanes = {'Training': 2, 'Validation': 1, 'Testing': 0}
        for role, a, b in seg:
            xa, xb = _time_axis(X['time'].iloc[[a, max(b - 1, a)]])[0][[0, 1]] if b > a else (None, None)
            if xa is None:
                continue
            fig.add_trace(go.Scatter(x=[xa, xb], y=[lanes[role]] * 2, mode='lines', line=dict(width=16,
                                     color=ROLE_COL[role]), name=role, hovertemplate=f"{role}: {b - a:,} samples"
                                     "<extra></extra>"))
            rows.append({'role': role, 'samples': b - a, 'days': (b - a) * ts / 86400, 'from sample': a,
                         'to sample': b})
        fig.update_yaxes(tickvals=[0, 1, 2], ticktext=['Testing', 'Validation', 'Training'], range=[-0.6, 2.6])
        title = "1-fold split along the time axis"
        info = ("Data split of loadData() with Shuffle off: training = first rT of the samples, testing = the rest. "
                "The validation data are the first rV of the training part (train_test_split keeps the leading "
                "part), i.e. they overlap with the beginning of the training data.")
    else:
        for i, (_, a, b) in enumerate(seg):
            xa, xb = _time_axis(X['time'].iloc[[a, b - 1]])[0][[0, 1]]
            fig.add_trace(go.Scatter(x=[xa, xb], y=[0, 0], mode='lines', line=dict(width=16, color=color(i)),
                                     name=f"fold {i + 1}", showlegend=False,
                                     hovertemplate=f"fold {i + 1}: {b - a:,} samples<extra></extra>"))
            rows.append({'role': f"fold {i + 1}", 'samples': b - a, 'days': (b - a) * ts / 86400, 'from sample': a,
                         'to sample': b})
        fig.update_yaxes(visible=False, range=[-1, 1])
        title = f"k-fold split ({len(seg)} folds, each one tests once)"
        info = ("KFold of scikit-learn without shuffling: the samples are cut into k contiguous folds. In run k the "
                "fold k is the test data and the others train the model (validation: first rV of them).")
    fig.update_layout(title=title, xaxis_title=xt, height=240)
    show(fig, height=240, key=f"{key}|split", info=info)
    agg = go.Figure(line(t, X[inp].iloc[::s], str(inp), c=MUTED, width=1))
    agg.update_layout(xaxis_title=xt, yaxis_title=str(inp), margin=dict(t=20))
    show(agg, height=200, key=f"{key}|splitagg", info="Aggregate over the used samples (after the limit).")
    table(pd.DataFrame(rows), hide_index=True)
    if cfg['Dat'].get('Shuffle'):
        st.caption(":material/shuffle: Shuffle is on: the samples of the roles are drawn randomly, the bars show the "
                   "shares only.")


def _normalised(X, y, D):
    """X, y after normData.py with the normalisation of the configuration (statistics of the given slice)."""
    from src.general.helpFnc import normVal
    from src.data.normData import normData
    keys = ['normMaxX', 'normMaxY', 'normMinX', 'normMinY', 'normAvgX', 'normAvgY', 'normVarX', 'normVarY', 'normQX',
            'normQY', 'normQ1X', 'normQ1Y', 'normQ3X', 'normQ3Y']
    exp = {'status': {w: {'count': 0, 'msg': [], 'idx': 0} for w in ('warnL', 'warnH')}}
    with contextlib.redirect_stdout(io.StringIO()):
        s = {**D, **dict(zip(keys, normVal(X, y)))}
        Xn, yn, _ = normData(X.copy(), y.copy(), s, exp, 0)
    return pd.DataFrame(Xn, columns=X.columns), pd.DataFrame(yn, columns=y.columns)


def pipeline(cfg, d, key):
    """Resampling, normalisation and framing of a short slice with the toolkit functions."""
    from src.data.resampleData import resampleData
    from src.general.framing import framing
    D, P = cfg['Dat'], cfg['Par']
    X, y = d['X'], d['y']
    ts = nilm.sample_time(X)
    fs_raw = 1 / ts if ts else 0
    sel = nilm.selected_outputs(cfg, y)
    xin = [c_ for c_ in (D.get('inp') or nilm.channels(X)) if c_ in X]
    n = min(len(X), int(max(2000, 4 * P.get('window', 20) * max(fs_raw / max(D.get('fs') or fs_raw, 1e-12), 1))))
    c1, c2 = st.columns([3, 1])
    with c2:
        start = st.number_input("Start sample", 0, max(len(X) - n, 0), 0, step=max(n // 2, 1), key=f"{key}|p0")
    sl = slice(int(start), int(start) + n)
    setup = {'fs': D.get('fs') or fs_raw}
    with contextlib.redirect_stdout(io.StringIO()):
        Xr, _ = resampleData(X[['time'] + xin].iloc[sl].reset_index(drop=True), fs_raw, setup)
        yr, _ = resampleData(y[['time'] + sel].iloc[sl].reset_index(drop=True), fs_raw, setup)
    with c1:
        st.markdown(f"**1 · Resampling** {fmt(fs_raw)} Hz → {fmt(setup['fs'])} Hz "
                    f"({len(X.iloc[sl]):,} → {len(Xr):,} samples; resampleData.py averages each new interval when downsampling)")
    inp = xin[0] if xin else None
    if inp is None:
        st.info("No input features.")
        return
    xa, xt = _time_axis(X['time'].iloc[sl])
    xb, _ = _time_axis(Xr['time'])
    fig = go.Figure([line(xa, X[inp].iloc[sl], f"{inp} raw", c=MUTED, width=1),
                     line(xb, Xr[inp], f"{inp} resampled", 0, width=1.6, line_shape='hv')])
    fig.update_layout(xaxis_title=xt, yaxis_title=str(inp))
    show(fig, height=300, key=f"{key}|res", info="Raw aggregate and the series after resampling to the sampling "
         "frequency fs of the configuration. A lower fs averages short events away.")

    # Framing
    st.markdown("**2 · Framing and target**")
    if P.get('frame') != 1:
        st.info("Framing is off (frame = 0): every time step is one sample.")
        return
    from plotly.subplots import make_subplots
    W = int(P.get('window', 20))
    ov = int(P.get('overlap', W - 1))
    ov = W - 1 if ov < 0 else min(ov, W - 1)
    step = W - ov
    Xs, ys = Xr[xin].reset_index(drop=True), yr[sel].reset_index(drop=True) if sel else None
    c1, c2 = st.columns([3, 1])
    with c2:
        norm = st.toggle("Normalised (model input)", key=f"{key}|norm",
                         help="Inputs and outputs after normData.py with the normalisation of the configuration "
                              "(statistics of this slice) — the values the model actually sees.")
    if norm and ys is not None:
        Xs, ys = _normalised(Xs, ys, D)
    with contextlib.redirect_stdout(io.StringIO()):
        fx = framing(Xs, W, ov)
        fy = framing(ys, W, ov) if ys is not None else None
    nfr = len(fx)
    if nfr == 0:
        st.info(f"The slice is shorter than one window ({W} samples).")
        return
    with c1:
        k = st.slider("Frame", 0, nfr - 1, min(nfr // 2, nfr - 1), key=f"{key}|frame",
                      help=f"{nfr:,} frames of {W} samples; a new frame starts every {step} sample(s) "
                           f"(overlap {ov}).")
    ta, tt = _time_axis(Xr['time'])
    ta = np.asarray(ta)
    ts_new = 1 / setup['fs'] if setup['fs'] else ts
    i0, i1 = k * step, k * step + W - 1
    os_ = int(P.get('outseq', 0))
    foc = P.get('yFocus', W // 2)
    foc = None if foc is None or foc < 0 else min(int(foc), W - 1)
    st.caption(f"Window {W} samples × {fmt(ts_new / 60)} min = **{fmt(W * ts_new / 60)} min** of the aggregate per "
               f"frame · a new frame every {step} sample(s) = {fmt(step * ts_new / 60)} min · frame {k:,} covers "
               f"samples {i0:,}–{i1:,} of the slice.")

    # Context: the frame (and the next ones) on the series, one row per input
    rows = len(xin) + (1 if fy is not None else 0)
    names = list(xin) + (["Appliances"] if fy is not None else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.025)
    lo, hi = max(0, i0 - 3 * W), min(len(ta) - 1, i1 + 3 * W)
    for r, c_ in enumerate(xin):
        fig.add_trace(line(ta[lo:hi + 1], Xs[c_].values[lo:hi + 1], str(c_), r, width=1.3), row=r + 1, col=1)
    if fy is not None:
        for i, a in enumerate(sel[:8]):
            fig.add_trace(line(ta[lo:hi + 1], ys[a].values[lo:hi + 1], str(a), len(xin) + i, width=1.1),
                          row=rows, col=1)
    for n in range(1, 4):                                              # the next frames (overlap)
        kk = k + n
        if kk < nfr and kk * step + W - 1 < len(ta):
            fig.add_vrect(x0=ta[kk * step], x1=ta[kk * step + W - 1], fillcolor=SERIES[2], opacity=0.05,
                          line_width=0, row='all', col=1)
    fig.add_vrect(x0=ta[i0], x1=ta[i1], fillcolor=SERIES[2], opacity=0.16, line_width=1, line_color=SERIES[2],
                  row='all', col=1)
    if os_ == 0 and foc is not None:
        fig.add_vline(x=ta[i0 + foc], line_dash='dash', line_color=SERIES[7], row='all', col=1)
    for r, n_ in enumerate(names):
        fig.update_yaxes(title_text=n_, row=r + 1, col=1, title_font_size=11)
    fig.update_xaxes(title_text=tt, row=rows, col=1)
    fig.update_layout(title=f"Frame {k} on the series (green box; the next frames lighter)", showlegend=True)
    show(fig, height=95 * rows + 150, key=f"{key}|ctx", legend_below=True, info="Every input feature has its own "
         "row and scale. The green box is the selected input window; the lighter boxes are the next frames, shifted "
         f"by {step} sample(s) — with overlap = window − 1 almost every frame shares {ov} samples with its "
         "neighbour. The red dashed line is the focus point whose appliance power is the seq2point target.")

    # The frame itself: input window and target
    cl, cr = st.columns(2)
    xw = ta[i0:i1 + 1]
    with cl:
        fig = make_subplots(rows=len(xin), cols=1, shared_xaxes=True, vertical_spacing=0.04)
        for r, c_ in enumerate(xin):
            fig.add_trace(line(xw, fx[k, :, r], str(c_), r, width=1.6, mode='lines+markers', marker=dict(size=4)),
                          row=r + 1, col=1)
            fig.update_yaxes(title_text=str(c_), row=r + 1, col=1, title_font_size=11)
        fig.update_layout(title=f"Input window · shape {W} × {len(xin)}", showlegend=False)
        show(fig, height=max(260, 85 * len(xin) + 120), key=f"{key}|fx", info="One input sample of the model: "
             "window length × input features (each feature on its own scale). The model sees only this window.")
    with cr:
        if fy is not None:
            fig = go.Figure()
            for i, a in enumerate(sel[:8]):
                fig.add_trace(line(xw, fy[k, :, i], str(a), len(xin) + i, width=1.3))
            if os_ >= 1:
                delta = int((W - os_) / 2)
                fig.add_vrect(x0=xw[delta], x1=xw[min(W - delta - 1, W - 1)], fillcolor=SERIES[2], opacity=0.14,
                              line_width=0, annotation_text="target sequence")
                mode = f"seq2subseq ({os_} samples)" if os_ < W else "seq2seq"
                tv = None
            elif foc is None:
                mode, tv = "seq2point (mean of the window)", np.nanmean(fy[k], axis=0)
            else:
                fig.add_vline(x=xw[foc], line_dash='dash', line_color=SERIES[7], annotation_text="focus point")
                mode, tv = f"seq2point (sample {foc})", fy[k, foc, :]
            if tv is not None:
                x_t = xw[foc] if foc is not None else xw[W // 2]
                fig.add_trace(go.Scatter(x=[x_t] * min(len(sel), 8), y=tv[:8], mode='markers', name='target',
                                         marker=dict(size=10, symbol='diamond', color=[color(len(xin) + i) for i in
                                                                                       range(min(len(sel), 8))],
                                                     line=dict(width=1, color='white'))))
            fig.update_layout(title=f"Target: {mode}", yaxis_title='normalised' if norm else 'Power')
            show(fig, height=max(260, 85 * len(xin) + 120), key=f"{key}|fy", legend_below=True, info="Appliance "
                 "power in the same window. createSeq.py takes the value at the focus point (diamonds, seq2point), "
                 "the centred sub-sequence (seq2subseq) or the whole window (seq2seq) as target of the model.")

    # Normalisation
    st.markdown("**3 · Normalisation**")
    names = {0: 'none', 1: '−1/+1', 2: '0/1', 3: 'mean/std', 4: 'Q1/Q3', 5: 'quantile'}
    st.caption(f"Inputs: {names.get(D.get('inpNorm'), D.get('inpNorm'))} · outputs: "
               f"{names.get(D.get('outNorm'), D.get('outNorm'))} — the statistics are computed on the training data "
               f"(normVal in loadData.py) and inverted after the prediction (postprocess.py).")


#######################################################################################################################
# Explorer
#######################################################################################################################
def explorer(cfg, key='ex', name=None):
    """Data explorer of the dataset name (default: first training dataset of the configuration)."""
    D = cfg['Dat']
    name = name or nilm.first_train(cfg)
    folder, typ = D.get('folder', ''), D.get('type', 'mat')
    if not name or not nilm.exists(cfg, name):
        st.warning(f"Dataset '{name}.{typ}' not found in data/{folder}. Choose folder, file format and training "
                   f"dataset on the Setup page.")
        return
    size = nilm.file_size(folder, name, typ)
    if size > 150e6 and not st.session_state.get(f"big|{name}"):
        st.info(f"data/{folder}/{name}.{typ} has {nilm.human(size)} — loading takes a while and needs memory.")
        if st.button(f"Load {name}", icon=':material/download:', key=f"{key}|big"):
            st.session_state[f"big|{name}"] = True
            st.rerun()
        return
    try:
        d = nilm.load_cfg(cfg, name)
    except Exception as e:
        st.error(str(e))
        return
    st.caption(f":material/database: data/{folder + '/' if folder else ''}{name}.{typ} · {nilm.human(size)} · "
               f"read with the toolkit loader (cached)")
    tabs = st.tabs([":material/summarize: Overview", ":material/show_chart: Time series",
                    ":material/electrical_services: Appliances", ":material/call_split: Split",
                    ":material/filter_alt: Pre-processing & framing"])
    with tabs[0]:
        overview(cfg, d, name)
    with tabs[1]:
        timeseries(cfg, d, f"{key}|ts")
    with tabs[2]:
        appliances(cfg, d, f"{key}|ap")
    with tabs[3]:
        split(cfg, d, f"{key}|sp")
    with tabs[4]:
        try:
            pipeline(cfg, d, f"{key}|pp")
        except Exception as e:
            st.warning(f"Preview not available ({type(e).__name__}: {e}).")


# Inputs per mode: mode -> function(cfg)
INPUTS = {0: lambda cfg: explorer(cfg, key='m0'), 1: lambda cfg: explorer(cfg, key='m1'),
          2: lambda cfg: explorer(cfg, key='m2')}
