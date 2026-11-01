#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         results_nilm.py
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
Interactive results of the BaseNILM operating modes. The data are recorded by adapter.run() from the calls of
performance(), featureRank() and printResults() in src/test.py: per test the appliance labels, the result matrix of
performance() (columns ACC, F1, R2, TECA, RMSE, MAE, MAX, EST, TRUTH as in printResults.py), the averages and the
ground truth / predictions after post-processing. TF trainings add their training_log.csv.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core.style import show, line, table, fmt, color, SERIES, MUTED

import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PCT = ['ACC', 'F1', 'R2', 'TECA', 'EST', 'TRUTH']                     # stored as fraction, shown in %
COLS = {'ACC': 'Accuracy [%]', 'F1': 'F1 [%]', 'R2': 'R² [%]', 'TECA': 'TECA [%]', 'RMSE': 'RMSE', 'MAE': 'MAE',
        'MAX': 'Max. error', 'SAE': 'SAE', 'EST': 'Est. share [%]', 'TRUTH': 'True share [%]'}
INFO = {
    'tiles': "AVG row of the console table (performance.py). TECA (total energy correctly assigned) = 1 − Σ|ŷ − y| "
             "/ (2 Σ|y|) over all appliances at once, so large appliances weigh more; 100 % is perfect. Accuracy and "
             "F1 (weighted over ON and OFF) use the ON/OFF states of the ON threshold; RMSE and MAE are means over "
             "the appliances (unit of the outputs).",
}


#######################################################################################################################
# Helper
#######################################################################################################################
def frame(t):
    """Per-appliance result table (as printResults) incl. the AVG row."""
    r = np.atleast_2d(t['result'])
    labels = list(t['labels'])[:r.shape[0]] + [f"#{i}" for i in range(len(t['labels']), r.shape[0])]
    df = pd.DataFrame(r[:, :9], columns=A.METRICS, index=labels)
    avg = pd.DataFrame([np.asarray(t['avg'])[:9]], columns=A.METRICS, index=['AVG'])
    df = pd.concat([df, avg])
    df['SAE'] = (df['EST'] - df['TRUTH']).abs() / (df['TRUTH'] + 1e-9)
    for c in PCT:
        df[c] = 100 * df[c]
    return df[['ACC', 'F1', 'R2', 'TECA', 'RMSE', 'MAE', 'MAX', 'SAE', 'EST', 'TRUTH']]


def _tiles(df, n_tests=1, std=None):
    avg = df.loc['AVG']
    c = st.columns(5)
    for col, k, unit in zip(c, ['TECA', 'F1', 'ACC', 'RMSE', 'MAE'], ['%', '%', '%', '', '']):
        delta = None
        if std is not None:
            delta = f"± {fmt(std.loc['AVG', k])}"
        col.metric(COLS[k].split(' [')[0], fmt(avg[k], unit), delta=delta, delta_color='off',
                   help=INFO['tiles'] + (f" Mean over {n_tests} folds, ± standard deviation." if n_tests > 1 else ''))


def _days(n, fs, step=1):
    return np.arange(n) * step / fs / 86400 if fs else np.arange(n) * step


#######################################################################################################################
# Simulation
#######################################################################################################################
def simulation(data, setup, key):
    tests = data.get('tests') or []
    logs = data.get('train_logs') or {}
    if not tests:
        st.info("No test results in this run (Testing off" + (" — the training curves are below)." if logs else ")."))
        if logs:
            training(logs, key)
        return

    # Fold selection (k-fold)
    if len(tests) > 1:
        names = ['mean of all folds'] + [t['name'] for t in tests]
        pick = st.segmented_control("Fold", list(range(len(names))), default=0, key=f"{key}|fold",
                                    format_func=lambda i: names[i].split('_')[-1] if i else names[0]) or 0
        frames = [frame(t) for t in tests]
        if pick == 0:
            df = sum(frames) / len(frames)
            std = pd.concat(frames).groupby(level=0).std().reindex(df.index)
            t = tests[-1]
        else:
            df, std, t = frames[pick - 1], None, tests[pick - 1]
        _tiles(df, len(tests) if pick == 0 else 1, std if pick == 0 else None)
    else:
        t = tests[0]
        df = frame(t)
        frames = [df]
        _tiles(df)
    labels = [i for i in df.index if i != 'AVG']

    tabs = st.tabs([":material/table: Scores", ":material/show_chart: Disaggregation", ":material/pie_chart: Energy",
                    ":material/insights: Training & features"] + ([":material/stacked_bar_chart: Folds"]
                                                                    if len(tests) > 1 else []))
    with tabs[0]:
        table(df.rename(columns=COLS), height=min(520, 38 + 35 * len(df)))
        fig = go.Figure()
        for i, m in enumerate(['TECA', 'F1', 'ACC']):
            fig.add_trace(go.Bar(x=labels, y=df.loc[labels, m], name=COLS[m].split(' [')[0], marker_color=color(i),
                                 hovertemplate='%{x}: %{y:.1f} %<extra>' + m + '</extra>'))
        fig.update_layout(barmode='group', yaxis_title="Score [%]", title="Scores per appliance",
                          yaxis_range=[min(0, float(np.nanmin(df.loc[labels, ['TECA', 'F1', 'ACC']].values)) - 5), 100])
        show(fig, key=f"{key}|scores", info="TECA, F1 and accuracy of every appliance. Small, rarely used "
             "appliances typically score high accuracy (they are mostly OFF) but low TECA/F1; large appliances "
             "with distinct signatures (heat pump, dryer) are easiest.")

    with tabs[1]:
        yT, yP = np.atleast_2d(t['yTrue']), np.atleast_2d(t['yPred'])
        if yT.shape[0] == 1 and yT.shape[1] > 1 and len(labels) == 1:
            yT, yP = yT.T, yP.T
        if yT.size == 0:
            st.info("No time series stored for this run.")
        else:
            step = t.get('step', 1)
            n = yT.shape[0]
            x = _days(n, t.get('fs'), step)
            c1, c2 = st.columns([1, 3])
            with c1:
                apps = st.multiselect("Appliances", labels, default=labels[:1], key=f"{key}|apps")
                span = st.selectbox("Span", [1, 3, 7, 14, 30, 'all'], index=2, key=f"{key}|span",
                                    format_func=lambda d: 'all' if d == 'all' else f"{d} days") if t.get('fs') else 'all'
            days = x[-1] if len(x) else 0
            w = days if span == 'all' else min(span, days)
            with c2:
                x0 = st.slider("Start [day]", 0.0, float(max(days - w, 0.0)), 0.0, key=f"{key}|x0") if days > w else 0.0
            sel = (x >= x0) & (x <= x0 + w)
            s = max(1, int(sel.sum() // 6000))
            fig = go.Figure()
            for i, a in enumerate(apps):
                j = labels.index(a)
                fig.add_trace(line(x[sel][::s], yT[sel, j][::s], f"{a} true", i, width=1.2))
                fig.add_trace(line(x[sel][::s], yP[sel, j][::s], f"{a} predicted", i, dash='dot', width=1.6))
            fig.update_layout(xaxis_title="Time in the test data [days]" if t.get('fs') else "Sample",
                              yaxis_title="Power", hovermode='x unified', title="Ground truth vs. prediction")
            show(fig, height=440, key=f"{key}|ts", info="Appliance power of the test data (solid) and the "
                 "disaggregated estimate of the model (dotted), after post-processing (de-normalisation, clipping to "
                 "outMin/outMax). Look for missed activations, false ON states and the tracking of the amplitude.")
            c1, c2 = st.columns(2)
            with c1:
                a = st.selectbox("Scatter of", labels, key=f"{key}|sc")
                j = labels.index(a)
                ss = max(1, n // 5000)
                fig = go.Figure(go.Scattergl(x=yT[::ss, j], y=yP[::ss, j], mode='markers',
                                             marker=dict(size=4, color=SERIES[0], opacity=0.4), name=a))
                mx = float(np.nanmax([yT[:, j].max(), yP[:, j].max(), 1]))
                fig.add_trace(line([0, mx], [0, mx], 'ideal', c=MUTED, dash='dash', width=1))
                fig.update_layout(xaxis_title="True", yaxis_title="Predicted", title=f"{a}: predicted vs. true",
                                  showlegend=False)
                show(fig, height=360, key=f"{key}|scatter", info="Every point is one test sample. Points on the "
                     "diagonal are perfect; a horizontal band at 0 shows missed activations, a vertical band at 0 "
                     "false ON predictions.")
            with c2:
                err = yP - yT
                fig = go.Figure([go.Box(y=err[::max(1, n // 20000), labels.index(a_)], name=str(a_),
                                        marker_color=color(i), boxpoints=False) for i, a_ in enumerate(labels)])
                fig.update_layout(yaxis_title="Prediction error", title="Error distribution per appliance",
                                  showlegend=False)
                show(fig, height=360, key=f"{key}|err", info="Distribution of ŷ − y per appliance (box: quartiles, "
                     "whiskers: 1.5 IQR). A box shifted from zero is a systematic over/under-estimation.")

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure([go.Bar(x=labels, y=df.loc[labels, 'TRUTH'], name='true', marker_color=SERIES[0]),
                             go.Bar(x=labels, y=df.loc[labels, 'EST'], name='estimated', marker_color=SERIES[1])])
            fig.update_layout(barmode='group', yaxis_title="Share of the total energy [%]",
                              title="Energy share: estimated vs. true")
            show(fig, key=f"{key}|share", info="Share of each appliance in the total energy of the test data, true "
                 "and as estimated by the model (EST / TRUTH columns of the console table).")
        with c2:
            fig = go.Figure(go.Pie(labels=labels, values=np.clip(df.loc[labels, 'EST'], 0, None), hole=0.5,
                                   marker=dict(colors=[color(i) for i in range(len(labels))]), sort=False))
            fig.update_layout(title="Estimated energy breakdown")
            show(fig, key=f"{key}|pie", info="Energy breakdown of the test period as the model estimates it — the "
                 "typical NILM output for the user. Compare with the true shares on the left.")
        table(df.loc[labels, ['EST', 'TRUTH', 'SAE']].rename(columns=COLS), height=min(360, 38 + 35 * len(labels)))

    with tabs[3]:
        if logs:
            training(logs, key)
        else:
            st.caption("No training in this run (the model was loaded) — training curves are recorded when "
                       "Training is on (TF).")
        if t.get('feat') is not None:
            score, errv = t['feat']
            try:
                score, errv = pd.Series(score).sort_values(ascending=False), pd.Series(errv)
                if np.any(np.asarray(score.values, dtype=float) != 0):
                    fig = go.Figure(go.Bar(x=[str(i) for i in score.index], y=score.values, marker_color=SERIES[2],
                                           error_y=dict(type='data', array=errv.reindex(score.index).values)))
                    fig.update_layout(title="Feature ranking (permutation importance)", yaxis_title="Importance")
                    show(fig, key=f"{key}|feat", info="Mean decrease of a random forest's score when one input "
                         "feature is shuffled (featureRank.py, averaged over the appliances, ± std). Features near "
                         "zero carry no information for the disaggregation.")
                else:
                    st.caption("Feature ranking off (ranking = 0).")
            except Exception:
                pass

    if len(tests) > 1:
        with tabs[4]:
            m = st.segmented_control("Metric", ['TECA', 'F1', 'ACC', 'RMSE', 'MAE'], default='TECA',
                                     key=f"{key}|fm") or 'TECA'
            fig = go.Figure()
            for i, a in enumerate(labels + ['AVG']):
                fig.add_trace(go.Box(y=[f_.loc[a, m] for f_ in frames], name=str(a), boxpoints='all',
                                     marker_color=MUTED if a == 'AVG' else color(i)))
            fig.update_layout(yaxis_title=COLS[m], title=f"{COLS[m]} across the {len(frames)} folds",
                              showlegend=False)
            show(fig, key=f"{key}|folds", info="Spread of the score over the folds of the cross-validation. A "
                 "wide box means the result depends on which part of the data is tested (seasonal loads, rare "
                 "events).")


def training(logs, key):
    pick = list(logs)
    name = st.selectbox("Model", pick, key=f"{key}|log") if len(pick) > 1 else pick[0]
    df = logs[name]
    ep = df['epoch'] + 1 if 'epoch' in df else np.arange(1, len(df) + 1)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure([line(ep, df['loss'], 'training', 0)] + ([line(ep, df['val_loss'], 'validation', 1)]
                                                              if 'val_loss' in df else []))
        fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss", yaxis_type='log')
        show(fig, height=320, key=f"{key}|loss", info="Training and validation loss per epoch (training_log.csv). "
             "A validation loss rising while the training loss falls indicates over-fitting; early stopping keeps "
             "the best epoch.")
    with c2:
        met = [c for c in df.columns if c not in ('epoch', 'loss', 'val_loss', 'learning_rate', 'lr')
               and not c.startswith('val_')]
        fig = go.Figure()
        for i, mname in enumerate(met[:2]):
            fig.add_trace(line(ep, df[mname], mname, i * 2))
            if f"val_{mname}" in df:
                fig.add_trace(line(ep, df[f"val_{mname}"], f"val_{mname}", i * 2 + 1))
        lr = 'learning_rate' if 'learning_rate' in df else 'lr' if 'lr' in df else None
        if lr:
            fig.add_trace(line(ep, df[lr], 'learning rate', 3, dash='dot', yaxis='y2'))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', type='log', title='learning rate',
                                          showgrid=False))
        fig.update_layout(title="Metric and learning rate", xaxis_title="Epoch")
        show(fig, height=320, key=f"{key}|met", info="Training metric (e.g. TECA via lossMetric) and the learning "
             "rate, which ReduceLROnPlateau lowers when the validation loss stalls.")
    st.caption(f"{len(df)} epochs · best validation loss "
               f"{fmt(df['val_loss'].min()) if 'val_loss' in df else '–'}")


#######################################################################################################################
# Hyperparameter optimisation
#######################################################################################################################
SK_MAP = {'RF': {'max_depth': 'Mdl.SK_RF_depth', 'n_estimators': 'Mdl.SK_RF_estimators', 'random_state':
                 'Mdl.SK_RF_state'},
          'SVM': {'kernel': 'Mdl.SK_SVM_kernel', 'C': 'Mdl.SK_SVM_C', 'gamma': 'Mdl.SK_SVM_gamma', 'epsilon':
                  'Mdl.SK_SVM_epsilon'},
          'KNN': {'n_neighbors': 'Mdl.SK_KNN_neighbors', 'weights': 'Mdl.SK_KNN_weights'}}
TUNER_KEYS = ('tuner/epochs', 'tuner/initial_epoch', 'tuner/bracket', 'tuner/round', 'tuner/trial_id')


def _trials(o):
    """Trials of the optimisation as table: trial, score, hyperparameters (tuner bookkeeping removed)."""
    rows = []
    for t in o.get('trials') or []:
        if t.get('score') is None:
            continue
        r = {'Trial': str(t.get('trial')), 'Score': float(t['score'])}
        r.update({k: v for k, v in (t.get('hp') or {}).items() if k not in TUNER_KEYS})
        if 'std' in t:
            r['± CV'] = t['std']
        if 'time' in t:
            r['Fit time [s]'] = t['time']
        rows.append(r)
    return pd.DataFrame(rows)


def _parcoords(df, hp, lower, key):
    dims = []
    for c in hp:
        s = df[c]
        if s.isna().all():
            continue
        if pd.api.types.is_bool_dtype(s) or set(s.dropna().unique()) <= {True, False}:
            dims.append(dict(label=c, values=s.fillna(False).astype(int), tickvals=[0, 1], ticktext=['no', 'yes']))
        elif pd.api.types.is_numeric_dtype(s):
            dims.append(dict(label=c, values=s.astype(float).fillna(s.astype(float).median())))
        else:
            cats = sorted(s.astype(str).unique())
            dims.append(dict(label=c, values=s.astype(str).map({v: i for i, v in enumerate(cats)}),
                             tickvals=list(range(len(cats))), ticktext=cats))
    if not dims:
        return
    dims.append(dict(label='Score', values=df['Score']))
    fig = go.Figure(go.Parcoords(line=dict(color=df['Score'], colorscale='Blues_r' if lower else 'Blues',
                                           showscale=True), dimensions=dims))
    show(fig, height=420, key=f"{key}|par", info="Every line is one trial; the colour is its score (darker = better). "
         "Lines of good trials that pass through the same range of an axis show which values of that "
         "hyperparameter work. Drag on an axis to filter the trials.")


def hyper(data, setup, key):
    from gui.core import grid as G
    o = data.get('opti')
    if not o:
        st.info("No optimisation result (results/<name>/opti_<model>_<name>.json) was found — the tuner output of "
                "the console is shown below.")
        log = data.get('_log', '')
        lines = [l for l in log.splitlines() if re.search(r'best|Best|score|Trial|val_loss', l)]
        st.code('\n'.join(lines[-80:]) or '(no tuner output found)', language=None)
        return
    df = _trials(o)
    lower = o.get('solver') == 'TF'                                    # val_loss: lower is better
    best = o.get('best') or {}
    hp = [c for c in df.columns if c not in ('Trial', 'Score', '± CV', 'Fit time [s]')]
    c = st.columns(4)
    c[0].metric("Search", f"{o.get('solver')} · {o.get('model')}", help="TF: Keras-Tuner Hyperband over the CNN "
                "search space tfMdloptiRCNN (models.py). SK: GridSearchCV over the grid of trainMdlSKopti.py.")
    c[1].metric("Trials", f"{len(df)}")
    sc = df['Score'].min() if lower else df['Score'].max() if len(df) else o.get('best_score')
    c[2].metric("Best " + ("val_loss" if lower else "CV score"), fmt(sc) if sc is not None else '–',
                help=o.get('objective', ''))
    if o.get('best_epoch'):
        c[3].metric("Best epoch (final fit)", f"{o['best_epoch']}")
    elif o.get('best_score') is not None:
        c[3].metric("Best mean CV score", fmt(o['best_score']))

    cl, cr = st.columns([2, 3])
    with cl:
        st.markdown("**Best hyperparameters**")
        table(pd.DataFrame({'Parameter': list(k for k in best if k not in TUNER_KEYS),
                            'Value': [str(best[k]) for k in best if k not in TUNER_KEYS]}), hide_index=True)
        m = SK_MAP.get(o.get('model')) if o.get('solver') == 'SK' else None
        if m:
            vals = {ref: best[k] for k, ref in m.items() if k in best}
            if vals and st.button("Apply to the configuration", icon=':material/input:', key=f"{key}|apply"):
                G.apply(vals, "Best hyperparameters applied (Parameters › Model)")
                st.rerun()
        elif o.get('solver') == 'TF':
            st.caption("The tuned CNN (tfMdloptiRCNN) is a separate search space in models.py — transfer the layer "
                       "sizes to tfMdlCNN by hand; the learning rate maps to Model › Learning rate.")
            if 'learning_rate' in best and st.button("Apply the learning rate", icon=':material/input:',
                                                      key=f"{key}|lr"):
                G.apply({'Mdl.lr': best['learning_rate']}, "Learning rate applied")
                st.rerun()
    with cr:
        if len(df):
            d = df.sort_values('Score', ascending=lower).reset_index(drop=True)
            cols = [SERIES[2] if i == 0 else SERIES[0] for i in range(len(d))]
            fig = go.Figure(go.Bar(x=d['Trial'], y=d['Score'], marker_color=cols,
                                   hovertext=[', '.join(f"{k}={r[k]}" for k in hp if pd.notna(r[k]))
                                              for _, r in d.iterrows()]))
            fig.update_layout(title="Score per trial (sorted, best in green)", xaxis_title="Trial",
                              yaxis_title='val_loss' if lower else 'mean CV score', xaxis_type='category')
            show(fig, height=330, key=f"{key}|bar", info="TF: validation loss of each Hyperband trial (lower is "
                 "better). SK: mean score of the 5 cross-validation folds (R² for regression — negative values mean "
                 "worse than predicting the mean; the folds are consecutive time blocks).")
    if len(df) > 1 and hp:
        _parcoords(df, hp, lower, key)
    hist = o.get('history')
    if hist:
        fig = go.Figure([line(np.arange(1, len(v) + 1), v, k, i) for i, (k, v) in enumerate(hist.items())
                         if k in ('loss', 'val_loss', 'mse', 'val_mse')])
        fig.update_layout(title="Final training of the best model", xaxis_title="Epoch", yaxis_title="Loss")
        show(fig, height=300, key=f"{key}|hist", info="The best hyperparameters are trained once more "
             "(trainMdlTFopti.py); the best epoch is the one with the lowest validation loss.")
    with st.expander("All trials"):
        table(df, hide_index=True)


#######################################################################################################################
# Grid optimisation
#######################################################################################################################
HIGHER = ('TECA', 'F1', 'ACC', 'R2')


def grid(data, setup, key):
    from gui.core import grid as G
    g = data.get('grid')
    tests = data.get('tests') or []
    if not g or not g.get('points'):
        st.info("No grid results recorded.")
        return
    v1, v2, r1, r2 = g['values1'], g['values2'] or [None], g['var1'], g.get('var2')
    by = {tuple(t['grid']): t for t in tests if t.get('grid') is not None}
    labels = next((list(t['labels']) for t in tests), [])
    c1, c2 = st.columns([3, 2])
    with c1:
        m = st.segmented_control("Metric", ['TECA', 'F1', 'ACC', 'R2', 'RMSE', 'MAE'], default='TECA',
                                 key=f"{key}|gm") or 'TECA'
    with c2:
        app = st.selectbox("Appliance", ['AVG'] + labels, key=f"{key}|gapp",
                           format_func=lambda a: 'Average (AVG row)' if a == 'AVG' else a)
    Z = np.full((len(v1), len(v2)), np.nan)
    for (i, j), t in by.items():
        try:
            Z[i, j] = frame(t).loc[app, m]
        except KeyError:
            pass
    hi = m in HIGHER
    x2 = [G.fmt_value(r2, v) for v in v2]
    y1 = [G.fmt_value(r1, v) for v in v1]
    if r2:
        fig = go.Figure(go.Heatmap(z=Z, x=x2, y=y1, colorscale='Blues' if hi else 'Blues_r',
                                   text=[['' if v != v else fmt(v) for v in row] for row in Z],
                                   texttemplate='%{text}', colorbar=dict(title=COLS[m])))
        fig.update_layout(title=f"{COLS[m]} · {'average' if app == 'AVG' else app} over {G.short(r1)} × "
                                f"{G.short(r2)}", xaxis_title=G.label(r2), yaxis_title=G.label(r1),
                          xaxis_type='category', yaxis_type='category')
        show(fig, height=max(320, 70 * len(v1) + 140), key=f"{key}|heat", info="Each cell is one complete "
             "simulation (training + testing) with these two values; all other parameters as configured. Darker "
             "is better. For window × fs the time covered by one window is window / fs.")
    else:
        fig = go.Figure([line(y1, Z[:, 0], COLS[m], 0, mode='lines+markers')])
        fig.update_layout(title=f"{COLS[m]} over {G.short(r1)}", xaxis_title=G.label(r1), yaxis_title=COLS[m],
                          xaxis_type='category')
        show(fig, height=340, key=f"{key}|sweep", info="One complete simulation per value of the swept parameter.")
    if np.isfinite(Z).any():
        b = np.unravel_index(np.nanargmax(Z) if hi else np.nanargmin(Z), Z.shape)
        txt = f"{G.short(r1)} = {y1[b[0]]}" + (f" and {G.short(r2)} = {x2[b[1]]}" if r2 else '')
        extra = ''
        if {r1, r2} == {'Par.window', 'Dat.fs'}:
            W, fs = (v1[b[0]], v2[b[1]]) if r1 == 'Par.window' else (v2[b[1]], v1[b[0]])
            extra = f" — one window covers {fmt(W / fs / 60)} min"
        c1, c2 = st.columns([4, 1])
        c1.success(f"Best {m} ({'average' if app == 'AVG' else app}): **{fmt(Z[b])}** at {txt}{extra}.")
        with c2:
            if st.button("Apply best values", icon=':material/input:', key=f"{key}|gapply",
                         help="Writes the values of the best grid point into the configuration."):
                vals = {r1: v1[b[0]]}
                if r2:
                    vals[r2] = v2[b[1]]
                G.apply(vals, "Best grid point applied to the configuration")
                st.rerun()

    # All appliances and run time
    cl, cr = st.columns(2)
    with cl:
        if labels and len(by):
            rows = []
            for (i, j), t in sorted(by.items()):
                f = frame(t)
                rows.append([f.loc[a, m] if a in f.index else np.nan for a in labels])
            pts = [f"{y1[i]}" + (f" · {x2[j]}" if r2 else '') for (i, j) in sorted(by)]
            A_ = np.array(rows).T
            fig = go.Figure(go.Heatmap(z=A_, x=pts, y=labels, colorscale='Blues' if hi else 'Blues_r',
                                       colorbar=dict(title=m)))
            fig.update_layout(title=f"{m} per appliance and grid point", xaxis_type='category')
            show(fig, height=max(280, 40 * len(labels) + 160), key=f"{key}|apps", info="Which appliances profit "
                 "from which setting: e.g. long windows help appliances with long cycles (heat pump, dryer), short "
                 "events need a high sampling frequency.")
    with cr:
        pts = g['points']
        fig = go.Figure(go.Bar(x=[f"{G.fmt_value(r1, p['v1'])}" + (f" · {G.fmt_value(r2, p['v2'])}" if r2 else '')
                                  for p in pts], y=[p['time'] for p in pts],
                               marker_color=[SERIES[7] if p['error'] else SERIES[0] for p in pts]))
        fig.update_layout(title="Run time per grid point", yaxis_title="Time [s]", xaxis_type='category')
        show(fig, height=max(280, 40 * len(labels) + 160), key=f"{key}|time", info="Training + testing time of "
             "each grid point (red: failed).")
    rows = []
    for p in g['points']:
        t = by.get((p['i'], p['j']))
        r = {G.short(r1): G.fmt_value(r1, p['v1'])}
        if r2:
            r[G.short(r2)] = G.fmt_value(r2, p['v2'])
        if t is not None:
            f = frame(t)
            r.update({COLS[k]: f.loc['AVG', k] for k in ('TECA', 'F1', 'ACC', 'R2', 'RMSE', 'MAE')})
        r['Time [s]'] = p['time']
        r['Status'] = p['error'] or 'ok'
        rows.append(r)
    with st.expander(f"All grid points ({sum(1 for p in g['points'] if not p['error'])} of {len(g['points'])} "
                     f"finished)", expanded=any(p['error'] for p in g['points'])):
        table(pd.DataFrame(rows), hide_index=True)


RENDER = {0: simulation, 1: hyper, 2: grid}
