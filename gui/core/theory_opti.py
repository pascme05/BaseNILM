#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         theory_opti.py
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
Theory of the hyperparameter optimisation (mode 1: Keras-Tuner Hyperband for TF, GridSearchCV for SK) and of the grid
optimisation (mode 2) with interactive toy examples. Search spaces and schedules are those of the toolkit:
src/model/models.py (tfMdloptiRCNN, tfMdloptiC), src/model/trainMdlTFopti.py (Hyperband, factor 3, objective
val_loss), src/model/trainMdlSKopti.py (parameter grids, 5-fold cross-validation). The Hyperband schedule uses the
formulas of keras_tuner HyperbandOracle (_get_num_brackets, _get_size, _get_epochs).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui.core.style import show, line, fmt, SERIES, SEQ, MUTED

FACTOR = 3                                                            # Hyperband factor of trainMdlTFopti.py
CV = 5                                                                # GridSearchCV default folds

# Search spaces of the toolkit: (name, values, number of values)
TF_SPACE = {
    0: ("tfMdloptiRCNN (regression, method 0) — loss MAE",
        [("cnn_layers", "1 … 5", 5), ("filters_i (per CNN layer)", "8 … 64, step 8", 8),
         ("kernel_size_0i (per CNN layer)", "2 … 10, step 2", 5), ("dropout_opt / dropout", "off, 0.1 … 0.5", 6),
         ("batch_opt (batch normalisation)", "off / on", 2), ("pooling_opt / pool_size / strides",
                                                               "off, 2 … 10 × 2 … 10", 26),
         ("dnn_layers", "1 … 4", 4), ("units_i (per dense layer)", "64 … 512, step 64", 8),
         ("learning_rate", "1e-2, 1e-3, 1e-4", 3)]),
    1: ("tfMdloptiC (classification, method 1) — loss binary cross-entropy",
        [("units (all dense layers)", "64 … 512, step 64", 8), ("dnn_layers", "2 … 6", 5),
         ("learning_rate", "1e-1, 5e-2, 1e-2", 3)])}
SK_GRID = {'RF': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'n_estimators': [16, 32, 64]},
           'KNN': {'n_neighbors': list(range(1, 11)), 'weights': ['uniform', 'distance']},
           'SVM': {'kernel': ['linear', 'rbf', 'poly'], 'C': [1, 10, 100], 'gamma': [0.01, 0.1, 1],
                   'epsilon (regression only)': [0.01, 0.1, 0.5]}}


def _intro(text):
    st.caption(text + " Values start from the current configuration; changes here do not modify it.")


#######################################################################################################################
# Toy validation loss (two hyperparameters scaled to 0 … 1)
#######################################################################################################################
A0, B0 = 0.8, 0.35                                                   # optimum of the toy loss


def _loss(a, b, w_b=0.1):
    """Validation loss of a toy model: a is important, b matters little (w_b)."""
    return 0.2 + 4.0 * (a - A0) ** 2 + w_b * (b - B0) ** 2


def _valley(a, b, inter):
    """Toy loss with a narrow diagonal valley: inter = 0 axis-parallel, 1 rotated by 45° (strong interaction)."""
    phi = inter * np.pi / 4
    da, db = a - 0.68, b - 0.35
    u, v = da * np.cos(phi) + db * np.sin(phi), -da * np.sin(phi) + db * np.cos(phi)
    return 0.2 + 6.0 * u ** 2 + 0.1 * v ** 2


def _contour(w_b=0.1, n=80):
    g = np.linspace(0, 1, n)
    A_, B_ = np.meshgrid(g, g)
    return go.Contour(x=g, y=g, z=_loss(A_, B_, w_b), colorscale=SEQ, reversescale=True, ncontours=18,
                      colorbar=dict(title='Loss', thickness=12), hovertemplate="a %{x:.2f}<br>b %{y:.2f}<br>"
                      "loss %{z:.3f}<extra></extra>", line=dict(width=0.5))


#######################################################################################################################
# Mode 1: hyperparameter optimisation
#######################################################################################################################
def _search_explorer():
    """Grid vs random search with the same budget (Bergstra & Bengio 2012)."""
    c1, c2, c3 = st.columns(3)
    with c1:
        n = st.slider("Budget (trained models)", 4, 49, 16, key='th1|n')
    with c2:
        w_b = st.slider("Influence of parameter b", 0.0, 1.5, 0.1, 0.05, key='th1|wb',
                        help="0: b has no effect on the loss, 1.5: as important as a.")
    with c3:
        seed = st.number_input("Random seed", 0, 999, 1, key='th1|seed')
    k = max(2, int(math.isqrt(n)))
    g = (np.arange(k) + 0.5) / k
    ga, gb = [a.ravel() for a in np.meshgrid(g, g)]
    rng = np.random.default_rng(int(seed))
    ra, rb = rng.random(n), rng.random(n)
    lg, lr = _loss(ga, gb, w_b), _loss(ra, rb, w_b)
    fig = go.Figure([_contour(w_b=w_b),
                     go.Scatter(x=ga, y=gb, mode='markers', name=f"grid {k} × {k}", marker=dict(
                         color=SERIES[1], size=9, symbol='square', line=dict(color='white', width=1))),
                     go.Scatter(x=ra, y=rb, mode='markers', name=f"random ({n})", marker=dict(
                         color=SERIES[2], size=9, line=dict(color='white', width=1))),
                     go.Scatter(x=[A0], y=[B0], mode='markers', name='optimum', marker=dict(
                         color='black', size=12, symbol='x'))])
    fig.update_layout(xaxis_title="Hyperparameter a (important, e.g. learning rate)",
                      yaxis_title="Hyperparameter b (e.g. dropout)", xaxis_range=[0, 1], yaxis_range=[0, 1])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best loss grid", fmt(lg.min()))
    k2.metric("Best loss random", fmt(lr.min()), delta=fmt(lr.min() - lg.min()), delta_color='inverse')
    k3.metric("Values of a tried (grid)", f"{k}")
    k4.metric("Values of a tried (random)", f"{n}")
    show(fig, height=430, key='th1|search', legend_below=True,
         info="Contour: validation loss of a toy model over two hyperparameters. A grid with the same budget tries "
              "only √N different values of each parameter; random search tries N different values of each. When one "
              "parameter matters much more than the other (small influence of b), random search usually finds a "
              "lower loss. Hyperband (TF) starts from random configurations; GridSearchCV (SK) uses a grid.")


def _hyperband_schedule(max_epochs, factor, min_epochs=1):
    """Brackets of keras_tuner HyperbandOracle: [(bracket, round, trials, epochs)]."""
    nb, e = 0, float(max_epochs)
    while e >= min_epochs:
        e /= factor
        nb += 1
    end0 = math.ceil(1 + math.log(max_epochs, factor)) if max_epochs > 1 else 1
    rows = []
    for b in reversed(range(nb)):
        for r in range(b + 1):
            rows.append((b, r, math.ceil(end0 / (b + 1) * factor ** (b - r)),
                         math.ceil(max_epochs / factor ** (b - r))))
    return rows


def _halving_explorer(max_epochs, factor):
    """Successive halving of the largest bracket on synthetic learning curves."""
    sched = _hyperband_schedule(max_epochs, factor)
    rows = [r_ for r_ in sched if r_[0] == sched[0][0]]              # largest bracket comes first
    n0 = rows[0][2]
    rng = np.random.default_rng(3)
    l_inf = rng.uniform(0.15, 0.6, n0)
    tau = rng.uniform(0.5, 4.0, n0) * max(1, max_epochs / 10)
    tau[np.argmin(l_inf)] = 3.5 * max(1, max_epochs / 10)             # the best model learns slowly
    ep = np.arange(1, max_epochs + 1)
    curves = l_inf[:, None] + (1.0 - l_inf[:, None]) * np.exp(-ep[None, :] / tau[:, None]) + \
        rng.normal(0, 0.01, (n0, max_epochs))
    alive, stop = np.arange(n0), np.full(n0, max_epochs)
    for (b, r, size, epochs), nxt in zip(rows, rows[1:] + [None]):
        if nxt is None:
            break
        score = curves[alive, epochs - 1]
        keep = alive[np.argsort(score)[:nxt[2]]]
        stop[np.setdiff1d(alive, keep)] = epochs
        alive = keep
    fig = go.Figure()
    for i in range(n0):
        win = stop[i] == max_epochs
        fig.add_trace(line(ep[:stop[i]], curves[i, :stop[i]], f"trial {i + 1}", i % len(SERIES),
                           width=2.2 if win else 1, showlegend=False, opacity=1 if win else 0.45,
                           **({} if win else {'c': MUTED})))
    for b, r, size, epochs in rows[:-1]:
        fig.add_vline(x=epochs, line_dash='dot', line_color=MUTED,
                      annotation_text=f"keep {math.ceil(size / factor)} of {size}", annotation_position='top')
    best_true = int(np.argmin(l_inf))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Validation loss (toy)", xaxis_range=[0.5, max_epochs + 0.5])
    c1, c2 = st.columns(2)
    c1.metric("Trials started in this bracket", n0)
    c2.metric("Best final model survived", "yes" if stop[best_true] == max_epochs else "no — stopped early",
              help="The trial with the lowest final loss learns slowly on purpose: successive halving judges trials "
                   "by their loss after few epochs and can drop such late bloomers. The other brackets start with "
                   "fewer trials and more epochs to hedge against this.")
    show(fig, height=380, key='th1|halving',
         info="Synthetic learning curves of the first (most aggressive) bracket. At every dotted line only the best "
              "1/factor of the trials (lowest validation loss at that epoch) is trained further; grey curves are "
              "stopped. Keras-Tuner continues the kept models from their checkpoint.")


def _hyper(cfg):
    _intro("Hyperparameters (layers, units, learning rate, tree depth …) are fixed before training; the optimiser "
           "searches the values with the lowest validation loss.")
    P, M = cfg['Par'], cfg['Mdl']
    t1, t2, t3, t4 = st.tabs([":material/function: Problem", ":material/scatter_plot: Search strategies",
                              ":material/filter_alt: Hyperband (TF)", ":material/grid_on: Search space"])
    with t1:
        st.latex(r"\theta^*(\lambda) = \arg\min_\theta \; \mathcal{L}_{train}(\theta;\, \lambda)")
        st.latex(r"\lambda^* = \arg\min_{\lambda \in \Lambda} \; \mathcal{L}_{val}\big(\theta^*(\lambda)\big)")
        st.markdown(
            "Two nested problems: training finds the weights $\\theta$ for given hyperparameters $\\lambda$, the "
            "optimisation searches $\\lambda$ in the search space $\\Lambda$. The outer loss must use data the "
            "weights were not trained on — otherwise larger models always win (overfitting):\n\n"
            "* **TF (Keras-Tuner Hyperband)**: objective = `val_loss` on the validation data of the configuration "
            "(*Validation split*); afterwards the best model is trained again for *Epochs* and the best epoch "
            "(minimum `val_loss`) is reported.\n"
            "* **SK (GridSearchCV)**: objective = mean score of a 5-fold cross-validation on the training data "
            "($R^2$ for regression, accuracy for classification); the best estimator is refitted on all training "
            "data.\n\n"
            "The test data are not used — the test score of the final model is therefore an unbiased estimate. "
            "Choosing hyperparameters by the test score would make it optimistic.")
        st.markdown("**Cost**: every evaluated configuration is a complete training. Budget the search with a "
                    "small *Limit* and few *Epochs* first (the *Quick check* uses 1 epoch), then refine.")
    with t2:
        st.markdown("| Strategy | Idea | In BaseNILM |\n|---|---|---|\n"
                    "| Grid search | every combination of given values | GridSearchCV (SK), grid mode 2 |\n"
                    "| Random search | N random combinations | first round of every Hyperband bracket |\n"
                    "| Successive halving / Hyperband | many random trials with few epochs, keep the best | "
                    "Keras-Tuner (TF) |\n"
                    "| Bayesian optimisation | model of the loss surface proposes the next trial | not used |")
        _search_explorer()
    with t3:
        st.markdown("Hyperband (Li et al. 2018) runs several **brackets** of *successive halving*. A bracket starts "
                    "$n$ random configurations with $r$ epochs, keeps the best $n/\\eta$, trains them $\\eta$ times "
                    "longer and repeats up to the maximum epochs $R$. Brackets differ in how aggressively they stop "
                    "trials; each bracket uses about the same number of epochs.")
        st.latex(r"s_{max} = \lfloor \log_\eta R \rfloor, \qquad n_{b,r} = \left\lceil \frac{\lceil 1 + "
                 r"\log_\eta R \rceil}{b+1}\, \eta^{\,b-r} \right\rceil, \qquad r_{b,r} = \left\lceil "
                 r"\frac{R}{\eta^{\,b-r}} \right\rceil")
        c1, c2 = st.columns(2)
        with c1:
            R = st.slider("Maximum epochs R (= Epochs of the configuration)", 1, 100,
                          int(min(100, max(1, M.get('epoch') or 10))), key='th1|R')
        with c2:
            eta = st.slider("Factor η (toolkit: 3)", 2, 5, FACTOR, key='th1|eta')
        rows = _hyperband_schedule(R, eta)
        import pandas as pd
        df = pd.DataFrame(rows, columns=['Bracket', 'Round', 'Trials', 'Epochs per trial'])
        df['Epochs of the round'] = df['Trials'] * df['Epochs per trial']
        trials = int(df[df['Round'] == 0]['Trials'].sum())
        k1, k2, k3 = st.columns(3)
        k1.metric("Brackets", int(df['Bracket'].nunique()))
        k2.metric("Configurations tried", trials)
        k3.metric("Epochs (upper bound)", int(df['Epochs of the round'].sum()),
                  help=f"Kept trials continue from their checkpoint, so fewer epochs are trained. Training all "
                       f"{trials} configurations for {R} epochs would take {trials * R} epochs.")
        st.dataframe(df, hide_index=True, width='stretch')
        st.caption("Schedule of one Hyperband iteration as computed by keras_tuner's HyperbandOracle. The toolkit "
                   "uses factor 3, objective val_loss and max_epochs = Epochs; EarlyStopping (Patience) can end "
                   "trials earlier.")
        if R >= 3:
            _halving_explorer(R, eta)
    with t4:
        if P.get('solver') == 'SK':
            mdl = P.get('model', 'RF')
            grid = SK_GRID.get(mdl)
            if grid is None:
                st.info(f"GridSearchCV has no parameter grid for the model {mdl} (RF, KNN, SVM).")
            else:
                grid = {k: v for k, v in grid.items() if not (k.startswith('epsilon') and P.get('method') == 1)}
                n = int(np.prod([len(v) for v in grid.values()]))
                st.markdown(f"**GridSearchCV — {mdl}** (`trainMdlSKopti.py`)")
                st.markdown("| Parameter | Values |\n|---|---|\n" +
                            "\n".join(f"| `{k}` | {', '.join(map(str, v))} |" for k, v in grid.items()))
                k1, k2 = st.columns(2)
                k1.metric("Combinations", n)
                k2.metric("Fits (5-fold CV + refit)", n * CV + 1)
                st.caption("Multi-output models get the prefix `estimator__` (parameters of the inner estimator). "
                           "*Apply to the configuration* in the results copies the best values to the SK "
                           "parameters (Model tab).")
        else:
            title, space = TF_SPACE[1 if P.get('method') == 1 else 0]
            st.markdown(f"**Keras-Tuner — {title}** (`models.py`)")
            st.markdown("| Hyperparameter | Range | Values |\n|---|---|---|\n" +
                        "\n".join(f"| `{n_}` | {r_} | {c_} |" for n_, r_, c_ in space))
            st.caption("The tuned network is the hypermodel of the method (regression → CNN + dense, classification "
                       "→ dense), whatever model is selected in *Model*. Per-layer parameters exist once per layer, "
                       "so the space has millions of combinations — a grid is impossible, which is why random "
                       "trials with early stopping (Hyperband) are used.")


#######################################################################################################################
# Mode 2: grid optimisation
#######################################################################################################################
def _grid_values(cfg):
    from gui import adapter as A
    G = {**A.GRID_DEFAULT, **(cfg.get('Grid') or {})}
    return G


def _window_fs(cfg):
    """Physical window length W / fs over the grid of window × sampling frequency."""
    from gui import adapter as A
    from gui.core import grid as gr
    G = _grid_values(cfg)
    refs = {G.get('var1'): G.get('values1'), G.get('var2'): G.get('values2')}
    Ws = refs.get('Par.window') or A.GRID_PRESETS['Par.window']
    fss = refs.get('Dat.fs') or A.GRID_PRESETS['Dat.fs']
    Ws = sorted(float(w) for w in Ws)
    fss = sorted((float(f) for f in fss), reverse=True)
    dur = np.array([[w / f / 60 for w in Ws] for f in fss])           # minutes
    ylab = [gr.fmt_value('Dat.fs', f) for f in fss]
    xlab = [f"{w:g}" for w in Ws]
    txt = [[(f"{d / 60:.3g} h" if d >= 120 else f"{d:.3g} min") for d in row] for row in dur]
    fig = go.Figure(go.Heatmap(z=np.log10(dur), x=xlab, y=ylab, text=txt, texttemplate='%{text}', colorscale=SEQ,
                               colorbar=dict(title='log10 min', thickness=12),
                               hovertemplate="W %{x}<br>fs %{y}<br>window %{text}<extra></extra>"))
    fig.update_layout(xaxis_title="Window length W [samples]", yaxis_title="Sampling frequency fs",
                      xaxis_type='category', yaxis_type='category')
    show(fig, height=380, key='th2|wfs',
         info="Time span of one input window, W / fs. Cells on a diagonal see the same time span with a different "
              "resolution. Appliances are recognised by their cycles: a window shorter than one cycle (fridge "
              "≈ 30–60 min, washing machine ≈ 1–2 h) sees only a part of it; a long window at low fs sees whole "
              "cycles but loses short events (kettle, microwave).")
    st.caption("Frames per day of data (step 1): " +
               ", ".join(f"{86400 * f:,.0f} at {gr.fmt_value('Dat.fs', f)}" for f in fss[:4]) +
               " … — lower fs gives fewer training samples and shorter runs.")


def _oat_explorer():
    """Full grid vs one-at-a-time search on a toy score surface with interaction and training noise."""
    c1, c2, c3 = st.columns(3)
    with c1:
        inter = st.slider("Interaction of the two parameters", 0.0, 1.0, 0.6, 0.05, key='th2|inter',
                          help="0: the best value of one parameter does not depend on the other.")
    with c2:
        noise = st.slider("Run-to-run noise of the score", 0.0, 0.1, 0.02, 0.005, key='th2|noise',
                          help="Random initialisation and shuffling give a different score each training run.")
    with c3:
        k = st.slider("Values per parameter", 3, 9, 5, key='th2|k')
    seed = st.number_input("Seed of the runs", 0, 999, 0, key='th2|seed')
    g = np.linspace(0.05, 0.95, k)
    A_, B_ = np.meshgrid(g, g)
    true = _valley(A_, B_, inter)
    obs = true + np.random.default_rng(int(seed)).normal(0, noise, true.shape)
    jb = int(np.argmin(np.abs(g - 0.5)))                             # baseline of b: mid value
    ia = int(np.argmin(obs[jb, :]))                                  # step 1: vary a at baseline b
    jb2 = int(np.argmin(obs[:, ia]))                                 # step 2: vary b at the best a
    jg, ig = np.unravel_index(np.argmin(obs), obs.shape)
    jt, it = np.unravel_index(np.argmin(true), true.shape)
    fig = go.Figure([go.Heatmap(x=g, y=g, z=obs, colorscale=SEQ, reversescale=True,
                                colorbar=dict(title='Loss', thickness=12),
                                hovertemplate="a %{x:.2f}<br>b %{y:.2f}<br>loss %{z:.3f}<extra></extra>"),
                     go.Scatter(x=[g[0], g[-1], None, g[ia], g[ia]], y=[g[jb], g[jb], None, g[0], g[-1]],
                                mode='lines', name='one at a time', line=dict(color=SERIES[1], width=2, dash='dot')),
                     go.Scatter(x=[g[ia]], y=[g[jb2]], mode='markers', name='best one at a time',
                                marker=dict(color=SERIES[1], size=14, symbol='diamond',
                                            line=dict(color='white', width=1))),
                     go.Scatter(x=[g[ig]], y=[g[jg]], mode='markers', name='best of the grid',
                                marker=dict(color=SERIES[2], size=14, symbol='star', line=dict(color='white',
                                                                                              width=1))),
                     go.Scatter(x=[g[it]], y=[g[jt]], mode='markers', name='true best point',
                                marker=dict(color='black', size=10, symbol='x'))])
    fig.update_layout(xaxis_title="Parameter 1 (scaled)", yaxis_title="Parameter 2 (scaled)")
    k1, k2, k3 = st.columns(3)
    k1.metric("Runs one at a time", 2 * k - 1)
    k2.metric("Runs full grid", k * k)
    k3.metric("True loss: one at a time / grid", f"{true[jb2, ia]:.3f} / {true[jg, ig]:.3f}",
              help=f"Lowest possible loss on this grid: {true.min():.3f}.")
    show(fig, height=430, key='th2|oat', legend_below=True,
         info="Observed loss of every grid point (true surface + run noise). One at a time: vary parameter 1 at a "
              "fixed parameter 2, keep the best, then vary parameter 2. With interaction the best value of one "
              "parameter depends on the other and the one-at-a-time search stops in the wrong place. With noise the "
              "best grid point can differ from the true one — differences smaller than the noise are not "
              "meaningful.")


def _grid(cfg):
    _intro("The grid optimisation runs the complete simulation (main.py) for every combination of the values of one "
           "or two parameters and compares the test scores.")
    from gui.core import grid as gr
    G = _grid_values(cfg)
    n1, n2 = len(G.get('values1') or []), len(G.get('values2') or []) if G.get('var2') else 1
    t1, t2, t3, t4 = st.tabs([":material/function: Problem", ":material/schedule: Window × sampling",
                              ":material/grid_on: Grid vs one at a time", ":material/fact_check: Reading the results"])
    with t1:
        st.latex(r"(v_1^*, v_2^*) = \arg\max_{v_1 \in V_1,\, v_2 \in V_2} \; S\big(\mathrm{main}(v_1, v_2)\big), "
                 r"\qquad N_{runs} = |V_1| \cdot |V_2|")
        st.markdown(
            "A full factorial grid: every combination is one complete run — loading, pre-processing, training and "
            "testing — and is scored with the selected score $S$ (TECA, F1, …, average or one appliance). Unlike "
            "the hyperparameter optimisation (mode 1), any parameter can be swept, including the data pipeline "
            "(sampling frequency, window, normalisation, filters) and the solver itself.")
        st.markdown(f"**Current grid**: {gr.label(G.get('var1'))} × "
                    f"{gr.label(G.get('var2')) if G.get('var2') else '—'} → **{n1 * n2} runs** "
                    f"(*{G.get('values1')}* × *{G.get('values2') if G.get('var2') else '—'}*).")
        st.markdown(
            "* **Coupled framing** (*Couple framing to the window*, as `optiGrid.py`): at every window $W$ the "
            "overlap is $W-1$ (step 1) and the seq2point focus $W/2$, so all points use the same framing rule.\n"
            "* **Cost** grows with the product of the value counts: a third value of each parameter turns 4 runs "
            "into 9. Space the values logarithmically (e.g. 4, 8, 16, 32) when the effect is multiplicative "
            "(window, learning rate, sampling time).\n"
            "* The scores are **test scores**. Picking the best of many test scores is slightly optimistic; for a "
            "fair final number, confirm the chosen point on other data (e.g. k-fold or transfer).")
    with t2:
        _window_fs(cfg)
    with t3:
        _oat_explorer()
    with t4:
        st.markdown(
            "* **Best point at the edge** of the grid → the optimum may lie outside; extend the values in that "
            "direction.\n"
            "* **Flat region** → the parameter has little effect there; choose the cheapest value (small window, "
            "low fs, few epochs).\n"
            "* **Noise**: every point is one training run with random initialisation. Differences of a few tenths of "
            "a percent TECA are usually within the run-to-run noise; repeat the best points or use k-fold for "
            "a firm decision.\n"
            "* **Per appliance**: the best setting differs between appliances (short events need a high fs, long "
            "cycles a long window); the average weights them by their energy.\n"
            "* **Run time per point** shows the cost side — a slightly lower score at a fraction of the time can be "
            "the better choice.\n"
            "* *Apply best values* copies the best point into the configuration (with the coupled framing).")
