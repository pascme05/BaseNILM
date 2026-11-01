#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         theory_methods
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
Help › Theory: the methods behind the models of BaseNILM with small interactive examples — neural networks (neuron,
activation functions, gradient descent, dense layers, convolution, LSTM, attention), classic machine learning (random
forest, KNN, SVM) and pattern matching / source separation (DTW, MVM, GAK, sDTW, NMF, DSC). The examples use toy
data and numpy / scikit-learn; widgets have the keys 'tm|…' and do not change the configuration.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core.style import show, line, fmt, color, SERIES, MUTED

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ACT = {'linear': lambda z: z, 'ReLU': lambda z: np.maximum(z, 0), 'sigmoid': lambda z: 1 / (1 + np.exp(-z)),
       'tanh': np.tanh}


def _where(text):
    st.caption(":material/link: " + text)


#######################################################################################################################
# Neural networks
#######################################################################################################################
def neuron():
    st.markdown("A neuron weights its inputs, adds a bias and passes the sum through an activation function. A dense "
                "layer is many neurons that see the same inputs; a network stacks layers.")
    st.latex(r"z = \sum_{j=1}^{3} w_j x_j + b, \qquad y = \sigma(z)")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        x = [st.slider(f"Input x{j + 1}", -2.0, 2.0, v, 0.1, key=f"tm|x{j}") for j, v in enumerate((1.0, 0.5, -1.0))]
    with c2:
        w = [st.slider(f"Weight w{j + 1}", -2.0, 2.0, v, 0.1, key=f"tm|w{j}") for j, v in enumerate((0.8, -0.5, 0.3))]
        b = st.slider("Bias b", -2.0, 2.0, 0.1, 0.1, key="tm|b")
        act = st.selectbox("Activation σ", list(ACT), index=1, key="tm|act")
    z = float(np.dot(w, x) + b)
    y = float(ACT[act](np.array(z)))
    with c3:
        fig = go.Figure()
        for j in range(3):                                             # edges, width ~ |w|, colour ~ sign
            fig.add_trace(go.Scatter(x=[0, 1], y=[2 - j, 1], mode='lines', hoverinfo='skip', showlegend=False,
                                     line=dict(width=1 + 4 * abs(w[j]), color=SERIES[0] if w[j] >= 0 else SERIES[7])))
            fig.add_annotation(x=0.45, y=1.55 - j * 0.5 + 0.05, text=f"w={w[j]:+.1f}", showarrow=False,
                               font=dict(size=11))
        fig.add_trace(go.Scatter(x=[0, 0, 0], y=[2, 1, 0], mode='markers+text', text=[f"x{j + 1}={x[j]:+.1f}" for j in
                                 range(3)], textposition='middle left', marker=dict(size=26, color=MUTED),
                                 showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[1], y=[1], mode='markers+text', text=[f"Σ + b → {act}"], textposition='top center',
                                 marker=dict(size=46, color=SERIES[2]), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[1, 1.6], y=[1, 1], mode='lines+text', text=['', f"y = {y:.3f}"],
                                 textposition='middle right', line=dict(color=SERIES[2], width=3), showlegend=False,
                                 hoverinfo='skip'))
        fig.update_xaxes(visible=False, range=[-0.6, 2.1])
        fig.update_yaxes(visible=False, range=[-0.5, 2.5])
        show(fig, height=300, key="tm|neuron", info="Line width = |weight|, blue = positive, red = negative weight.")
    st.success(f"z = {' + '.join(f'{w[j]:+.1f}·{x[j]:+.1f}' for j in range(3))} {b:+.1f} = **{z:.3f}** → "
               f"{act}(z) = **{y:.3f}**")
    _where("Dense layers: DNN (tfMdlDNN) and the output layer of every BaseNILM network; seq2point networks end in "
           "one linear neuron per appliance (regression).")


def activations():
    z = np.linspace(-5, 5, 401)
    fig = go.Figure([line(z, f(z), k, i) for i, (k, f) in enumerate(ACT.items())])
    fig.update_layout(xaxis_title="z", yaxis_title="σ(z)", yaxis_range=[-1.5, 3])
    show(fig, height=320, key="tm|actfig", info="ReLU (hidden layers of BaseNILM) is cheap and does not saturate for "
         "positive z; sigmoid squeezes to 0…1 (output for classification, setupPar['method'] = 1); linear keeps the "
         "power values (regression output).")


def gradient_descent():
    st.markdown("Training minimises the loss $L(\\theta)$ by stepping against its gradient. The learning rate "
                "$\\eta$ (Model › Learning rate) sets the step size:")
    st.latex(r"\theta_{k+1} = \theta_k - \eta \, \nabla_\theta L(\theta_k)")
    c1, c2 = st.columns([1, 3])
    with c1:
        lr = st.select_slider("Learning rate η", [0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.05], 0.1, key="tm|lr")
        steps = st.slider("Steps (epochs)", 1, 40, 15, key="tm|steps")
        th0 = st.slider("Start θ₀", -3.0, 3.0, -2.5, 0.1, key="tm|th0")
    L = lambda t: (t - 1) ** 2 + 0.3 * np.sin(3 * t)                  # noqa: E731
    dL = lambda t: 2 * (t - 1) + 0.9 * np.cos(3 * t)                  # noqa: E731
    th = [th0]
    for _ in range(steps):
        th.append(float(np.clip(th[-1] - lr * dL(th[-1]), -6, 6)))
    th = np.array(th)
    with c2:
        t = np.linspace(-3.5, 4, 400)
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], subplot_titles=("Loss surface", "Loss per step"))
        fig.add_trace(line(t, L(t), 'L(θ)', c=MUTED, width=1.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=th, y=L(th), mode='lines+markers', name='steps', marker=dict(size=7,
                                 color=np.arange(len(th)), colorscale='Blues'), line=dict(color=SERIES[0], width=1)),
                      row=1, col=1)
        fig.add_trace(line(np.arange(len(th)), L(th), 'loss', 1, mode='lines+markers'), row=1, col=2)
        fig.update_xaxes(title_text="θ", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=1, col=2)
        show(fig, height=320, key="tm|gd", info="Too small: slow progress; too large (≥ 1 here): the steps overshoot "
             "and oscillate. Adam (default optimiser) adapts the step per parameter; ReduceLROnPlateau halves the "
             "learning rate when the validation loss stalls (trainMdlTF.py).")
    st.caption(f"Final θ = {th[-1]:.3f}, loss = {L(th[-1]):.4f} (minimum near θ ≈ 1.1).")


def mlp():
    st.markdown("Stacked dense layers approximate non-linear functions: every hidden ReLU neuron adds one kink. Here "
                "a network with one hidden layer is fitted (least squares on random ReLU features) to a non-linear "
                "appliance characteristic.")
    n = st.slider("Hidden neurons", 1, 40, 6, key="tm|hid")
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 200)
    y = np.where(x > 0.35, 1.0, 0.0) * (0.6 + 0.4 * np.sin(6 * x)) + 0.1 * x
    W, B = rng.normal(0, 4, n), rng.uniform(-4, 4, n)
    H = np.maximum(np.outer(x, W) + B, 0)
    H1 = np.c_[H, np.ones_like(x)]
    coef = np.linalg.lstsq(H1, y, rcond=None)[0]
    fig = go.Figure([line(x, y, 'target', c=MUTED, width=2), line(x, H1 @ coef, f'{n} ReLU neurons', 0)])
    for j in range(min(n, 8)):
        fig.add_trace(line(x, H[:, j] * coef[j], f'neuron {j + 1}', j + 1, dash='dot', width=1, opacity=0.5,
                           showlegend=False))
    fig.update_layout(xaxis_title="Input (normalised aggregate)", yaxis_title="Output")
    show(fig, height=320, key="tm|mlp", info="Dotted: contribution of single hidden neurons (first 8); solid: their "
         "weighted sum, the network output. More neurons fit finer details — and noise (overfitting).")


def convolution():
    st.markdown("A 1D convolution slides a small kernel along the window and responds where the signal looks like "
                "the kernel — e.g. an edge detector finds switching events. Many kernels (filters) give many feature "
                "maps; pooling keeps the strongest response.")
    st.latex(r"h[t] = \mathrm{ReLU}\Big(\sum_{m=0}^{K-1} k[m]\, x[t+m] + b\Big)")
    c1, c2 = st.columns([1, 3])
    with c1:
        ker = st.selectbox("Kernel", ['edge ON [-1, 1]', 'edge OFF [1, -1]', 'smoothing [1/3]×3', 'peak [-1, 2, -1]'],
                           key="tm|ker")
        pos = st.slider("Kernel position", 0, 57, 20, key="tm|kpos")
        pool = st.slider("Max-pooling size", 1, 8, 1, key="tm|pool")
    k = {'edge ON [-1, 1]': [-1, 1], 'edge OFF [1, -1]': [1, -1], 'smoothing [1/3]×3': [1 / 3] * 3,
         'peak [-1, 2, -1]': [-1, 2, -1]}[ker]
    x = np.zeros(60)
    x[15:35] = 1.0
    x[45:48] = 0.6
    x += np.random.default_rng(2).normal(0, 0.03, 60)
    h = np.maximum(np.correlate(x, k, mode='valid'), 0)
    hp = h if pool == 1 else np.array([h[i:i + pool].max() for i in range(0, len(h), pool)])
    with c2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Input window", f"Feature map (ReLU){' after pooling' if pool > 1 else ''}"))
        fig.add_trace(line(np.arange(60), x, 'input', 0, mode='lines+markers', marker=dict(size=3)), row=1, col=1)
        p = min(pos, 60 - len(k))
        fig.add_vrect(x0=p - 0.5, x1=p + len(k) - 0.5, fillcolor=SERIES[2], opacity=0.25, line_width=0, row=1, col=1)
        xs = np.arange(len(h)) if pool == 1 else np.arange(len(hp)) * pool
        fig.add_trace(go.Bar(x=xs, y=hp, marker_color=[SERIES[2] if (pool == 1 and i == p) else SERIES[1]
                                                       for i in range(len(hp))], name='feature map'), row=2, col=1)
        show(fig, height=380, key="tm|conv", info="The green box is the kernel; its output at this position is the "
             "green bar. Edge kernels fire at the switching instants, the smoothing kernel follows the level.")
    _where("CNN (tfMdlCNN: Conv1D layers on the window × features), CNN2 (Conv2D on 2D features), DAE (convolutional "
           "auto-encoder). Kernel sizes and filters are defined in models.py.")


def lstm():
    st.markdown("An LSTM reads the window step by step and keeps a memory cell $c_t$. Gates (sigmoids between 0 and "
                "1) decide what to forget, what to write and what to output — so it can remember that an appliance "
                "was switched on many steps ago.")
    st.latex(r"f_t = \sigma(W_f[h_{t-1}, x_t] + b_f),\; i_t = \sigma(\dots),\; o_t = \sigma(\dots),\; "
             r"c_t = f_t c_{t-1} + i_t \tanh(W_c[h_{t-1}, x_t] + b_c),\; h_t = o_t \tanh(c_t)")
    c1, c2 = st.columns([1, 3])
    with c1:
        fgt = st.slider("Forget gate bias", -3.0, 3.0, 2.0, 0.5, key="tm|fb",
                        help="High: the cell keeps its memory; low: it forgets quickly.")
        inp = st.slider("Input gate weight", 0.0, 4.0, 2.0, 0.5, key="tm|iw")
    x = np.zeros(50)
    x[10:14] = 1.0
    x[30:31] = 1.0
    c, h, cs, hs, fs_, is_ = 0.0, 0.0, [], [], [], []
    sig = lambda z: 1 / (1 + np.exp(-z))                              # noqa: E731
    for xt in x:                                                       # one-unit LSTM with hand-set weights
        f = sig(fgt - 3 * xt)
        i = sig(inp * 2 * xt - 2)
        o = sig(1.0)
        c = f * c + i * np.tanh(2 * xt)
        h = o * np.tanh(c)
        cs.append(c), hs.append(h), fs_.append(f), is_.append(i)
    with c2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            subplot_titles=("Input x_t (switching events)", "Cell state, output and gates"))
        fig.add_trace(line(np.arange(50), x, 'x', c=MUTED), row=1, col=1)
        for j, (name, v) in enumerate((('cell c', cs), ('output h', hs), ('forget f', fs_), ('input i', is_))):
            fig.add_trace(line(np.arange(50), v, name, j, dash='dot' if j > 1 else None), row=2, col=1)
        fig.update_xaxes(title_text="Time step in the window", row=2, col=1)
        show(fig, height=380, key="tm|lstm", info="With a forget gate near 1 the cell holds the information of the "
             "first event for many steps; lowering its bias lets the memory decay.")
    _where("LSTM (tfMdlLSTM: recurrent layers over the window).")


def attention():
    st.markdown("Self-attention lets every time step of the window look at all others. The weights come from the "
                "similarity of a query with all keys (softmax); the output is the weighted sum of the values.")
    st.latex(r"\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V")
    c1, c2 = st.columns([1, 3])
    with c1:
        q = st.slider("Query time step", 0, 29, 22, key="tm|q")
        temp = st.slider("Sharpness 1/√d", 0.5, 8.0, 3.0, 0.5, key="tm|temp")
    x = np.zeros(30)
    x[5:9] = 1.0
    x[20:24] = 1.0
    x[14] = 0.5
    t = np.arange(30)
    feat = np.c_[x, np.sin(t / 3), np.cos(t / 3)]                    # value + positional encoding
    s = feat @ feat[q] * temp
    a = np.exp(s - s.max())
    a /= a.sum()
    A = np.exp((feat @ feat.T) * temp)
    A /= A.sum(axis=1, keepdims=True)
    with c2:
        fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                            subplot_titles=(f"Attention of step {q}", "Attention matrix (all queries)"))
        fig.add_trace(go.Bar(x=t, y=a, marker_color=SERIES[0], name='weight'), row=1, col=1)
        fig.add_trace(line(t, x * a.max(), 'input (scaled)', c=MUTED, dash='dot'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), row=1, col=2)
        fig.update_yaxes(autorange='reversed', title_text='query', row=1, col=2)
        fig.update_xaxes(title_text='key (time step)', row=1, col=2)
        show(fig, height=340, key="tm|att", info="The query of an ON step attends to the other ON event (similar "
             "value) and to its neighbours (positional encoding). Sharper attention concentrates on fewer steps.")
    _where("TRA (tfMdlTran: transformer encoder blocks with multi-head attention) and INF (Informer).")


#######################################################################################################################
# Classic machine learning
#######################################################################################################################
def _toy(n=160, seed=4):
    """Toy events: (power step, duration) of two appliances."""
    rng = np.random.default_rng(seed)
    a = np.c_[rng.normal(2.0, 0.35, n // 2), rng.normal(1.0, 0.4, n // 2)]       # kettle: high power, short
    b = np.c_[rng.normal(1.2, 0.35, n // 2), rng.normal(2.4, 0.5, n // 2)]       # washer: lower power, long
    X = np.r_[a, b]
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def _boundary(clf, X, y, title, key, info, extra=None):
    xx, yy = np.meshgrid(np.linspace(0, 3.2, 160), np.linspace(-0.5, 4.2, 160))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape) if hasattr(clf, 'predict_proba') \
        else clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig = go.Figure(go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale=[[0, '#dbe9f9'], [1, '#fde3d6']], showscale=False,
                               contours=dict(showlines=False), opacity=0.9))
    for c_, name in ((0, 'kettle'), (1, 'washer')):
        m = y == c_
        fig.add_trace(go.Scatter(x=X[m, 0], y=X[m, 1], mode='markers', name=name,
                                 marker=dict(color=SERIES[0] if c_ == 0 else SERIES[1], size=7,
                                             line=dict(width=0.5, color='white'))))
    if extra:
        extra(fig)
    fig.update_layout(title=title, xaxis_title="Power step [kW]", yaxis_title="Duration [h]")
    show(fig, height=380, key=key, info=info)


def forest():
    from sklearn.ensemble import RandomForestClassifier
    st.markdown("A decision tree splits the feature space with thresholds (power > a, duration < b …); a random "
                "forest averages many trees trained on random subsets — smoother and less overfitted.")
    c1, c2 = st.columns([1, 3])
    with c1:
        depth = st.slider("Max depth (SK_RF_depth)", 1, 12, 3, key="tm|depth")
        trees = st.select_slider("Trees (SK_RF_estimators)", [1, 4, 16, 64], 16, key="tm|trees")
    X, y = _toy()
    clf = RandomForestClassifier(max_depth=depth, n_estimators=trees, random_state=0).fit(X, y)
    with c2:
        _boundary(clf, X, y, f"Random forest: {trees} tree(s), depth {depth}", "tm|rf", "Background: share of trees "
                  "voting 'washer'. One deep tree gives box-shaped, overfitted regions; many trees smooth them.")
    _where("RF (trainMdlSK.py, RandomForestRegressor per appliance for regression).")


def knn():
    from sklearn.neighbors import KNeighborsClassifier
    st.markdown("KNN predicts from the k most similar training samples — no training, but every prediction searches "
                "the training set.")
    c1, c2 = st.columns([1, 3])
    with c1:
        k = st.slider("Neighbours k (SK_KNN_neighbors)", 1, 40, 5, key="tm|k")
        px = st.slider("New event: power [kW]", 0.0, 3.0, 1.6, 0.05, key="tm|px")
        py = st.slider("New event: duration [h]", 0.0, 4.0, 1.7, 0.05, key="tm|py")
    X, y = _toy()
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    d = np.hypot(X[:, 0] - px, X[:, 1] - py)
    nb = np.argsort(d)[:k]

    def extra(fig):
        for i in nb:
            fig.add_trace(go.Scatter(x=[px, X[i, 0]], y=[py, X[i, 1]], mode='lines', line=dict(color=MUTED, width=1),
                                     showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers', name='new event',
                                 marker=dict(symbol='star', size=16, color=SERIES[2])))
    with c2:
        _boundary(clf, X, y, f"KNN, k = {k}: {int(y[nb].sum())} of {k} neighbours are 'washer'", "tm|knn",
                  "Small k follows single points (noisy boundary), large k averages over a wide neighbourhood.", extra)
    _where("KNN (trainMdlSK.py, KNeighborsRegressor; SK_KNN_weights: uniform or distance weighting).")


def svm():
    from sklearn.svm import SVC
    st.markdown("An SVM separates the classes with the widest margin; kernels map the data to a space where a "
                "linear separation works. C trades margin width against training errors.")
    c1, c2 = st.columns([1, 3])
    with c1:
        kern = st.selectbox("Kernel (SK_SVM_kernel)", ['linear', 'rbf', 'poly'], 1, key="tm|kern")
        C = st.select_slider("C (SK_SVM_C)", [0.01, 0.1, 1, 10, 100], 1, key="tm|C")
        gamma = st.select_slider("γ (SK_SVM_gamma)", [0.01, 0.1, 1, 10], 1, key="tm|gamma")
    X, y = _toy(seed=6)
    clf = SVC(kernel=kern, C=C, gamma=gamma).fit(X, y)

    def extra(fig):
        sv = clf.support_vectors_
        fig.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode='markers', name='support vectors',
                                 marker=dict(size=12, color='rgba(0,0,0,0)', line=dict(width=1.5, color=MUTED))))
    with c2:
        _boundary(clf, X, y, f"SVM ({kern}), {len(clf.support_)} support vectors", "tm|svm", "Background: decision "
                  "function (sign = class). Only the circled support vectors define the boundary. Large γ (rbf) "
                  "gives tight islands around the points.", extra)
    _where("SVM (trainMdlSK.py, SVR for regression with ε-insensitive loss, SK_SVM_epsilon).")


#######################################################################################################################
# Pattern matching and source separation
#######################################################################################################################
def _dtw(a, b, band=None):
    """DTW distance (sum of |a - b| along the path) and warping path, optional Sakoe-Chiba band."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        lo, hi = (1, m) if band is None else (max(1, i - band), min(m, i + band))
        for j in range(lo, hi + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j, path = n, m, [(n - 1, m - 1)]
    while i > 1 or j > 1:
        _, i, j = min((D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1))
        path.append((i - 1, j - 1))
    return D[n, m], path[::-1]


def dtw():
    st.markdown("Pattern matching compares the test window with all stored training windows (templates) and takes "
                "the appliance power of the most similar one. The Euclidean distance compares sample by sample; "
                "**DTW** warps the time axis to align shifted or stretched events; **MVM** (minimal variance matching) "
                "may skip up to PM_MVM_steps samples of the template, which ignores outliers; **GAK / soft-DTW** "
                "are smooth versions of DTW (kernel σ, softness γ).")
    c1, c2 = st.columns([1, 3])
    with c1:
        shift = st.slider("Shift of the test event", -8, 8, 4, key="tm|shift")
        stretch = st.slider("Stretch of the test event", 0.6, 1.6, 1.0, 0.1, key="tm|stretch")
        method = st.radio("Distance", ['Euclidean', 'DTW', 'DTW Sakoe-Chiba band 3'], index=1, key="tm|dist")
    t = np.arange(40)
    tpl = np.where((t >= 10) & (t < 22), 1.0, 0.0) + np.where((t >= 13) & (t < 16), 0.8, 0.0)
    ts = (t - 10 - shift) / stretch + 10
    test = np.where((ts >= 10) & (ts < 22), 1.0, 0.0) + np.where((ts >= 13) & (ts < 16), 0.8, 0.0)
    if method == 'Euclidean':
        d, path = float(np.sum(np.abs(test - tpl))), [(i, i) for i in range(40)]
    else:
        d, path = _dtw(test, tpl, band=3 if 'band' in method else None)
    with c2:
        fig = go.Figure([line(t, tpl + 2.5, 'template (training window)', 0), line(t, test, 'test window', 1)])
        for i, j in path[::2]:
            fig.add_trace(go.Scatter(x=[i, j], y=[test[i], tpl[j] + 2.5], mode='lines', hoverinfo='skip',
                                     line=dict(color=MUTED, width=0.7), showlegend=False))
        fig.update_layout(title=f"{method}: distance {d:.2f}", xaxis_title="Sample", yaxis_visible=False)
        show(fig, height=360, key="tm|dtw", info="Grey lines connect the samples that are compared. Euclidean "
             "compares equal time steps (a shifted event looks very different); DTW bends the lines to align the "
             "event; the band (PM_DTW_const = sakoechiba) limits how far the warping may go.")
    _where("DTW, GAK, sDTW, MVM (trainMdlPM.py stores the training windows, testMdlPM.py searches the nearest "
           "template; only the PM_Gen_cDTW share of templates closest in energy is compared, which keeps it fast).")


def nmf():
    from sklearn.decomposition import NMF
    st.markdown("Source separation models the aggregate as a non-negative combination of appliance signatures: "
                "**NMF** learns a basis per appliance from its training data, the test aggregate is then explained by "
                "non-negative weights of all bases (testMdlSS.py, `scipy.optimize.nnls`). **DSC** (discriminative "
                "sparse coding) learns sparse dictionaries per appliance and refines them so that the codes of the "
                "aggregate separate the appliances.")
    st.latex(r"\mathbf{x} \approx \sum_i B_i \mathbf{a}_i, \quad \mathbf{a}_i \ge 0 \;(\text{NMF}),\qquad "
             r"\min_{\mathbf{a}} \|\mathbf{x} - B\mathbf{a}\|^2 + \lambda \|\mathbf{a}\|_1 \;(\text{sparse coding})")
    k = st.slider("Basis vectors per appliance", 1, 4, 2, key="tm|nmfk")
    rng = np.random.default_rng(5)
    T, n = 48, 60
    t = np.arange(T)
    fr = np.array([np.roll(np.where((t % 8) < 3, 1.0, 0.0), rng.integers(8)) * rng.uniform(0.8, 1.2)
                   for _ in range(n)]) * 0.15
    hp = np.array([np.where((t > rng.integers(10, 20)) & (t < rng.integers(28, 40)), 1.0, 0.0) * rng.uniform(1.5, 3)
                   for _ in range(n)])
    Bf = NMF(k, init='nndsvda', max_iter=500, random_state=0).fit(fr).components_
    Bh = NMF(k, init='nndsvda', max_iter=500, random_state=0).fit(hp).components_
    from scipy.optimize import nnls
    x = fr[0] + hp[0]
    B = np.r_[Bf, Bh].T
    a, _ = nnls(B, x)
    ef, eh = Bf.T @ a[:k], Bh.T @ a[k:]
    fig = make_subplots(rows=1, cols=2, column_widths=[0.4, 0.6], subplot_titles=("Learned bases", "Separation"))
    for j in range(k):
        fig.add_trace(line(t, Bf[j], f'fridge basis {j + 1}', 0, width=1.2, dash='dot' if j else None), row=1, col=1)
        fig.add_trace(line(t, Bh[j], f'heat pump basis {j + 1}', 1, width=1.2, dash='dot' if j else None),
                      row=1, col=1)
    fig.add_trace(line(t, x, 'aggregate', c=MUTED, width=2), row=1, col=2)
    fig.add_trace(line(t, fr[0], 'fridge true', 0, width=1, dash='dot'), row=1, col=2)
    fig.add_trace(line(t, ef, 'fridge est.', 0), row=1, col=2)
    fig.add_trace(line(t, hp[0], 'heat pump true', 1, width=1, dash='dot'), row=1, col=2)
    fig.add_trace(line(t, eh, 'heat pump est.', 1), row=1, col=2)
    fig.update_xaxes(title_text="Sample", row=1, col=2)
    show(fig, height=360, key="tm|nmf", legend_below=True, info="Left: the bases NMF learned from 60 training "
         "windows per appliance. Right: one aggregate window explained by non-negative weights of all bases; the "
         "part assigned to the bases of each appliance is its estimate. Few bases cannot represent shifted cycles "
         "(fridge), more bases fit better but can also absorb the other appliance.")
    _where("NMF and DSC (trainMdlSS.py / testMdlSS.py; SS_DSC_n = number of dictionary atoms, SS_Gen_lr = learning "
           "rate of the discriminative refinement).")


#######################################################################################################################
# Page
#######################################################################################################################
SECTIONS = {
    "Neural networks": [("Neuron", neuron), ("Activation functions", activations), ("Gradient descent",
                        gradient_descent), ("Dense network", mlp), ("Convolution", convolution), ("LSTM", lstm),
                        ("Attention", attention)],
    "Machine learning": [("Random forest", forest), ("KNN", knn), ("SVM", svm)],
    "Pattern matching & source separation": [("DTW", dtw), ("NMF / DSC", nmf)],
}


def page():
    st.title("Theory of the methods")
    st.caption("The model families of BaseNILM with small interactive examples on toy data. The sliders do not "
               "change the configuration.")
    sec = st.segmented_control("Family", list(SECTIONS), default=list(SECTIONS)[0], key="tm|sec",
                               label_visibility='collapsed') or list(SECTIONS)[0]
    items = SECTIONS[sec]
    for tab, (_, fn) in zip(st.tabs([t for t, _ in items]), items):
        with tab:
            fn()
