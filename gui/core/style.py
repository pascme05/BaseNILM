#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         style.py
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
This function defines the common style of the GUI charts (plotly template, categorical/sequential/diverging colours)
and small chart helpers. Categorical colours are assigned in fixed order and follow the entity (e.g. a component keeps
its colour on every chart).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

#######################################################################################################################
# Colours
#######################################################################################################################
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SEQ = [[0.0, '#cde2fb'], [0.25, '#86b6ef'], [0.5, '#3987e5'], [0.75, '#1c5cab'], [1.0, '#0d366b']]
DIV = [[0.0, '#1c5cab'], [0.25, '#86b6ef'], [0.5, '#f0efec'], [0.75, '#ec8a85'], [1.0, '#b32d2d']]
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
INK, INK2, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SYS_COLOR = INK2                                                       # the system total is drawn neutral


def color(i):
    """Categorical colour of slot i (folded onto the eight slots, callers keep series counts small)."""
    return SERIES[i % len(SERIES)]


def entity_colors(names):
    """Fixed colour per entity name; 'Sys' is drawn neutral."""
    out, k = {}, 0
    for n in names:
        if str(n) == 'Sys':
            out[n] = SYS_COLOR
        else:
            out[n] = color(k)
            k += 1
    return out


#######################################################################################################################
# Template
#######################################################################################################################
_axis = dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zerolinecolor=AXIS, ticks='outside', tickcolor=AXIS,
             exponentformat='power', separatethousands=True,              # 10⁻⁴ instead of SI suffixes like 100µ
             tickfont=dict(color=INK2, size=12), title=dict(font=dict(color=INK2, size=13)), automargin=True)
pio.templates['gui'] = go.layout.Template(layout=dict(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=INK, size=13),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor=SURFACE, colorway=SERIES,
    xaxis=_axis, yaxis=_axis, margin=dict(l=10, r=10, t=84, b=10),
    # title in the first row of the top margin, legend in the second row (they never overlap)
    legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0, font=dict(color=INK2)),
    hoverlabel=dict(bgcolor='white', bordercolor=AXIS, font=dict(color=INK)),
    title=dict(font=dict(size=15, color=INK), x=0, xanchor='left', xref='paper', y=1, yref='container',
               yanchor='top', pad=dict(t=10)),
    bargap=0.25, colorscale=dict(sequential=SEQ, diverging=DIV)))
pio.templates.default = 'gui'


#######################################################################################################################
# Helpers
#######################################################################################################################
def show(fig, height=380, key=None, legend_below=False, info=None):
    """
    Renders a figure with the GUI template and a consistent height (legend below for narrow columns). An
    explanation of the chart (argument info or gui/core/chart_help.py for the calling view) is shown on hover over
    the info line above the chart.
    """
    if info is None:
        from gui.core.chart_help import lookup
        info = lookup(sys._getframe(1).f_code.co_name, key)
    if info:
        st.caption(":material/info: How to read this chart", help=info)
    fig.update_layout(height=height)
    if legend_below or any(a.yanchor == 'bottom' and (a.y or 0) >= 0.9 for a in fig.layout.annotations or []):
        fig.update_layout(legend=dict(y=-0.3, yanchor='top'), margin=dict(b=100))      # subplot titles on top
    st.plotly_chart(fig, theme=None, key=key, config={'displaylogo': False, 'toImageButtonOptions': {'scale': 2}})


def line(x, y, name, i=0, dash=None, width=2, **kw):
    return go.Scatter(x=x, y=y, name=name, mode=kw.pop('mode', 'lines'),
                      line=dict(color=kw.pop('c', color(i)), width=width, dash=dash), **kw)


def stem(x, y, name, i=0, hover='%{x:.2f}<br>%{y:.4g}', base=0.0):
    """Stem plot (vertical line from base with a dot on top) as two traces; base > 0 for log axes."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    xs = np.column_stack([x, x, np.full_like(x, np.nan)]).ravel()
    ys = np.column_stack([np.full_like(y, base), y, np.full_like(y, np.nan)]).ravel()
    c = color(i)
    return [go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=c, width=1.2), hoverinfo='skip',
                       showlegend=False, connectgaps=False),
            go.Scatter(x=x, y=y, mode='markers', name=name, marker=dict(color=c, size=6),
                       hovertemplate=hover + '<extra></extra>')]


def band(x, lo, hi, name, c, opacity=0.18, showlegend=True):
    """Filled confidence band between lo and hi (two traces)."""
    rgba = _rgba(c, opacity)
    return [go.Scatter(x=x, y=hi, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False),
            go.Scatter(x=x, y=lo, mode='lines', line=dict(width=0), fill='tonexty', fillcolor=rgba, name=name,
                       hoverinfo='skip', showlegend=showlegend)]


def _rgba(hex_color, a):
    h = hex_color.lstrip('#')
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def downsample(n, max_points=4000):
    """Stride that keeps long time series responsive in the browser."""
    return max(1, int(n // max_points))


SCI_HI, SCI_LO = 1e6, 1e-3                                  # outside this range numbers are shown scientifically


def fmt(x, unit='', digits=3):
    """
    Number for tiles, captions and labels: `digits` significant digits, thousands separators, no exponent between
    1e-3 and 1e6 (1,192 · 0.0959 · 146), compact scientific notation outside (4.36e-8, 2.5e6).
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return '–'
    if not np.isfinite(x):
        return '–' if x != x else ('∞' if x > 0 else '−∞')
    if x == 0:
        s = '0'
    elif abs(x) >= SCI_HI or abs(x) < SCI_LO:
        m, e = f"{x:.{max(digits - 1, 0)}e}".split('e')
        m = m.rstrip('0').rstrip('.') if '.' in m else m
        s = f"{m}e{int(e)}"
    else:
        d = max(0, digits - 1 - int(np.floor(np.log10(abs(x)))))
        s = f"{round(x, d):,.{d}f}"
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
    return f"{s} {unit}".strip()


GENERIC = {None, '%g', '%.2g', '%.3g', '%.4g', '%.5g', '%.6g'}


def num_format(values, digits=4):
    """printf format for a numeric table column: integers, fixed decimals or scientific (see fmt)."""
    v = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    a = np.abs(v)
    amax = a.max()
    nz = a[a > 0]
    amin = nz.min() if nz.size else 0.0
    if np.all(v == np.round(v)) and amax < SCI_HI:                    # exact integers only
        return '%d'
    if amax >= SCI_HI or (amin and amin < SCI_LO) or (amin and amax / amin > 1e5):
        return f"%.{digits - 2}e"
    d = int(np.clip(digits - 1 - np.floor(np.log10(amax)), 0, digits))
    return f"%.{d}f"


def table(df, column_config=None, **kw):
    """st.dataframe with an automatic number format per numeric column (explicit formats are kept)."""
    cfg = dict(column_config or {})
    if isinstance(df, pd.DataFrame):
        for c in df.columns:
            col = df[c]
            if not (pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col)):
                continue
            entry = cfg.get(c)
            if isinstance(entry, dict):
                tc = entry.get('type_config') or {}
                if tc.get('type') not in (None, 'number') or tc.get('format') not in GENERIC:
                    continue
                f = num_format(col)
                if f:
                    cfg[c] = {**entry, 'type_config': {**tc, 'type': 'number', 'format': f}}
            elif entry is None:
                f = num_format(col)
                if f:
                    cfg[c] = st.column_config.NumberColumn(format=f)
    return st.dataframe(df, column_config=cfg, **kw)


def status_badge(text, level):
    """Status shown as icon + label (colour never carries the meaning alone)."""
    icon = {'good': ':material/check_circle:', 'warning': ':material/warning:', 'serious': ':material/error:',
            'critical': ':material/cancel:'}[level]
    col = {'good': 'green', 'warning': 'orange', 'serious': 'orange', 'critical': 'red'}[level]
    st.markdown(f":{col}-badge[{icon} {text}]")
