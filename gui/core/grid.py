#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         grid
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
Grid optimisation of the GUI: selection of one or two parameters and their values (cfg['Grid']), labels and value
formatting of the grid variables and transfer of the best grid point / hyperparameters into the configuration.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms

import math

import streamlit as st

#######################################################################################################################
# Parameters
#######################################################################################################################
SECTIONS = ('Par', 'Dat', 'Mdl')
EXCLUDE = {'Par.overlap', 'Par.yFocus', 'Dat.lim', 'Dat.fold', 'Mdl.verbose', 'Mdl.SK_RF_state'}
NONE = '— none (1-D sweep) —'


#######################################################################################################################
# Helper
#######################################################################################################################
def _param(ref):
    sec, var = ref.split('.', 1)
    return next((p for p in forms.schema().get(sec, []) if p['var'] == var and not p['nested']), None)


def candidates():
    """'Sec.var' of all scalar parameters that can be swept (numbers and options)."""
    out = []
    for sec in SECTIONS:
        for p in forms.schema().get(sec, []):
            ref = f"{sec}.{p['var']}"
            if p['nested'] or ref in EXCLUDE or p['type'] in ('list', 'bool'):
                continue
            if p['type'] in ('int', 'float') or p['options']:
                out.append(ref)
    return out


def label(ref):
    """'Window length (Par.window)'."""
    if not ref:
        return NONE
    p = _param(ref)
    return f"{p['name'] if p else ref} ({ref})"


def short(ref):
    p = _param(ref)
    return p['name'] if p else str(ref)


def fmt_value(ref, v):
    """Readable value: sampling frequencies as 1/T Hz (T in s / min), options by their label."""
    if v is None:
        return '–'
    if ref == 'Dat.fs' and isinstance(v, (int, float)) and v > 0:
        T = 1 / v
        t = f"{T / 60:g} min" if T >= 60 else f"{T:g} s"
        return f"1/{round(T)} Hz ({t})" if abs(T - round(T)) < 1e-6 else f"{v:.4g} Hz ({t})"
    p = _param(ref)
    if p and p['options']:
        lab = p['options'].get(v, p['options'].get(str(v)))
        if lab is not None:
            return f"{v}: {lab}" if str(v) != str(lab) else str(v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _parse(ref, txt):
    """Values from the text field: numbers (incl. fractions such as 1/60) or option keys."""
    out = []
    for part in [t.strip() for t in txt.replace(';', ',').split(',') if t.strip()]:
        try:
            if '/' in part:
                a, b = part.split('/', 1)
                out.append(float(a) / float(b))
            else:
                f = float(part)
                out.append(int(f) if f.is_integer() and '.' not in part and 'e' not in part.lower() else f)
        except ValueError:
            out.append(part.strip("'\""))
    p = _param(ref)
    if p and p['type'] == 'int':
        out = [int(v) if isinstance(v, float) and v.is_integer() else v for v in out]
    return out


def _text(ref, vals):
    if ref == 'Dat.fs':
        return ', '.join(f"1/{round(1 / v)}" if v and abs(1 / v - round(1 / v)) < 1e-6 else f"{v:g}" for v in vals)
    return ', '.join(f"{v:g}" if isinstance(v, float) else str(v) for v in vals)


def grid(cfg):
    """Grid settings of the configuration (defaults filled in, stored in cfg['Grid'])."""
    cfg['Grid'] = {**A.GRID_DEFAULT, **(cfg.get('Grid') or {})}
    return cfg['Grid']


#######################################################################################################################
# Form (Parameters view of the grid mode)
#######################################################################################################################
def _values(G, n, key):
    ref = G[f'var{n}']
    p = _param(ref)
    if p and p['options']:
        opts = list(p['options'].keys())
        cur = [v for v in G[f'values{n}'] if v in opts] or opts[:2]
        G[f'values{n}'] = st.multiselect(f"Values of {short(ref)}", opts, default=cur, key=f"grid|v{n}|{ref}",
                                         format_func=lambda v: fmt_value(ref, v))
    else:
        txt = st.text_input(f"Values of {short(ref)}", _text(ref, G[f'values{n}']), key=f"grid|v{n}|{ref}",
                            help="Comma-separated values; fractions such as 1/60 are allowed.")
        G[f'values{n}'] = _parse(ref, txt)
    if G[f'values{n}']:
        st.caption(' · '.join(fmt_value(ref, v) for v in G[f'values{n}']))


def _var(G, n, key, refs):
    cur = G.get(f'var{n}')
    opts = ([None] if n == 2 else []) + refs
    idx = opts.index(cur) if cur in opts else 0
    new = st.selectbox(f"Variable {n}", opts, index=idx, format_func=label, key=f"grid|var{n}")
    if new != cur:                                                     # new variable: preset values
        G[f'var{n}'] = new
        p = _param(new) if new else None
        val = (p or {}).get('default')
        G[f'values{n}'] = A.GRID_PRESETS.get(new) or (list((p or {}).get('options') or {})[:3] if p and
                                                       p['options'] else [val] if val is not None else [])
        for k in [k for k in st.session_state if str(k).startswith(f"grid|v{n}|")]:
            del st.session_state[k]
    return new


def form(cfg):
    """Selection of the grid variables and values."""
    G = grid(cfg)
    refs = candidates()
    st.caption("The simulation (training + testing as set under *Experiment*) is repeated for every combination of "
               "the values. Each grid point trains its own model mdl/mdl_<model>_<name>_G<i>_<j>.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            _var(G, 1, 'v1', refs)
            _values(G, 1, 'v1')
    with c2:
        with st.container(border=True):
            if _var(G, 2, 'v2', [r for r in refs if r != G['var1']]):
                _values(G, 2, 'v2')
            else:
                G['values2'] = []
                st.caption("Only variable 1 is swept (line chart of the scores).")
    c1, c2 = st.columns(2)
    with c1:
        G['couple'] = st.toggle("Couple framing to the window", G.get('couple', True), key="grid|couple",
                                help="As optiGrid.py: overlap = window − 1 and focus point = window / 2 at every grid "
                                     "point, so seq2point stays centred when the window changes.")
    with c2:
        G['keep_models'] = st.toggle("Keep the models of the grid points", G.get('keep_models', False),
                                     key="grid|keep", help="Off: mdl/mdl_<model>_<name>_G<i>_<j> are deleted after "
                                                           "each grid point (they are only needed to re-test them).")
    n = max(len(G['values1']), 1) * (max(len(G['values2']), 1) if G.get('var2') else 1)
    st.info(f"{n} grid points = {n} complete simulations"
            + (" (each with training)" if cfg['Exp'].get('train') == 1 else
               " — Training is off: each grid point needs its own trained model, switch on Training under "
               "*Experiment*.") + " Use *Quick check* in the Run view (1 epoch, 2 × 2 points) for a first test.")


#######################################################################################################################
# Apply to the configuration
#######################################################################################################################
def apply(values, toast=None):
    """Writes {'Sec.var': value} into the session configuration and re-initialises the widgets."""
    cfg = forms.cfg()
    for ref, v in values.items():
        sec, var = ref.split('.', 1)
        if isinstance(v, float) and math.isfinite(v) and v.is_integer() and (sec, var) in A.INTS:
            v = int(v)
        cfg[sec][var] = v
    if values.get('Par.window') is not None and (cfg.get('Grid') or {}).get('couple', True):
        d = {s: cfg[s] for s in ('Exp', 'Dat', 'Par', 'Mdl')}
        A.couple_window(d)
    forms.set_config(cfg, st.session_state.get('cfg_name', 'config'))
    st.toast(toast or "Values applied to the configuration", icon=':material/check:')
