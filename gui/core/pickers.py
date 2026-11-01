#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         pickers.py
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
Custom widgets of the BaseNILM parameters (registered in forms.CUSTOM on import): dataset folder and files from
data/, appliances and input features from the loaded dataset, models filtered by the solver.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import forms, nilm
from gui.core.help import MODELS

import streamlit as st


#######################################################################################################################
# Helper
#######################################################################################################################
def _with(cur, opts):
    """Options incl. the current value(s), so a config with a missing file still shows its value."""
    cur = cur if isinstance(cur, list) else [cur]
    return list(dict.fromkeys([*opts, *[c for c in cur if c not in opts and c not in (None, '')]]))


def _missing(vals, opts):
    miss = [v for v in (vals if isinstance(vals, list) else [vals]) if v not in opts and v not in (None, '')]
    if miss:
        st.caption(f":orange[:material/warning:] not found: {', '.join(map(str, miss))}")


def _labels(c, side):
    """Columns of the training dataset (None if not loaded: big file or error)."""
    D = c['Dat']
    name = nilm.first_train(c)
    if not nilm.exists(c, name):
        return None
    if nilm.is_big(D.get('folder', ''), name, D['type']) and not st.session_state.get(f"big|{name}"):
        return None
    try:
        return nilm.channels(nilm.load_cfg(c)[side])
    except Exception:
        return None


#######################################################################################################################
# Widgets: fn(p, cur, key, label, hlp, c) -> value
#######################################################################################################################
def folder(p, cur, key, label, hlp, c):
    opts = _with(cur, nilm.folders())
    return st.selectbox(label, opts, index=opts.index(cur), key=key, help=hlp,
                        format_func=lambda f: f"data/{f}" if f else "data/ (head folder)")


def train(p, cur, key, label, hlp, c):
    opts = list(nilm.datasets(c['Dat'].get('folder', ''), c['Dat'].get('type')))
    cur = [str(x) for x in (cur or [])]
    val = st.multiselect(label, _with(cur, opts), default=cur, key=key, help=hlp,
                         placeholder=f"no {c['Dat'].get('type')} files in this folder" if not opts else "choose …")
    _missing(val, opts)
    return val


def one(p, cur, key, label, hlp, c):
    opts = _with(cur, list(nilm.datasets(c['Dat'].get('folder', ''), c['Dat'].get('type'))))
    if not opts:
        return st.text_input(label, value=str(cur or ''), key=key, help=hlp)
    return st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=key, help=hlp)


def _columns(side):
    def widget(p, cur, key, label, hlp, c):
        cols = _labels(c, side)
        cur = list(cur or [])
        if cols is None:                                               # dataset not loaded: free entry
            return forms._list_parse(st.text_input(label, value=forms._list_text(cur), key=key, help=hlp +
                                                   "  \n\nComma-separated (load the dataset under Inputs to pick)."))
        val = st.multiselect(label, _with(cur, cols), default=cur, key=key, help=hlp,
                             placeholder="all" if side == 'X' else "all appliances")
        _missing(val, cols)
        return val
    return widget


def model(p, cur, key, label, hlp, c):
    allowed = MODELS.get(c['Par'].get('solver'), []) or list(p['options'])
    opts = _with(cur, allowed)
    val = st.selectbox(label, opts, index=opts.index(cur), key=key, help=hlp,
                       format_func=lambda k: f"{k}) {p['options'].get(k, '(not defined)')}")
    if val not in allowed:
        st.caption(f":orange[:material/warning:] {val} is not a model of the solver {c['Par'].get('solver')}")
    return val


forms.CUSTOM.update({('Dat', 'folder'): folder, ('Dat', 'train'): train, ('Dat', 'test'): one, ('Dat', 'val'): one,
                     ('Dat', 'out'): _columns('y'), ('Dat', 'inp'): _columns('X'), ('Par', 'model'): model})
