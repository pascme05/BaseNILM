#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         tutorials.py
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
Tutorials page: guided scenarios for the operating modes (gui/core/tutorials.py). A tutorial loads its own config,
then the steps lead through the views of the mode page.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms
from gui.core import tutorials as tut
from gui.core.style import table
from gui.views.registry import PAGES

import pandas as pd
import streamlit as st

LEVEL_COL = {'Beginner': 'green', 'Intermediate': 'blue', 'Advanced': 'orange'}


#######################################################################################################################
# Helper
#######################################################################################################################
def _mode(m):
    return f"{m} · {A.MODES[m]['title']}" if m in A.MODES else str(m)


def _changes(t):
    """Readable table of the base config and the changes of a tutorial."""
    rows = [{'setting': 'base configuration', 'value': f"config/{t.get('base', 'default')}.json"}]
    runs = {var: label for var, _, label, _ in A.RUN_FIELDS}
    for k, v in t['changes'].items():
        label, (sec, *rest) = k, k.split('.')
        if sec == A.RUN_SECTION and len(rest) == 1 and rest[0] in runs:
            label = f"Run settings › {runs.get(rest[0], rest[0])}"
        elif len(rest) in (1, 2):
            try:
                p = forms.find(sec, *rest) if len(rest) == 2 else \
                    next(q for q in forms.schema()[sec] if q['var'] == rest[0] and not q.get('nested'))
                label = f"{sec} › {p['cat']} › {p['name']}"
                if p.get('options') and v in p['options']:
                    v = f"{v}: {p['options'][v]}"
                elif p.get('unit'):
                    v = f"{v} {p['unit']}"
            except (KeyError, StopIteration):
                pass
        rows.append({'setting': label, 'value': '(empty)' if v is None else str(v)})
    return pd.DataFrame(rows)


def _load(t):
    forms.set_config(tut.load(t['id']), f"tutorials/{t['id']}")
    st.session_state[f"goto|{t['mode']}"] = 'Theory' if 'Theory' in _views(t['mode']) else 'Parameters'
    st.session_state['tutorial'] = t['id']


def _views(m):
    from gui.views.mode import views
    return views(m)


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Tutorials")
    st.caption("Guided scenarios for the operating modes. *Load* sets the configuration of the tutorial — your current "
               "configuration is replaced, save it first in the sidebar if needed. The values under *What you should "
               "see* come from runs of the tutorial configurations.")
    modes = sorted({t['mode'] for t in tut.TUTORIALS})
    m = st.segmented_control("Mode", ['all'] + modes, default='all', key='tut|mode',
                             format_func=lambda x: 'All modes' if x == 'all' else _mode(x))
    lst = [t for t in tut.TUTORIALS if m in (None, 'all') or t['mode'] == m]
    c1, c2 = st.columns([2, 5])
    with c1:
        ids = [t['id'] for t in lst]
        cur = st.session_state.get('tutorial')
        sel = st.radio("Tutorial", ids, index=ids.index(cur) if cur in ids else 0, key='tut|sel',
                       format_func=lambda i: next(f"{t['mode']} · {t['title']}" for t in lst if t['id'] == i),
                       label_visibility='collapsed')
    t = next(x for x in lst if x['id'] == sel)
    with c2:
        with st.container(border=True):
            st.markdown(f"### {t['title']}")
            st.markdown(f":{LEVEL_COL[t['level']]}-badge[{t['level']}] :gray-badge[:material/schedule: about "
                        f"{t['minutes']} min] :gray-badge[{_mode(t['mode'])}]")
            st.markdown(f"**Scenario.** {t['story']}")
            st.markdown("**You will learn to** " + "; ".join(t['goals']) + ".")
            b1, b2, _ = st.columns([2, 2, 3])
            with b1:
                if st.button("Load tutorial", type='primary', icon=':material/download:', key='tut|load',
                             width='stretch'):
                    _load(t)
                    st.toast(f"Tutorial '{t['title']}' loaded", icon=':material/check:')
                    if t['mode'] in PAGES:
                        st.switch_page(PAGES[t['mode']])
            with b2:
                if t['mode'] in PAGES:
                    st.page_link(PAGES[t['mode']], label=f"Open {A.MODES[t['mode']]['title']}",
                                 icon=':material/arrow_forward:')
            if st.session_state.get('cfg_name') == f"tutorials/{t['id']}":
                st.caption(":material/check_circle: This tutorial is the active configuration.")
        t1, t2, t3, t4 = st.tabs(["Steps", "What you should see", "Exercises", "Settings of the tutorial"])
        with t1:
            for i, s in enumerate(t['steps'], start=1):
                st.markdown(f"{i}. {s}")
        with t2:
            st.markdown(t['look'] or "–")
        with t3:
            for s in t['exercise']:
                st.markdown(f"- {s}")
        with t4:
            table(_changes(t), hide_index=True, width='stretch')
            st.caption(f"Config file: {A.CONFIG_DIR}/tutorials/{t['id']}.json — all other parameters are those of the "
                       f"base configuration. Load it in the sidebar or import it as JSON.")
