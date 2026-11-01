#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         docs.py
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
Documentation page: renders docs/GUI.md (one tab per chapter) and a searchable reference of all parameters.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import forms
from gui.core.config import ROOT
from gui.core.help import HELP, BASIC, key as help_key
from gui.core.style import table

import os
import re

import pandas as pd
import streamlit as st

DOC = os.path.join(ROOT, getattr(A, 'DOC', 'docs/GUI.md'))


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Documentation")
    chapters = []
    if os.path.isfile(DOC):
        parts = re.split(r'(?m)^## ', open(DOC, encoding='utf-8').read())
        st.markdown(re.sub(r'(?m)^# .*\n', '', parts[0]))
        chapters = parts[1:]
    titles = [c.split('\n', 1)[0].strip() for c in chapters] + ['Parameter reference']
    tabs = st.tabs(titles)
    for tab, c in zip(tabs, chapters):
        with tab:
            st.markdown(c.split('\n', 1)[1] if '\n' in c else '')
    with tabs[-1]:
        q = st.text_input("Search", placeholder="name, variable or text", key='doc|q')
        rows = []
        for sec, ps in forms.schema().items():
            for p in ps:
                k = help_key(sec, p)
                rows.append({'key': k, 'name': p['name'], 'unit': '%' if p['percent'] else (p['unit'] or ''),
                             'level': 'basic' if not BASIC or k in BASIC else 'expert',
                             'options': '; '.join(f"{a}) {b}" for a, b in p['options'].items()) if p['options']
                             else '', 'help': HELP.get(k) or p['desc'] or ''})
        df = pd.DataFrame(rows)
        if q:
            df = df[df.apply(lambda r: q.lower() in ' '.join(map(str, r.values)).lower(), axis=1)]
        st.caption(f"{len(df)} parameters")
        table(df, hide_index=True, height=560, column_config={
            'help': st.column_config.TextColumn('help', width='large'),
            'options': st.column_config.TextColumn('options', width='medium')})
