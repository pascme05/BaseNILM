#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         results.py
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
Results browser: all results folders of all projects (also runs started with start.py). GUI runs are restored
interactively, all other runs show their figures, reports and data files.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import files, forms
from gui.core import viz_results as vr
from gui.core.config import MODE_TITLES, to_json
from gui.core.style import table

import os
import time

import pandas as pd
import streamlit as st


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Results browser")
    projects = files.list_projects()
    if not projects:
        st.info("No results yet.")
        return
    cur = os.path.basename(A.results_root(forms.cfg()))
    c1, c2 = st.columns([1, 3])
    with c1:
        proj = projects[0] if projects == [''] else \
            st.selectbox("Project", projects, index=projects.index(cur) if cur in projects else 0, key='rb|proj')
    runs = files.list_runs(proj)
    if not runs:
        st.info("No results folders in this project.")
        return

    rows = []
    for f in runs:
        meta = files.run_meta(f) or {}
        rf = files.run_files(f)
        rows.append({'folder': os.path.basename(f), 'mode': MODE_TITLES.get(meta.get('mode'), '–'),
                     'started': meta.get('time') or time.strftime('%Y-%m-%d %H:%M',
                                                                  time.localtime(os.path.getmtime(f))),
                     'source': 'GUI' if meta else 'script', 'figures': len(rf['figures']),
                     'status': 'error' if meta.get('error') else 'ok'})
    with c2:
        st.caption(f"{len(rows)} results folders in {os.path.join(A.RESULTS_DIR, proj)}/ — select a row to open it.")
    sel = table(pd.DataFrame(rows), hide_index=True, on_select='rerun', selection_mode='single-row', key='rb|table',
                height=260)
    idx = sel.selection.rows[0] if sel and sel.selection.rows else 0
    folder = runs[idx]
    st.subheader(os.path.basename(folder))

    run = None
    try:
        run = files.load_run(folder)
    except Exception as e:
        st.warning(f"Stored data could not be restored ({e}); showing the files only.")
    if run is None:
        vr.gallery({'folders': [folder]}, key='rb|g')
        return
    vr.run_view(run, key='rb|run')
    if run.get('setup'):
        with st.expander("Configuration of this run"):
            st.download_button("Download as config JSON", to_json(run['setup']),
                               file_name=f"{os.path.basename(folder)}.json", icon=':material/download:')
            if st.button("Load into the GUI", icon=':material/input:'):
                forms.set_config(run['setup'], os.path.basename(folder))
                st.toast("Configuration loaded", icon=':material/check:')
                st.rerun()
