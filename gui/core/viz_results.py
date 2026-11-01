#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         viz_results.py
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
This function shows the results of a run: interactive charts (RENDER: mode -> function(data, setup, key), generic key
figures otherwise), the figures of the toolkit (captured from matplotlib and found in the results folder), the files
of the results folder and the console log.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import files
from gui.core.style import fmt, table

import os

import numpy as np
import pandas as pd
import streamlit as st


#######################################################################################################################
# Helper
#######################################################################################################################
def arr(x):
    return np.asarray(pd.to_numeric(pd.Series(np.ravel(np.asarray(x, dtype=object))), errors='coerce'), dtype=float)


def scalars(d, prefix='', depth=0):
    """Flat table of the scalar entries of a nested result dictionary."""
    rows = []
    if depth > 6:
        return rows
    for k, v in (d or {}).items() if isinstance(d, dict) else []:
        if isinstance(v, dict):
            rows += scalars(v, f"{prefix}{k}.", depth + 1)
        elif isinstance(v, (int, float, np.integer, np.floating, str)) and not isinstance(v, (bool, np.bool_)):
            rows.append({'quantity': f"{prefix}{k}", 'value': v if isinstance(v, str) else float(v)})
    return rows


def frames(d, prefix='', depth=0):
    """DataFrames and Series of a nested result dictionary {path: DataFrame}."""
    out = {}
    if depth > 6 or not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, pd.DataFrame) and not v.empty:
            out[f"{prefix}{k}"] = v
        elif isinstance(v, pd.Series) and not v.empty:
            out[f"{prefix}{k}"] = v.to_frame()
        elif isinstance(v, dict):
            out.update(frames(v, f"{prefix}{k}.", depth + 1))
    return out


def generic(data, setup, key):
    """Key figures and tables of a result dictionary (until a dedicated view is added to RENDER)."""
    rows = scalars(data)
    if rows:
        st.markdown("**Key figures**")
        table(pd.DataFrame(rows), hide_index=True, height=min(420, 38 + 35 * len(rows)))
    tabs = frames(data)
    if tabs:
        pick = st.selectbox("Table", list(tabs), key=f"{key}|tab")
        table(tabs[pick], height=360)
    if not rows and not tabs:
        st.info("The run returned no scalar results or tables — see *Figures & files*.")


#######################################################################################################################
# Figures and files
#######################################################################################################################
def gallery(run, key='gal'):
    figs = [(t, b) for t, b in run.get('figs') or []]
    disk, reports, datas = [], [], []
    for f in run.get('folders') or []:
        rf = files.run_files(f)
        disk += rf['figures']
        reports += rf['report']
        datas += rf['data']
    items = figs + [(os.path.splitext(os.path.basename(f))[0].replace('_', ' '), f) for f in disk]
    if not items and not reports and not datas:
        st.info("No figures or files for this run (enable rendering/saving on the Setup page).")
        return

    if items:
        st.markdown(f"**Figures of the toolkit** ({len(items)})")
        mode = st.segmented_control("View", ['Grid', 'Single'], default='Grid', key=f"{key}|view",
                                    label_visibility='collapsed')
        if mode == 'Single':
            i = st.selectbox("Figure", range(len(items)), format_func=lambda i: items[i][0], key=f"{key}|one")
            st.image(items[i][1], caption=items[i][0], width='stretch')
        else:
            cols = st.columns(2)
            for i, (n, img) in enumerate(items):
                with cols[i % 2]:
                    st.image(img, caption=n, width='stretch')

    if reports or datas:
        st.markdown("**Files**")
        cols = st.columns(3)
        for i, f in enumerate(reports + datas):
            with cols[i % 3]:
                with open(f, 'rb') as fh:
                    st.download_button(os.path.basename(f), fh.read(), file_name=os.path.basename(f),
                                       icon=':material/picture_as_pdf:' if f.lower().endswith('.pdf')
                                       else ':material/download:', key=f"{key}|dl|{f}", width='stretch')
    if run.get('folders'):
        st.caption("Folder: " + " · ".join(os.path.relpath(f, files.RESULTS_DIR) for f in run['folders']))


#######################################################################################################################
# Interactive results per mode: mode -> function(data, setup, key)
#######################################################################################################################
from gui.core.results_nilm import RENDER                        # noqa: E402  (mode -> renderer)


#######################################################################################################################
# Run
#######################################################################################################################
def run_view(run, key='res'):
    """Interactive results (if the data are available), figures and files, and the log of one run."""
    if run is None:
        st.info("No results yet — start a simulation in the **Run** view or open a results folder.")
        return
    if run.get('error'):
        st.error(f"The simulation stopped with an error:\n\n```\n{run['error']}\n```")
    st.caption(f"Run {run.get('time', '')} · {fmt(run.get('duration', 0))} s")
    view = st.segmented_control("Show", ['Interactive', 'Figures & files', 'Console log'], default='Interactive',
                                key=f"{key}|v", label_visibility='collapsed')
    if view == 'Interactive':
        if run.get('data') is not None:
            if isinstance(run['data'], dict):
                run['data'].setdefault('_log', run.get('log', ''))
            try:
                RENDER.get(run['mode'], generic)(run['data'], run.get('setup'), key=f"{key}|{run['mode']}")
            except Exception as e:
                st.warning(f"Interactive charts not available for this run ({type(e).__name__}: {e}). The figures "
                           f"of the toolkit are shown under 'Figures & files'.")
        else:
            st.info("No interactive data stored for this run — see 'Figures & files'.")
    elif view == 'Figures & files':
        gallery(run, key=f"{key}|g")
    elif view == 'Console log':
        st.code(run.get('log') or '(log not stored for this run)', language=None, height=500)
