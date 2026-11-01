#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         files.py
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
This function lists and loads the input workbooks for the GUI, stores the runs of the GUI in their results folders and
re-opens them. Loaded workbooks are cached by file name and modification time. Where the toolkit has own loaders, use
them (see load_workbook) so the GUI shows exactly what a simulation reads.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core.config import ROOT

import json
import os
import pickle

import pandas as pd
import streamlit as st

#######################################################################################################################
# Folders
#######################################################################################################################
FOLDERS = {k: os.path.join(ROOT, v) for k, v in A.FOLDERS.items()}
RESULTS_DIR = os.path.join(ROOT, A.RESULTS_DIR)
GUI_PICKLE = 'gui_run.pkl'
GUI_META = 'gui_run.json'


def list_files(kind):
    """Workbook names (without extension) of an input folder, Excel lock files are skipped."""
    folder = FOLDERS[kind]
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(folder)
                  if f.lower().endswith(A.FILE_EXT) and not f.startswith('~$'))


def file_path(kind, name):
    for ext in A.FILE_EXT:
        p = os.path.join(FOLDERS[kind], name + ext)
        if os.path.isfile(p):
            return p
    return os.path.join(FOLDERS[kind], name + A.FILE_EXT[0])


def mtime(kind, name):
    p = file_path(kind, name)
    return os.path.getmtime(p) if os.path.isfile(p) else 0.0


def save_upload(kind, upload):
    """Stores an uploaded workbook in the input folder, returns its name."""
    name, ext = os.path.splitext(os.path.basename(upload.name))
    os.makedirs(FOLDERS[kind], exist_ok=True)
    with open(os.path.join(FOLDERS[kind], name + ext.lower()), 'wb') as f:
        f.write(upload.getbuffer())
    st.cache_data.clear()
    return name


@st.cache_data(show_spinner="Loading workbook …")
def load_workbook(kind, name, _mtime):
    """All sheets of an input workbook {sheet: DataFrame} (_mtime: cache key, reloads after a change)."""
    p = file_path(kind, name)
    if p.lower().endswith('.csv'):
        return {name: pd.read_csv(p)}
    return pd.read_excel(p, sheet_name=None)


#######################################################################################################################
# Results
#######################################################################################################################
def list_projects():
    """Project folders under results/ ([''] if the toolkit writes results/<run> directly, adapter.RESULT_PROJECTS)."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    if not getattr(A, 'RESULT_PROJECTS', True):
        return ['']
    return sorted(d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d)))


def list_runs(project, root=None):
    """Result folders of a project (or of the folder root), newest first."""
    root = root or os.path.join(RESULTS_DIR, project)
    if not os.path.isdir(root):
        return []
    runs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return sorted(runs, key=os.path.getmtime, reverse=True)


def run_files(folder):
    """Figures, reports and data files of a results folder (any depth)."""
    out = {'figures': [], 'report': [], 'data': []}
    for base, _, names in os.walk(folder):
        for f in sorted(names):
            p, low = os.path.join(base, f), f.lower()
            if low.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                out['figures'].append(p)
            elif low.endswith('.pdf'):
                out['report'].append(p)
            elif low.endswith(('.xlsx', '.csv', '.json', '.txt', '.mat')) and f != GUI_META:
                out['data'].append(p)
    return out


def save_run(run):
    """Stores the results of a GUI run in its results folder(s) so it can be re-opened in the results browser."""
    keep = {k: run.get(k) for k in ('mode', 'setup', 'time', 'duration', 'figs', 'log', 'error')}
    keep['data'] = A.compact(run.get('data'), run['mode'])
    meta = {'mode': run['mode'], 'time': run['time'], 'duration': run['duration'], 'error': bool(run.get('error')),
            'name': str(run['setup'].get(A.RUN_SECTION, {}).get('Name', run['setup'].get(A.RUN_SECTION, {}).get('name',
                                                                                                                '')))}
    if (not run['folders'] or run['mode'] == 2) and hasattr(A, 'run_folder'):  # no folder of its own / grid
        run['folders'] = [A.run_folder(run['setup'])]
    for folder in run['folders']:
        try:
            with open(os.path.join(folder, GUI_PICKLE), 'wb') as f:
                pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:                                                  # unpicklable data: figures and log only
            with open(os.path.join(folder, GUI_PICKLE), 'wb') as f:
                pickle.dump({**keep, 'data': None}, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(folder, GUI_META), 'w', encoding='utf-8') as f:
            json.dump(meta, f)


def run_meta(folder):
    """Mode and time of a GUI run (None for runs not started from the GUI)."""
    p = os.path.join(folder, GUI_META)
    if os.path.isfile(p):
        with open(p, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_run(folder):
    """Re-opens a GUI run (None for script runs, these show their files only)."""
    p = os.path.join(folder, GUI_PICKLE)
    if not os.path.isfile(p):
        return None
    with open(p, 'rb') as f:
        run = pickle.load(f)
    run['folders'] = [folder]
    run.setdefault('log', '')
    run.setdefault('figs', [])
    return run
