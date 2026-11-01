#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         inputs.py
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
Input data page: the input files of the configuration (grouped as in adapter.FILE_GROUPS, the group used by the
current settings first) and a browser of any workbook of the input folders (adapter.FOLDERS).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui.core import forms, nilm
from gui.core import viz_inputs as vi

import streamlit as st


#######################################################################################################################
# Page
#######################################################################################################################
def page():
    st.title("Input data")
    cfg = forms.cfg()
    D = cfg['Dat']
    st.caption("Datasets of data/ read with the loaders of the toolkit. The charts use the current configuration "
               "(appliances, threshold, limit, split, framing); the dataset picked here is only viewed — the training "
               "datasets are set on the Setup page.")
    files_ = nilm.datasets(D.get('folder', ''), D.get('type'))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"**Folder** data/{D.get('folder', '')} · **format** {D.get('type')}")
    if not files_:
        st.warning(f"No {D.get('type')} files in data/{D.get('folder', '')}.")
        return
    names = list(files_)
    cur = nilm.first_train(cfg)
    with c2:
        name = st.selectbox("Dataset", names, index=names.index(cur) if cur in names else 0, key='in|ds',
                            format_func=lambda n: f"{n}  ({nilm.human(nilm.file_size(D.get('folder', ''), n, D['type']))})"
                            + ("  · training dataset" if n in (D.get('train') or []) else ''))
    st.divider()
    vi.explorer(cfg, key='in', name=name)
