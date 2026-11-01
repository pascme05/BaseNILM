#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         mode.py
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
Page of one operating mode with the views workflow (if defined), parameters, inputs, run and results.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core import files, forms, theory, workflow, architecture
from gui.core import viz_inputs as vi
from gui.core import viz_results as vr
from gui.views.project import files_form, file_groups, active_files

import os

import streamlit as st

VIEW_ICONS = {'Workflow': ':material/account_tree:', 'Parameters': ':material/tune:', 'Inputs': ':material/table_chart:',
              'Theory': ':material/function:', 'Run': ':material/play_arrow:', 'Results': ':material/monitoring:',
              'Model': ':material/hub:'}
MODEL_VIEW = (0, 2)                                                   # modes with the view 'Model' (architecture)


def views(m):
    return (['Workflow'] if m in workflow.WORKFLOWS else []) + ['Parameters', 'Inputs'] + \
        (['Model'] if m in MODEL_VIEW else []) + (['Theory'] if m in theory.THEORY else []) + ['Run', 'Results']


#######################################################################################################################
# Theory
#######################################################################################################################
def theory_view(m):
    theory.THEORY[m](forms.cfg())


#######################################################################################################################
# Parameters
#######################################################################################################################
def feature_cats():
    """Categories of setupMdl that hold the feature lists (banner 'Features' of mdlPara.py)."""
    return [p['cat'] for p in forms.schema().get('Mdl', []) if p.get('group') == 'Features']


def parameters(m):
    d = A.MODES[m]
    secs = [d['section']] + list(d.get('extra', []))
    if 'Mdl' in secs and feature_cats():                               # features get their own tab after 'Model'
        secs.insert(secs.index('Mdl') + 1, 'Feat')
    titles = {**getattr(A, 'SECTION_TITLES', {}), 'Feat': 'Features'}
    icons = {'Exp': ':material/science:', 'Dat': ':material/database:', 'Par': ':material/settings:',
             'Mdl': ':material/neurology:', 'Feat': ':material/insights:', 'Grid': ':material/grid_on:'}
    fc = list(dict.fromkeys(feature_cats()))
    for tab, sec in zip(st.tabs([f"{icons.get(s_, '')} {titles.get(s_, s_)}" for s_ in secs]), secs):
        with tab:
            if sec == 'Grid':
                from gui.core import grid
                grid.form(forms.cfg())
                continue
            name = 'Mdl' if sec == 'Feat' else sec
            cats = fc if sec == 'Feat' else ([c_ for c_ in dict.fromkeys(p['cat'] for p in forms.schema()[sec])
                                              if c_ not in fc] if sec == 'Mdl' else None)
            c1, c2 = st.columns([5, 1])
            with c1:
                st.caption(SECTION_TEXT.get(sec, ''))
            with c2:
                forms.reset_section(name, cats=cats, key=f"reset|{sec}")
            forms.section(name, cats=cats)


SECTION_TEXT = {'Exp': "Validation method and whether a model is trained and/or tested (setupExp of start.py).",
                'Dat': "Dataset, appliances, sampling and pre-processing (setupDat of start.py).",
                'Par': "Solver, model, framing, features and post-processing (setupPar of start.py).",
                'Mdl': "Hyperparameters of the selected solver (setupMdl of mdlPara.py).",
                'Feat': "Features computed from the input frames: statistical, rolling and 2D features (setupMdl"
                        "['feat'], ['feat_roll'], ['feat2D'] of mdlPara.py); used when Features (Par) is switched on."}


#######################################################################################################################
# Inputs
#######################################################################################################################
def inputs(m):
    cfg = forms.cfg()
    if m in vi.INPUTS:
        vi.INPUTS[m](cfg)
        return
    fields = active_files(cfg)
    if not fields:
        st.info("This mode reads no input workbooks.")
        return
    title = next((g['title'] for g, _, a in file_groups(cfg) if a and g['title']), None)
    if title:
        st.caption(f"Input files of the **{title}** (selected by the topology).")
    for tab, (var, kind, label, _) in zip(st.tabs([f"{f[2]} · {cfg[A.RUN_SECTION].get(f[0])}" for f in fields]),
                                          fields):
        with tab:
            vi.workbook(kind[5:], cfg[A.RUN_SECTION].get(var), key=f"m{m}|{var}")


#######################################################################################################################
# Run
#######################################################################################################################
def _summary(m):
    cfg = forms.cfg()
    if hasattr(A, 'summary'):
        return A.summary(cfg, m)
    G = cfg[A.RUN_SECTION]
    rows = [('Simulation', f"{G.get('Name', '')}  →  {os.path.relpath(A.results_root(cfg), A.ROOT)}/")]
    rows += [(label, str(G.get(var))) for var, kind, label, _ in active_files(cfg)]
    if A.MODES[m].get('sub'):
        rows.append(('Sub-mode', forms.option_label(*A.MODES[m]['sub'])))
    return rows


def run_view(m):
    cfg = forms.cfg()
    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.markdown("**Run settings**")
            for k, v in _summary(m):
                st.markdown(f"<span style='color:#52514e'>{k}</span> &nbsp; {v}", unsafe_allow_html=True)
    with c2:
        start = st.button(f"Run {A.MODES[m]['title']}", type='primary', icon=':material/play_arrow:',
                          width='stretch', key=f"run|{m}")
        q = A.QUICK
        is_quick = st.toggle("Quick check", key=f"quick|{m}", value=st.session_state.get('quick_default', False),
                             help=f"Checks the whole pipeline fast: {q['epoch']} epoch, at most {q['lim']:,} samples, "
                                  f"k-fold ≤ {q['kfold']} folds, grid ≤ {q['grid']} × {q['grid']} points. Trained "
                                  f"models get the suffix _check, so no trained model is overwritten. The scores are "
                                  f"not meaningful.")
        st.caption("The simulation runs locally in this process; the console output is streamed below.")
    for level, txt in (A.warnings(cfg, m) if hasattr(A, 'warnings') else []):
        (st.error if level == 'error' else st.warning if level == 'warning' else st.info)(txt)

    if start:
        from gui.core.runner import run_mode
        with st.status(f"Running {A.MODES[m]['title']} …", expanded=True) as status:
            box = st.empty()
            run_cfg = {**cfg, '_quick': bool(is_quick)}
            run = run_mode(run_cfg, m, forms.schema(), stream=lambda txt: box.code(txt[-8000:], language=None,
                                                                                    height=380))
            if run['error']:
                status.update(label=f"Stopped with an error after {run['duration']:.1f} s", state='error')
            else:
                status.update(label=f"Finished in {run['duration']:.1f} s", state='complete', expanded=False)
        st.session_state.runs[m] = run
        try:
            files.save_run(run)
        except Exception as e:
            st.warning(f"The run could not be stored for the results browser: {e}")
        if not run['error']:
            st.session_state[f"goto|{m}"] = 'Results'
            st.rerun()
        else:
            st.error(run['error'])
            with st.expander("Console log"):
                st.code(run['log'][-20000:], language=None)


#######################################################################################################################
# Results
#######################################################################################################################
def results_view(m):
    cfg = forms.cfg()
    root = A.results_root(cfg)
    past = []
    if os.path.isdir(root):
        for f in files.list_runs('', root=root):
            meta = files.run_meta(f)
            if meta and meta.get('mode') == m:
                past.append((f, meta))
    opts = (['session'] if m in st.session_state.runs else []) + [f for f, _ in past]
    if not opts:
        vr.run_view(None)
        return
    lab = {f: f"{os.path.basename(f)} · {meta.get('time', '')}" for f, meta in past}
    lab['session'] = "Latest run of this session"
    sel = st.selectbox("Results", opts, format_func=lambda o: lab[o], key=f"res|pick|{m}",
                       help=f"GUI runs of this mode in {os.path.relpath(root, A.ROOT)}/")
    run = st.session_state.runs[m] if sel == 'session' else files.load_run(sel)
    vr.run_view(run, key=f"res|{m}")


#######################################################################################################################
# Page
#######################################################################################################################
def make_page(m):
    def page():
        d = A.MODES[m]
        if A.MODE_VAR:                                                  # activity rules and saved configs see the mode
            forms.cfg()[A.RUN_SECTION][A.MODE_VAR] = m
        st.title(f"{m} · {d['title']}" if len(A.MODES) > 1 else d['title'])
        st.caption(d['text'])
        vs = views(m)
        key = f"view|{m}"
        if f"goto|{m}" in st.session_state:
            st.session_state[key] = st.session_state.pop(f"goto|{m}")
        if st.session_state.get(key) not in vs:
            st.session_state[key] = vs[0]
        view = st.segmented_control("View", vs, key=key, label_visibility='collapsed',
                                    format_func=lambda v: f"{VIEW_ICONS[v]} {v}") or vs[0]
        st.divider()
        {'Workflow': workflow.view, 'Parameters': parameters, 'Inputs': inputs, 'Theory': theory_view,
         'Run': run_view, 'Results': results_view, 'Model': architecture.view}[view](m)

    page.__name__ = f"mode{m}"
    return page
