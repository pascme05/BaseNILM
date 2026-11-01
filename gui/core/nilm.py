#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         nilm.py
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
Domain helpers of the BaseNILM GUI: discovery of the datasets under data/, loading of a dataset with the loaders of
the toolkit (src/data/loadData.py: loadMat, loadPkl, loadXlsx, loadH5, loadParquet), the train / validation / test
split as loadData() performs it, and the probe of the neural network: the training pipeline of the toolkit (main ->
train -> trainMdlTF) runs on a short slice of the data and is stopped right before fit(), so the model is built with
exactly the input and output shapes of a real run.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from gui import adapter as A

# ==============================================================================
# External
# ==============================================================================
import contextlib
import copy
import io
import os
import shutil

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(A.ROOT, 'data')
MDL_DIR = os.path.join(A.ROOT, 'mdl')
PROBE_NAME = '_gui_probe'


#######################################################################################################################
# Datasets
#######################################################################################################################
def folders():
    """Dataset folders under data/ ('' = data/ itself)."""
    if not os.path.isdir(DATA_DIR):
        return ['']
    return [''] + sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))
                         and not d.startswith(('.', '_')))


def datasets(folder, typ=None):
    """{name: extension} of the data files of a folder (optionally of one type)."""
    path = os.path.join(DATA_DIR, folder or '')
    if not os.path.isdir(path):
        return {}
    out = {}
    for f in sorted(os.listdir(path)):
        name, ext = os.path.splitext(f)
        ext = ext.lstrip('.').lower()
        if ext in A.DATA_EXT and not f.startswith('~$') and (typ is None or ext == typ):
            out[name] = ext
    return out


def data_file(folder, name, typ):
    return os.path.join(DATA_DIR, folder or '', f"{name}.{typ}")


def file_size(folder, name, typ):
    p = data_file(folder, name, typ)
    return os.path.getsize(p) if os.path.isfile(p) else 0


def is_big(folder, name, typ):
    return file_size(folder, name, typ) > A.BIG_FILE


def human(n):
    for u in ('B', 'kB', 'MB', 'GB'):
        if n < 1024:
            return f"{n:.0f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


#######################################################################################################################
# Loading
#######################################################################################################################
@contextlib.contextmanager
def quiet():
    """Swallows the console output of the toolkit loaders, the root is the working directory."""
    cwd = os.getcwd()
    buf = io.StringIO()
    try:
        os.chdir(A.ROOT)
        with contextlib.redirect_stdout(buf):
            yield buf
    finally:
        os.chdir(cwd)


@st.cache_resource(show_spinner="Loading dataset with the toolkit loader …", max_entries=3)
def load(folder, name, typ, freq, dim, out_feat, house, _mtime):
    """
    Complete dataset as the loaders of the toolkit return it (before selection, limit and split):
    {'X': DataFrame (time, id, inputs), 'y': DataFrame (time, id, appliances), 'units': {...}, 'log': str}.
    Cached without copies (read-only); _mtime reloads after a change of the file.
    """
    from src.data import loadData as L
    filename = data_file(folder, name, typ)
    setup = {'freq': freq, 'dim': int(dim), 'outFeat': int(out_feat), 'house': int(house or 1)}
    data, units = {}, {}
    with quiet() as buf:
        if typ in ('xlsx', 'csv'):
            data, units = L.loadXlsx(filename, data, units)
        elif typ == 'mat':
            data, units, setup = L.loadMat(filename, setup, data, units)
        elif typ == 'pkl':
            data, units = L.loadPkl(filename, data, units)
        elif typ == 'h5':
            data, units = L.loadH5(filename, data, units, setup)
        elif typ == 'parquet':
            data, units = L.loadParquet(filename, data, units)
    if 'X' not in data or 'y' not in data:
        raise RuntimeError(f"The loader of the toolkit could not read {os.path.relpath(filename, A.ROOT)}:\n"
                           f"{buf.getvalue()[-1500:]}")
    return {'X': data['X'], 'y': data['y'], 'units': units, 'log': buf.getvalue(), 'HF_W': setup.get('HF_W'),
            'HF_F': setup.get('HF_F')}


def load_cfg(cfg, name=None):
    """Dataset of the configuration (name: a file of the dataset folder, default: the first training dataset)."""
    D = cfg['Dat']
    name = name or first_train(cfg)
    return load(D.get('folder', ''), name, D['type'], D.get('freq', 'LF'), D.get('dim', 2), D.get('outFeat', 0),
                D.get('house', 1), os.path.getmtime(data_file(D.get('folder', ''), name, D['type'])))


def first_train(cfg):
    t = cfg['Dat'].get('train') or []
    return str(t[0]) if isinstance(t, list) and t else str(t or cfg['Dat'].get('test', ''))


def exists(cfg, name=None):
    D = cfg['Dat']
    return os.path.isfile(data_file(D.get('folder', ''), name or first_train(cfg), D.get('type', 'mat')))


def channels(df):
    return [c for c in df.columns if c not in ('time', 'id')]


def unit_of(units, side, col):
    try:
        u = units[side][col].iloc[0]
        u = u[0] if isinstance(u, (list, np.ndarray)) and len(u) else u
        return str(u).strip() or ''
    except Exception:
        return ''


def sample_time(df):
    """Sampling time (s) of a dataset as loadData() computes it (time[1] - time[0])."""
    try:
        return float(df['time'].iloc[1] - df['time'].iloc[0])
    except Exception:
        return float('nan')


def selected_outputs(cfg, y):
    """Appliances the toolkit uses: the list 'out', or the largest ones up to the energy share 'outEnergy'."""
    D = cfg['Dat']
    cols = channels(y)
    if D.get('out') and not D.get('outEnergy'):
        return [c for c in D['out'] if c in cols]
    if not D.get('outEnergy'):
        return cols
    energy = y[cols].abs().sum(axis=0).sort_values(ascending=False)
    keep, acc = [], 0.0
    for c, e in energy.items():
        keep.append(c)
        acc += e
        if acc / max(energy.sum(), 1e-12) > D['outEnergy']:
            break
    return keep


#######################################################################################################################
# Split (as in loadData)
#######################################################################################################################
def split_segments(cfg, n, ids=None):
    """
    Index ranges of training / validation / testing for n samples after the limit, as loadData() selects them:
    method 0: training = first rT, validation = first rV of the training part (taken from it), testing = last 1 - rT;
    method 1: k folds (testing fold k, training the others); method 2: whole datasets; method 3: by id.
    Returns [(role, start, stop)] or None for id-based splits.
    """
    E, D = cfg['Exp'], cfg['Dat']
    m = int(E.get('method', 0))
    rT, rV = float(D.get('rT', 0.8)), float(D.get('rV', 0.2))
    if m == 0:
        nT = int(np.ceil(n * rT)) if not D.get('Shuffle') else int(n * rT)
        return [('Training', 0, nT), ('Validation', 0, int(nT * rV)), ('Testing', nT, n)]
    if m == 1:
        k = max(int(E.get('kfold', 10)), 2)
        edges = np.linspace(0, n, k + 1).astype(int)
        return [('Fold', int(edges[i]), int(edges[i + 1])) for i in range(k)]
    return None


#######################################################################################################################
# Model probe
#######################################################################################################################
class _Probe(Exception):
    pass


def _layer_rows(mdl):
    rows = []
    for layer in mdl.layers:
        cfg = layer.get_config()
        try:
            shape = tuple(layer.output.shape)
        except Exception:
            shape = None
        info = []
        for k in ('units', 'filters', 'kernel_size', 'strides', 'pool_size', 'rate', 'num_heads', 'key_dim',
                  'return_sequences'):
            if k in cfg and cfg[k] not in (None, False):
                v = cfg[k]
                v = tuple(v) if isinstance(v, (list, tuple)) else v
                if isinstance(v, tuple) and len(v) == 1:
                    v = v[0]
                if k == 'strides' and v in (1, (1, 1)):                # default strides are not shown
                    continue
                info.append(f"{k}={v}")
        inbound = []
        try:
            for node in layer._inbound_nodes:
                for t in node.input_tensors:
                    inbound.append(t._keras_history[0].name)
        except Exception:
            pass
        rows.append({'name': layer.name, 'type': type(layer).__name__, 'shape': shape, 'params': layer.count_params(),
                     'activation': cfg.get('activation', ''), 'info': ', '.join(info),
                     'inbound': list(dict.fromkeys(inbound))})
    return rows


@st.cache_data(show_spinner="Building the model with the toolkit pipeline …", max_entries=16)
def probe(setup_json):
    """
    Runs main() in training mode on a short slice of the data and stops in fit(): returns the layers of the model that
    trainMdlTF() builds, the shapes of the training data and the log. setup_json: the GUI config (JSON, cache key).
    """
    import json
    import tensorflow as tf
    import src.main as M

    cfg = json.loads(setup_json)
    setup = A.cast(json.loads(json.dumps({**A.init_setup(), **cfg}, default=str)))
    exp, dat, par, mdl = A.split(setup)
    exp.update({'train': 1, 'test': 0, 'method': 0, 'trainBatch': 0, 'plot': 0, 'save': 0, 'name': PROBE_NAME})
    mdl.update({'epoch': 1, 'verbose': 0})
    ratio = 1.0
    dat['lim'] = int(min(dat.get('lim') or 10 ** 9, 20000 * ratio))
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    got = {}
    fit = tf.keras.Model.fit

    def stop(self, x=None, *a, **k):
        got['model'] = self
        try:
            got['batch'] = tuple(t.shape for t in next(iter(x.take(1))))
        except Exception:
            pass
        raise _Probe()

    tf.keras.Model.fit = stop
    buf = io.StringIO()
    cwd = os.getcwd()
    try:
        os.chdir(A.ROOT)
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                M.main(exp, dat, par, mdl, A.paths())
            except _Probe:
                pass
    finally:
        tf.keras.Model.fit = fit
        os.chdir(cwd)
        shutil.rmtree(os.path.join(MDL_DIR, f"mdl_{par.get('model')}_{PROBE_NAME}"), ignore_errors=True)
        for f in (f"mdl_{par.get('model')}_{PROBE_NAME}.weights.h5",):
            if os.path.isfile(os.path.join(MDL_DIR, f)):
                os.remove(os.path.join(MDL_DIR, f))

    if 'model' not in got:
        raise RuntimeError("The model was not built — the training pipeline stopped before fit():\n\n" +
                           buf.getvalue()[-3000:])
    m = got['model']
    try:
        inp = tuple(m.input_shape)
    except Exception:
        inp = None
    try:
        out = tuple(m.output_shape)
    except Exception:
        out = None
    return {'layers': _layer_rows(m), 'input': inp, 'output': out, 'params': int(m.count_params()),
            'trainable': int(sum(np.prod(w.shape) for w in m.trainable_weights)), 'batch': got.get('batch'),
            'log': buf.getvalue(), 'cls': type(m).__name__, 'optimizer': None}


def weights_file(cfg):
    """Weights a test run without training loads (mdl/mdl_<model>_<name>.weights.h5)."""
    return os.path.join(MDL_DIR, f"mdl_{cfg['Par'].get('model')}_{cfg['Exp'].get('name')}.weights.h5")


def probe_key(cfg):
    """Part of the configuration the model depends on (JSON, cache key of probe())."""
    import json
    c = copy.deepcopy(cfg)
    c['Exp'].update({'name': PROBE_NAME, 'author': '', 'plot': 0, 'save': 0, 'log': 0, 'warn': 3, 'train': 1,
                     'test': 0, 'kfold': 2, 'sim': 0})                # values that do not change the model
    c['Mdl'].update({'epoch': 1, 'patience': 1, 'verbose': 0})
    return json.dumps(c, sort_keys=True, default=str)
