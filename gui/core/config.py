#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         config.py
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
This function handles the configuration of the GUI. The configuration is a nested dictionary with the same structure as
the setup variable of the toolkit. It is stored as JSON file under config/ and replaces the config workbook; workbooks
can still be imported, and for a run the configuration is written back into a copy of the workbook (adapter.FEED).
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A

import copy
import io
import json
import math
import os

#######################################################################################################################
# Paths
#######################################################################################################################
ROOT = A.ROOT
SCHEMA_FILE = os.path.join(ROOT, 'gui', 'schema.json')
CONFIG_DIR = os.path.join(ROOT, A.CONFIG_DIR)
DEFAULT_FILE = os.path.join(CONFIG_DIR, 'default.json')
RUN_FILE = '_gui_run'                                                 # config workbook written for a run

MODES = {m: d['section'] for m, d in A.MODES.items()}
MODE_TITLES = {m: d['title'] for m, d in A.MODES.items()}


#######################################################################################################################
# Schema
#######################################################################################################################
def load_schema():
    """Parameter schema {section: [param, ...]} with integer option keys."""
    with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    for params in schema.values():
        for p in params:
            p.setdefault('nested', True)
            if p['options'] and p['type'] == 'int':
                p['options'] = {int(k): v for k, v in p['options'].items()}
    return schema


def slot(cfg, section, p):
    """Dictionary holding the value of parameter p (setup[section][cat] or setup[section] for flat sections)."""
    sec = cfg.setdefault(section, {})
    return sec.setdefault(p['cat'], {}) if p.get('nested', True) else sec


def get(cfg, section, p):
    return slot(cfg, section, p).get(p['var'], p['default'])


#######################################################################################################################
# Helper
#######################################################################################################################
def _plain(x):
    """Converts numpy scalars and NaN into plain JSON values."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, 'item') and not isinstance(x, (str, bytes)):
        try:
            x = x.item()
        except (ValueError, TypeError):
            return str(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def merge(base, upd):
    """Recursive merge of upd into a copy of base (lists and values are replaced)."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


#######################################################################################################################
# Load and Save
#######################################################################################################################
def default_config():
    with open(DEFAULT_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_configs():
    if not os.path.isdir(CONFIG_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CONFIG_DIR) if f.lower().endswith('.json'))


def from_json(text):
    """Config from JSON text, missing parameters are filled with the defaults."""
    cfg = json.loads(text)
    cfg.pop('_meta', None)
    return merge(default_config(), cfg)


def to_json(cfg):
    out = {'_meta': {'tool': A.NAME, 'format': 'config', 'version': 1}}
    out.update(_plain(cfg))
    return json.dumps(out, indent=2, ensure_ascii=False)


def save_config(cfg, name):
    name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in name.strip()) or 'config'
    filename = os.path.join(CONFIG_DIR, name + '.json')
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(to_json(cfg))
    return filename


def read_config(name):
    with open(os.path.join(CONFIG_DIR, name + '.json'), 'r', encoding='utf-8') as f:
        return from_json(f.read())


#######################################################################################################################
# Workbooks
#######################################################################################################################
def _rows(ws):
    """(row index, {column name: cell}) of a config sheet; the header is the first row."""
    head = {str(c.value).strip(): i for i, c in enumerate(ws[1]) if c.value is not None}
    for row in ws.iter_rows(min_row=2):
        yield {k: row[i] for k, i in head.items() if i < len(row)}


def _key(section, r):
    """(section, cat, var) of a sheet row, same as in build_schema (empty category = section name)."""
    def txt(name):
        c = r.get(name)
        return None if c is None or c.value is None or str(c.value).strip() == '' else str(c.value).strip()
    return section, txt('Category') or section, txt('Variable')


def from_xlsx(content, schema):
    """Imports a config workbook (bytes): the values of the schema parameters are taken, the rest from the defaults."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(content), data_only=True)
    cfg = default_config()
    index = {(s, p['cat'], p['var']): p for s, ps in schema.items() for p in ps}
    for sheet, (section, _) in A.SHEETS.items():
        if sheet not in wb.sheetnames:
            continue
        for r in _rows(wb[sheet]):
            p = index.get(_key(section, r))
            if p is not None:
                v = r['Value'].value
                slot(cfg, section, p)[p['var']] = _plain(str(v) if p['type'] == 'str' and v is not None else v)
    return cfg


def from_py(content):
    """Imports a BaseNILM start file (bytes, e.g. setup/tutorials/Tutorial_2_ML.py) with the mdlPara.py of the root."""
    import tempfile
    from gui.build_schema import build
    with tempfile.NamedTemporaryFile('wb', suffix='.py', delete=False) as f:
        f.write(content)
        name = f.name
    try:
        _, cfg = build(name, os.path.join(ROOT, A.MDL_FILE))
    finally:
        os.remove(name)
    return merge(default_config(), cfg)


def to_xlsx(cfg, schema, filename):
    """Writes the configuration into a copy of the config workbook (values only, layout and other sheets kept)."""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(ROOT, A.CONFIG_WORKBOOK))
    index = {(s, p['cat'], p['var']): p for s, ps in schema.items() for p in ps}
    for sheet, (section, _) in A.SHEETS.items():
        for r in _rows(wb[sheet]):
            p = index.get(_key(section, r))
            if p is not None:
                v = get(cfg, section, p)
                r['Value'].value = v if not isinstance(v, (list, dict)) else json.dumps(v)
    wb.save(filename)
    return filename


#######################################################################################################################
# Setup for a Run
#######################################################################################################################
def build_setup(cfg, mode, schema):
    """Setup variable of the toolkit for operating mode 'mode'."""
    setup = merge(A.init_setup(), cfg)
    if A.MODE_VAR:
        setup[A.RUN_SECTION][A.MODE_VAR] = int(mode)
    if A.FEED == 'xlsx':
        to_xlsx(cfg, schema, os.path.join(CONFIG_DIR, RUN_FILE + '.xlsx'))
        setup[A.RUN_SECTION][A.CONFIG_VAR] = RUN_FILE
        return setup
    return A.cast(setup)
