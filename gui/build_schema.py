#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         build_schema.py
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
Builds the parameter schema of the GUI from the Python setup files of BaseNILM (there is no config workbook). Every
line 'setupX['var'] = value  # comment' of start.py and of the function mdlPara() in mdlPara.py becomes a parameter:
the banner titles above it ('# ----' / '# ====' blocks) give the category, the comment the description and the
options ('0) cpu, 1) gpu'). Nested dictionaries (setupMdl['feat']['Mean']) are stored as setup[section][dict][var].

Outputs:    gui/schema.json, config/default.json and config/<folder>_<file>.json for every start file in
            adapter.SETUP_DIRS (with its own mdlPara*.py if it imports one that exists)

Usage:      python gui/build_schema.py
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import ast
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui import adapter as A                                           # noqa: E402

#######################################################################################################################
# Parsing
#######################################################################################################################
ASSIGN = re.compile(r"^\s*(setup(?:Exp|Dat|Par|Mdl))\['(\w+)'\](?:\['(\w+)'\])?\s*=\s*(.*)$")
BANNER = re.compile(r'^\s*#\s*(-{10,}|={10,}|#{10,})\s*$')
TITLE = re.compile(r'^\s*#\s+(\S.*?)\s*$')
MARK = re.compile(r'(?:^|(?<=[,;:|] )|(?<= ))(\d+)\)\s')
UNIT = re.compile(r'\((Hz|samples|dB|s|W|sec)\)')
LEVEL = {'#': 0, '=': 1, '-': 2}
GROUP_PREFIX = {'Sklearn Parameters': 'Sklearn', 'Pattern Matching Parameters': 'Pattern matching',
                'Source Separation Parameters': 'Source separation'}


def _split_comment(rhs):
    """'value  # comment' -> (value text, comment); '#' inside quotes is kept."""
    q = None
    for i, ch in enumerate(rhs):
        if ch in '\'"' and q is None:
            q = ch
        elif ch == q:
            q = None
        elif ch == '#' and q is None:
            return rhs[:i].strip(), rhs[i + 1:].strip()
    return rhs.strip(), ''


def _eval(txt):
    """Literal value of the right-hand side; arithmetic of literals (10/12, 365*24*60) is evaluated."""
    node = ast.parse(txt, mode='eval').body

    def ev(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.List):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            return {ast.Div: lambda: a / b, ast.Mult: lambda: a * b, ast.Add: lambda: a + b,
                    ast.Sub: lambda: a - b, ast.Pow: lambda: a ** b}[type(n.op)]()
        if isinstance(n, ast.Dict) and not n.keys:
            return {}
        raise ValueError(txt)
    return ev(node)


def _options(desc):
    """'0) cpu, 1) gpu' -> (head, {0: 'cpu', 1: 'gpu'}); needs at least two numbered entries."""
    marks = list(MARK.finditer(desc))
    if len(marks) < 2:
        return desc, None
    opts = {}
    for i, mk in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(desc)
        opts[mk.group(1)] = desc[mk.end():end].strip().rstrip(',;|').strip()
    head = desc[:marks[0].start()].strip().rstrip(':').strip()
    return head, opts


def parse(filename, func=None):
    """
    Assignments of a setup file as [(dict name, var, sub var, value, comment, category, group)]. func: only the body
    of this function (mdlPara). Category = the most specific banner title above the line.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if func:
        start = next(i for i, l in enumerate(lines) if re.match(rf'\s*def {func}\(', l))
        lines = lines[start + 1:]
    titles = {0: '', 1: '', 2: ''}
    out = []
    i = 0
    while i < len(lines):
        m = BANNER.match(lines[i])
        if m and i + 2 < len(lines) and TITLE.match(lines[i + 1]) and BANNER.match(lines[i + 2]):
            lvl = LEVEL[m.group(1)[0]]
            titles[lvl] = TITLE.match(lines[i + 1]).group(1)
            for deeper in range(lvl + 1, 3):
                titles[deeper] = ''
            i += 3
            continue
        a = ASSIGN.match(lines[i])
        if a:
            dname, var, sub, rhs = a.groups()
            val_txt, comment = _split_comment(rhs)
            try:
                val = _eval(val_txt)
            except (ValueError, SyntaxError, KeyError):
                i += 1
                continue
            if val != {}:                                              # 'setupMdl['feat'] = {}' only initialises
                cat = titles[2] or titles[1] or titles[0]
                group = titles[0] if titles[2] or titles[1] else ''
                out.append((dname, var, sub, val, comment, cat, group))
        i += 1
    return out


#######################################################################################################################
# Schema
#######################################################################################################################
def _param(section, var, sub, val, comment, cat, group):
    nested = sub is not None
    if nested:                                                         # setupMdl['feat']['Mean']
        cat, var = var, sub
    key = (section, cat, var) if nested else (section, var)
    head, opts = _options(comment)
    if key in A.NO_OPTIONS:
        head, opts = comment, None
    if key in A.OPTIONS:
        opts = {str(k): v for k, v in A.OPTIONS[key].items()}
    unit = UNIT.search(comment)

    if isinstance(val, bool):
        typ = 'bool'
        opts = None
    elif isinstance(val, list):
        typ, opts = 'list', None
    elif opts and all(re.fullmatch(r'-?\d+', k) for k in opts) and not isinstance(val, str):
        typ = 'int'
        opts = {int(k): v for k, v in opts.items()}
    elif isinstance(val, str):
        typ = 'str'
        if opts and val not in opts:                                   # '1) mae, 2) mse' for a text value: by label
            labels = {v.strip("'‘’ ").split(' ')[0]: v for v in opts.values()}
            opts = labels if val in labels else None
    elif isinstance(val, int) or key in A.INTS:
        typ = 'int'
        opts = None
    else:
        typ = 'float'
        opts = None
    return {'cat': cat, 'group': group, 'var': var, 'name': A.LABELS.get(key, var), 'desc': head or comment, 'unit':
            unit.group(1) if unit else None, 'type': typ, 'percent': False, 'options': opts, 'nested': nested,
            'optional': key in A.OPTIONAL, 'default': val}


def build(start_file, mdl_file):
    """Schema {section: [param]} and config {section: {...}} of one start file and its mdlPara file."""
    items = parse(start_file) + parse(mdl_file, 'mdlPara')
    schema, cfg = {}, {}
    seen = {}
    for dname, var, sub, val, comment, cat, group in items:
        section = A.SECTIONS[dname][0]
        # Model parameters of one solver family: one category per family ('Sklearn', 'Pattern matching'), the
        # sub-banner (RF, DTW, ...) becomes a sub-heading
        subcat = None
        if sub is None and section == 'Mdl' and group in GROUP_PREFIX:
            cat, subcat = GROUP_PREFIX[group], cat
        # Same category title under different groups: prefix the group
        elif sub is None and cat and seen.setdefault((section, cat), group) != group:
            cat = f"{group.replace(' Parameters', '')} · {cat}"
        p = _param(section, var, sub, val, comment, cat or A.SECTIONS[dname][1], group)
        if subcat:
            p['sub'] = subcat
        lst = schema.setdefault(section, [])
        old = next((q for q in lst if q['cat'] == p['cat'] and q['var'] == p['var'] and q['nested'] == p['nested']),
                   None)
        if old:                                                        # assigned twice: the last value counts
            old['default'] = p['default']
        else:
            lst.append(p)
        if p['nested']:
            cfg.setdefault(section, {}).setdefault(p['cat'], {})[p['var']] = val
        else:
            cfg.setdefault(section, {})[p['var']] = val
    cfg[A.RUN_SECTION] = {**A.RUN, **cfg.get(A.RUN_SECTION, {})}
    return schema, cfg


def mdl_file_of(start_file):
    """mdlPara file a start file imports ('from mdlParaBench import mdlPara' next to it, else mdlPara.py)."""
    with open(start_file, 'r', encoding='utf-8') as f:
        m = re.search(r'^from (\w+) import mdlPara', f.read(), re.M)
    name = (m.group(1) if m else 'mdlPara') + '.py'
    for d in (os.path.dirname(start_file), A.ROOT):
        if os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return os.path.join(A.ROOT, A.MDL_FILE)


#######################################################################################################################
# Main
#######################################################################################################################
if __name__ == '__main__':
    schema, default = build(os.path.join(A.ROOT, A.START_FILE), os.path.join(A.ROOT, A.MDL_FILE))
    with open(os.path.join(A.ROOT, 'gui', 'schema.json'), 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=1, ensure_ascii=False)
    os.makedirs(os.path.join(A.ROOT, A.CONFIG_DIR), exist_ok=True)
    with open(os.path.join(A.ROOT, A.CONFIG_DIR, 'default.json'), 'w', encoding='utf-8') as f:
        json.dump(default, f, indent=2, ensure_ascii=False)

    # Ready-to-load configs from the setup folders
    n = 0
    for d in A.SETUP_DIRS:
        folder = os.path.join(A.ROOT, d)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            path = os.path.join(folder, fn)
            if not fn.endswith('.py') or fn.lower().startswith('mdlpara'):
                continue
            try:
                _, cfg = build(path, mdl_file_of(path))
            except Exception as e:
                print(f"WARN: {d}/{fn} skipped ({e})")
                continue
            if not cfg.get('Dat'):                                     # no setup assignments (helper file)
                continue
            name = f"{os.path.basename(d)}_{os.path.splitext(fn)[0]}"
            with open(os.path.join(A.ROOT, A.CONFIG_DIR, name + '.json'), 'w', encoding='utf-8') as f:
                json.dump({'_meta': {'tool': A.NAME, 'format': 'config', 'source': f"{d}/{fn}"}, **cfg}, f,
                          indent=2, ensure_ascii=False)
            n += 1
    for sec, ps in schema.items():
        n_opt = sum(1 for p in ps if p['options'])
        print(f"INFO: {sec:<6} {len(ps):>4} parameters ({n_opt} with options)")
    print(f"INFO: {n} configs written from {', '.join(A.SETUP_DIRS)}")
    print(f"DONE: schema with {sum(len(v) for v in schema.values())} parameters written")
