#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         runner.py
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
This function runs a simulation of the toolkit from the GUI configuration. The console output is captured (and can be
streamed to the GUI), figures are rendered without a window (Agg backend) and kept as PNG, and the created results
folders are detected.
Inputs:     1) cfg:     GUI configuration
            2) mode:    operating mode
            3) schema:  parameter schema
            4) stream:  optional callable receiving the console text
Outputs:    1) run:     dictionary with data, setup, figures, results folders, log, error and duration
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# External
# ==============================================================================
import matplotlib
matplotlib.use('Agg')

import contextlib
import copy
import io
import os
import time
import traceback

# ==============================================================================
# Internal
# ==============================================================================
from gui import adapter as A
from gui.core.config import ROOT, build_setup

MAX_FIGS = 60


#######################################################################################################################
# Log Stream
#######################################################################################################################
class LogStream:
    """File-like object collecting the console output; forwards it to a callback at most every 'every' seconds."""

    def __init__(self, callback=None, every=0.5):
        self.parts, self.callback, self.every, self.last = [], callback, every, 0.0

    def write(self, text):
        # tqdm redraws its bar with carriage returns, only the final state of a line is kept
        if '\r' in text:
            text = text.split('\r')[-1]
            if self.parts and not self.parts[-1].endswith('\n'):
                self.parts[-1] = ''
        self.parts.append(text)
        now = time.time()
        if self.callback is not None and now - self.last > self.every:
            self.last = now
            try:
                self.callback(self.text())
            except Exception:
                pass
        return len(text)

    def flush(self):
        pass

    def close(self):                                                   # loggers that kept the stream close it at exit
        pass

    def isatty(self):
        return False

    def text(self):
        return ''.join(self.parts)


#######################################################################################################################
# Helper
#######################################################################################################################
def _folders(root):
    """Sub-folders of root with their modification time {name: mtime}."""
    if not os.path.isdir(root):
        return {}
    out = {}
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            out[d] = os.path.getmtime(p)
    return out


def _changed(before, after):
    """Folders created or written to by the run (toolkits often reuse results/<Name>)."""
    return sorted(d for d, t in after.items() if d not in before or t > before[d])


@contextlib.contextmanager
def headless():
    """
    Keeps the Agg backend during a run: toolkits that call matplotlib.use('TkAgg') at import would otherwise open
    hidden windows in plt.show() and block the run. plt.show() is a no-op, the figures stay open for capturing.
    """
    import matplotlib.pyplot as plt
    use, show = matplotlib.use, plt.show                                # not switch_backend: pyplot needs it itself
    matplotlib.use = lambda *a, **k: None
    plt.show = lambda *a, **k: None
    try:
        yield
    finally:
        matplotlib.use, plt.show = use, show


def _figures():
    """Open matplotlib figures as (title, PNG bytes); toolkits that only call plt.show() are shown this way."""
    import matplotlib.pyplot as plt
    out = []
    for n in plt.get_fignums()[:MAX_FIGS]:
        fig = plt.figure(n)
        title = fig.get_label() or (fig._suptitle.get_text() if getattr(fig, '_suptitle', None) else '') or \
            next((a.get_title() for a in fig.axes if a.get_title()), '') or f"Figure {n}"
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
            out.append((title, buf.getvalue()))
        except Exception:
            pass
    return out


#######################################################################################################################
# Function
#######################################################################################################################
def run_mode(cfg, mode, schema, stream=None):
    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    root = A.results_root(cfg)
    before = _folders(root)
    log = LogStream(stream)
    out = {'mode': mode, 'data': None, 'setup': copy.deepcopy(cfg), 'folders': [], 'figs': [], 'error': None,
           'log': '', 'duration': 0.0, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    import matplotlib.pyplot as plt
    plt.close('all')
    t0 = time.perf_counter()
    cwd = os.getcwd()
    try:
        os.chdir(ROOT)                                                       # toolkits use paths relative to the root
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log), headless(), A.session():
            setup = build_setup(cfg, mode, schema)
            out['data'] = A.run(setup, A.paths(), mode)
    except SystemExit as e:                                                   # loaders stop with sys.exit
        out['error'] = str(e)
    except Exception as e:
        out['error'] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    finally:
        os.chdir(cwd)
        try:
            out['figs'] = _figures()
        finally:
            plt.close('all')

    ###################################################################################################################
    # Post-processing
    ###################################################################################################################
    out['duration'] = time.perf_counter() - t0
    out['log'] = log.text()
    out['folders'] = [os.path.join(root, d) for d in _changed(before, _folders(root))]
    if stream is not None:
        stream(out['log'])

    return out
