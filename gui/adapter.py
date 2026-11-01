#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         adapter.py
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
Connection of the GUI to the BaseNILM toolkit. BaseNILM has no config workbook: its parameters are the commented
assignments of start.py (setupExp, setupDat, setupPar) and mdlPara.py (setupMdl). gui/build_schema.py parses these
files into the schema; the configuration is a dict {'Exp': .., 'Dat': .., 'Par': .., 'Mdl': ..} that is handed to
main()/optiHyp()/optiGrid() directly (FEED='dict'). main() returns nothing, so run() records the test results by
wrapping the functions called in src/test.py.
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
import contextlib
import copy
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#######################################################################################################################
# Names
#######################################################################################################################
NAME = 'BaseNILM'                                                     # short name (page title, JSON meta)
TITLE = 'BaseNILM toolkit for energy disaggregation'                  # full name (Setup page)
TAGLINE = "Choose the dataset here, then open an operating mode in the sidebar. All parameters of start.py and " \
          "mdlPara.py are edited in the GUI and stored as JSON config (sidebar); the data are read with the loaders " \
          "of the toolkit."
SRC_PATHS = ['', 'src']                                               # added to sys.path (relative to ROOT)
LOGO = None
DOC = 'docs/GUI.md'                                                   # user guide shown on the Documentation page

#######################################################################################################################
# Parameter sources (start file | Name | Category | ... replaced by the Python setup files)
#######################################################################################################################
CONFIG_DIR = 'config'                                                 # folder of the JSON configs
CONFIG_WORKBOOK = None                                                # no workbook: schema from the Python sources
START_FILE = 'start.py'                                               # source of setupExp / setupDat / setupPar
MDL_FILE = 'mdlPara.py'                                               # source of setupMdl
SETUP_DIRS = ['setup/default', 'setup/tutorials', 'setup/benchmark', 'setup/journal']   # -> ready-to-load configs
# setup dict of the toolkit -> (section of the GUI config, title)
SECTIONS = {'setupExp': ('Exp', 'Experiment'), 'setupDat': ('Dat', 'Data'), 'setupPar': ('Par', 'General'),
            'setupMdl': ('Mdl', 'Model')}
SECTION_TITLES = {**{v[0]: v[1] for v in SECTIONS.values()}, 'Grid': 'Grid'}
SHEETS = {}                                                           # unused (no workbook)
INTS = {('Par', 'nDim'), ('Par', 'modelInpDim'), ('Dat', 'inpFilLen'), ('Dat', 'outFilLen'), ('Par', 'outseq'),
        ('Par', 'window'), ('Par', 'overlap')}
STRS = set()
NO_OPTIONS = {('Dat', 'outEnergy'), ('Dat', 'threshold'), ('Par', 'outseq'), ('Par', 'yFocus'), ('Par', 'overlap')}
OPTIONAL = set()
_ONOFF = {0: 'off', 1: 'on'}
# options the comments do not express (or express in a form that does not parse); keys (section, var) or
# (section, cat, var) for the nested feature dictionaries
OPTIONS = {
    ('Dat', 'type'): {'xlsx': 'Excel workbook', 'csv': 'CSV file', 'mat': 'MATLAB file', 'pkl': 'Pickle file',
                      'h5': 'nilmtk HDF5 (partly supported)', 'parquet': 'Parquet file'},
    ('Dat', 'freq'): {'LF': 'low-frequency data', 'HF': 'high-frequency data'},
    ('Par', 'solver'): {'TF': 'TensorFlow', 'PT': 'PyTorch', 'SK': 'scikit-learn', 'PM': 'Pattern matching',
                        'SS': 'Source separation', 'CU': 'Custom (placeholder)'},
    ('Par', 'model'): {'DNN': 'DNN (TF/PT)', 'CNN': 'CNN (TF/PT)', 'CNN2': 'CNN 2D (TF)', 'LSTM': 'LSTM (TF/PT)',
                       'TRA': 'Transformer (TF)', 'DAE': 'Denoising auto-encoder (TF)', 'INF': 'Informer (TF)',
                       'RF': 'Random forest (SK)', 'SVM': 'Support vector machine (SK)', 'KNN': 'k-nearest neighbours '
                       '(SK)', 'DTW': 'Dynamic time warping (PM)', 'MVM': 'Minimum variance matching (PM)',
                       'GAK': 'Global alignment kernel (PM)', 'sDTW': 'Soft DTW (PM)', 'NMF': 'Non-negative matrix '
                       'factorisation (SS)', 'DSC': 'Discriminative sparse coding (SS)'},
    ('Par', 'modelInpDim'): {3: '3D input (samples × window × features)', 4: '4D input (e.g. CNN2D)'},
    ('Par', 'nDim'): {1: '1D', 2: '2D', 3: '3D'},
    ('Mdl', 'shuffle'): {'False': 'no shuffling', 'True': 'shuffling'},
    ('Mdl', 'loss'): {'mae': 'mean absolute error', 'mse': 'mean squared error',
                      'BinaryCrossentropy': 'binary cross-entropy', 'KLDivergence': 'Kullback-Leibler divergence'},
    ('Mdl', 'metric'): {'mae': 'mean absolute error', 'mse': 'mean squared error', 'BinaryCrossentropy':
                        'binary cross-entropy', 'KLDivergence': 'Kullback-Leibler divergence', 'accuracy': 'accuracy',
                        'TECA': 'total energy correctly assigned'},
    # trainMdlTF checks the key 'SDG' for stochastic gradient descent; 'SGD' would fall back to Adam
    ('Mdl', 'opt'): {'Adam': 'Adam', 'RMSprop': 'RMSprop', 'SDG': 'SGD (key SDG as checked in trainMdlTF)'},
    ('Mdl', 'SK_SVM_kernel'): {'linear': 'linear', 'poly': 'polynomial', 'rbf': 'radial basis function',
                               'sigmoid': 'sigmoid'},
    ('Mdl', 'SK_KNN_weights'): {'uniform': 'uniform (all neighbours equal)', 'distance': 'distance (closer weigh more)'},
    ('Mdl', 'PM_DTW_const'): {'none': 'none', 'sakoechiba': 'Sakoe-Chiba band', 'itakura': 'Itakura parallelogram'},
    ('Mdl', 'PM_MVM_const'): {'none': 'none', 'sakoechiba': 'Sakoe-Chiba band', 'itakura': 'Itakura parallelogram'},
    ('Mdl', 'PM_DTW_metric'): {'euclidean': 'euclidean', 'cityblock': 'city block'},
    ('Mdl', 'PM_MVM_metric'): {'euclidean': 'euclidean', 'cityblock': 'city block'},
    ('Mdl', 'feat2D', 'PQ'): {0: 'off', 1: 'product of V and I', 2: 'P and Q as input'},
    ('Mdl', 'feat2D', 'DFIA'): {0: 'off', 1: 'FFT amplitudes', 2: 'FFT phases', 3: 'amplitudes, phases, absolute'},
}
for _v in ['Mean', 'Std', 'RMS', 'Peak2Rms', 'Median', 'Min', 'Max', 'Per25', 'Per75', 'Energy', 'Var', 'Range',
           '3rdMoment', '4thMoment']:
    OPTIONS[('Mdl', 'feat', _v)] = _ONOFF
for _v in ['VI', 'REC', 'GAF', 'MKF']:
    OPTIONS[('Mdl', 'feat2D', _v)] = _ONOFF
OPTIONS[('Mdl', 'feat_roll', 'diff')] = _ONOFF
# readable names of the parameters (the variable is shown in the help)
LABELS = {
    ('Exp', 'method'): 'Validation method', ('Exp', 'trainBatch'): 'Training batches', ('Exp', 'kfold'): 'Folds (k)',
    ('Exp', 'train'): 'Training', ('Exp', 'test'): 'Testing',
    ('Dat', 'type'): 'File format', ('Dat', 'freq'): 'Data type', ('Dat', 'dim'): 'Data dimension',
    ('Dat', 'batch'): 'Batch size (data)', ('Dat', 'Shuffle'): 'Shuffle split', ('Dat', 'rT'): 'Training share',
    ('Dat', 'rV'): 'Validation share', ('Dat', 'idT'): 'Test ids', ('Dat', 'idV'): 'Validation ids',
    ('Dat', 'folder'): 'Dataset folder', ('Dat', 'house'): 'House (h5)', ('Dat', 'train'): 'Training dataset(s)',
    ('Dat', 'test'): 'Testing dataset', ('Dat', 'val'): 'Validation dataset', ('Dat', 'inp'): 'Input features',
    ('Dat', 'out'): 'Appliances', ('Dat', 'outFeat'): 'Output feature axis', ('Dat', 'outEnergy'): 'Energy share',
    ('Dat', 'fs'): 'Sampling frequency', ('Dat', 'lim'): 'Sample limit', ('Dat', 'weightNorm'): 'Normalisation mode',
    ('Dat', 'inpNorm'): 'Input normalisation', ('Dat', 'outNorm'): 'Output normalisation',
    ('Dat', 'inpNoise'): 'Input noise', ('Dat', 'outNoise'): 'Output noise', ('Dat', 'inpFil'): 'Input filter',
    ('Dat', 'outFil'): 'Output filter', ('Dat', 'inpFilLen'): 'Input filter length',
    ('Dat', 'outFilLen'): 'Output filter length', ('Dat', 'threshold'): 'ON threshold', ('Dat', 'balance'): 'Balancing',
    ('Dat', 'ghost'): 'Ghost data',
    ('Par', 'method'): 'Problem type', ('Par', 'solver'): 'Solver', ('Par', 'model'): 'Model',
    ('Par', 'modelInpDim'): 'Model input dimension', ('Par', 'frame'): 'Framing', ('Par', 'feat'): 'Features',
    ('Par', 'window'): 'Window length', ('Par', 'overlap'): 'Overlap', ('Par', 'outseq'): 'Output sequence',
    ('Par', 'yFocus'): 'Focus point', ('Par', 'nDim'): 'Feature dimension', ('Par', 'ranking'): 'Feature ranking',
    ('Par', 'outMin'): 'Output minimum', ('Par', 'outMax'): 'Output maximum',
    ('Mdl', 'batch'): 'Batch size', ('Mdl', 'epoch'): 'Epochs', ('Mdl', 'patience'): 'Patience',
    ('Mdl', 'valsteps'): 'Validation steps', ('Mdl', 'shuffle'): 'Shuffle batches', ('Mdl', 'verbose'): 'Verbosity',
    ('Mdl', 'loss'): 'Loss function', ('Mdl', 'metric'): 'Metric', ('Mdl', 'opt'): 'Optimiser',
    ('Mdl', 'lr'): 'Learning rate', ('Mdl', 'beta1'): 'β1 (Adam)', ('Mdl', 'beta2'): 'β2 (Adam)',
    ('Mdl', 'eps'): 'ε (stability)', ('Mdl', 'rho'): 'ρ (RMSprop)', ('Mdl', 'mom'): 'Momentum',
    ('Mdl', 'SK_RF_depth'): 'Max. tree depth', ('Mdl', 'SK_RF_state'): 'Random state',
    ('Mdl', 'SK_RF_estimators'): 'Trees', ('Mdl', 'SK_SVM_kernel'): 'Kernel', ('Mdl', 'SK_SVM_C'): 'C (regularisation)',
    ('Mdl', 'SK_SVM_gamma'): 'γ (kernel)', ('Mdl', 'SK_SVM_epsilon'): 'ε (tube)', ('Mdl', 'SK_KNN_neighbors'):
    'Neighbours', ('Mdl', 'SK_KNN_weights'): 'Neighbour weighting', ('Mdl', 'PM_Gen_cDTW'): 'Model size constraint', ('Mdl', 'PM_DTW_metric'): 'Warping metric',
    ('Mdl', 'PM_DTW_const'): 'Warping constraint', ('Mdl', 'PM_GAK_sigma'): 'σ (kernel)',
    ('Mdl', 'PM_sDTW_gamma'): 'γ (soft alignment)', ('Mdl', 'PM_MVM_steps'): 'Skippable steps',
    ('Mdl', 'PM_MVM_metric'): 'Metric', ('Mdl', 'PM_MVM_const'): 'Constraint', ('Mdl', 'SS_Gen_lr'): 'Learning rate',
    ('Mdl', 'SS_DSC_n'): 'Model order',
    ('Mdl', 'feat_roll', 'EWMA'): 'EWMA spans', ('Mdl', 'feat_roll', 'EWMS'): 'EWMS spans',
    ('Mdl', 'feat_roll', 'diff'): 'Difference', ('Mdl', 'feat', 'Peak2Rms'): 'Peak/RMS',
    ('Mdl', 'feat', 'Per25'): '25 % percentile', ('Mdl', 'feat', 'Per75'): '75 % percentile',
    ('Mdl', 'feat', '3rdMoment'): 'Skewness', ('Mdl', 'feat', '4thMoment'): 'Kurtosis',
    ('Mdl', 'feat2D', 'PQ'): 'PQ plane', ('Mdl', 'feat2D', 'VI'): 'VI trajectory', ('Mdl', 'feat2D', 'REC'):
    'Recurrence plot', ('Mdl', 'feat2D', 'GAF'): 'Gramian angular field', ('Mdl', 'feat2D', 'MKF'):
    'Markov transition field', ('Mdl', 'feat2D', 'DFIA'): 'FFT image',
}
PERCENT_AS_FRACTION = False
EXTRA_CONFIGS = {}

#######################################################################################################################
# Run settings (setupExp of start.py)
#######################################################################################################################
RUN_SECTION = 'Exp'                                                   # section of setup holding the run settings
CONFIG_VAR = None
MODE_VAR = 'sim'                                                      # 0) simulation, 1) hyperparameters, 2) grid
RUN = {'name': 'Default', 'author': 'Pascal Schirmer', 'sim': 0, 'gpu': 0, 'warn': 3, 'save': 0, 'log': 0, 'plot': 1}
# (var, kind, label, help): kind 'text', 'toggle', 'int', 'file:<folder key>', 'select:a|b|c', 'expert:<kind>'
RUN_FIELDS = [
    ('name', 'text', "Simulation name", "Name of the experiment. It names the trained model mdl/mdl_<model>_<name> "
                                        "(loaded again when 'train' is off) and the saved results."),
    ('author', 'text', "Author", "Author of the simulation (written with the saved results)."),
    ('plot', 'toggle', "Render figures", "The matplotlib figures of the toolkit (plotting.py) are shown under Results "
                                         "› Figures. The interactive charts of the GUI do not need it."),
    ('save', 'toggle', "Save results", "Write the predictions and the result table to results/ (save.py)."),
    ('log', 'expert:toggle', "Log input data", "Logging of the input data (setupExp['log'])."),
    ('gpu', 'expert:select:0|1', "GPU", "0: CPU, 1: GPU (only if TensorFlow / PyTorch see a GPU)."),
    ('warn', 'expert:select:0|1|2|3', "TensorFlow log level", "0: all messages, 1: no INFO, 2: no INFO/WARN, "
                                                              "3: disabled (TF_CPP_MIN_LOG_LEVEL)."),
]

#######################################################################################################################
# Operating modes
#######################################################################################################################
MODES = {
    0: dict(title='Simulation', section='Exp', extra=['Dat', 'Par', 'Mdl'], icon=':material/model_training:',
            text="Training and/or testing of one disaggregation model on the selected dataset: 1-fold with a time "
                 "split, k-fold cross-validation, transfer between datasets or id-based splits (main.py).",
            sub=('Exp', 'Training/Testing', 'method')),
    1: dict(title='Hyperparameter optimisation', section='Exp', extra=['Dat', 'Par', 'Mdl'],
            icon=':material/tune:',
            text="Keras-Tuner Hyperband (TF: CNN search space of models.py) or GridSearchCV (SK: RF, SVM, KNN) of the "
                 "model hyperparameters on the training dataset (optiHyp.py).", sub=None),
    2: dict(title='Grid optimisation', section='Exp', extra=['Grid', 'Dat', 'Par', 'Mdl'], icon=':material/grid_on:',
            text="Repeats the simulation (training + testing) over a grid of one or two parameters — e.g. window "
                 "length × sampling frequency as optiGrid.py — and compares the scores of all grid points.",
            sub=None),
}

#######################################################################################################################
# Files
#######################################################################################################################
FOLDERS = {'data': 'data', 'mdl': 'mdl'}                              # input folders: key -> folder (rel. ROOT)
FILE_EXT = ('.mat', '.pkl', '.xlsx', '.csv', '.h5', '.parquet')
DATA_EXT = ('mat', 'pkl', 'xlsx', 'csv', 'h5', 'parquet')
FILE_GROUPS = []
GROUP_SELECTOR = None
RESULTS_DIR = 'results'
RESULT_PROJECTS = False                                               # BaseNILM writes results/<name> flat
BIG_FILE = 150e6                                                      # data files above (bytes) are loaded on request


def results_root(cfg):
    """Folder in which a run creates its results folder (new sub-folders are detected after the run)."""
    return os.path.join(ROOT, RESULTS_DIR)


def run_folder(cfg):
    """Folder of a GUI run that wrote no results itself (save off): results/<name> as save.py would use."""
    name = ''.join('_' if c in '<>:"/\\|?*' else c for c in str(cfg[RUN_SECTION].get('name', 'Default'))) or 'Default'
    if cfg.get('_quick'):
        name += '_check'
    folder = os.path.join(ROOT, RESULTS_DIR, name)
    os.makedirs(folder, exist_ok=True)
    return folder


#######################################################################################################################
# Toolkit
#######################################################################################################################
FEED = 'dict'                                                         # main() receives the four setup dicts


def init_setup():
    """Empty setup of the toolkit as GUI config {'Exp': setupExp, 'Dat': .., 'Par': .., 'Mdl': ..}."""
    from src.general.helpFnc import initSetup
    exp, dat, par, mdl = initSetup()
    return {'Exp': exp, 'Dat': dat, 'Par': par, 'Mdl': mdl}


def paths():
    """Path variable of the toolkit; called with ROOT as working directory (initPath uses os.getcwd())."""
    from src.general.helpFnc import initPath
    return initPath(os.path.basename(ROOT))


def cast(setup):
    """Types as in start.py / mdlPara.py: integers, lists, the string 'False' of setupMdl['shuffle']."""
    for sec in ('Exp', 'Dat', 'Par', 'Mdl'):
        for k, v in list(setup[sec].items()):
            if isinstance(v, float) and v.is_integer() and (sec, k) in INTS:
                setup[sec][k] = int(v)
    setup['Dat']['Shuffle'] = bool(setup['Dat'].get('Shuffle', False))
    setup['Mdl']['shuffle'] = str(setup['Mdl'].get('shuffle', 'False'))
    for k in ('inp', 'out', 'train', 'idT', 'idV'):
        v = setup['Dat'].get(k)
        if v is None:
            setup['Dat'][k] = []
        elif not isinstance(v, list):
            setup['Dat'][k] = [v]
    return setup


def split(setup):
    """The four setup dicts of the toolkit (deep copies, the GUI config stays untouched)."""
    s = copy.deepcopy(setup)
    return s['Exp'], s['Dat'], s['Par'], s['Mdl']


#######################################################################################################################
# Grid (GUI loop over main(), cfg['Grid'])
#######################################################################################################################
GRID_DEFAULT = {'var1': 'Par.window', 'values1': [4, 8, 16, 32, 64], 'var2': 'Dat.fs',
                'values2': [1 / 60, 1 / 300, 1 / 600, 1 / 900, 1 / 1800, 1 / 3600], 'couple': True,
                'keep_models': False}
GRID_PRESETS = {'Par.window': [4, 8, 16, 32, 64], 'Dat.fs': [1 / 60, 1 / 300, 1 / 600, 1 / 900, 1 / 1800, 1 / 3600],
                'Mdl.lr': [1e-2, 1e-3, 1e-4], 'Mdl.epoch': [5, 10, 20], 'Mdl.batch': [256, 1000, 4096],
                'Par.outseq': [0, 5, 10], 'Dat.inpNorm': [0, 1, 2, 3], 'Dat.outNorm': [0, 1, 2, 3],
                'Mdl.SK_RF_depth': [3, 5, 10], 'Mdl.SK_RF_estimators': [16, 32, 64], 'Mdl.SK_KNN_neighbors': [3, 5, 10],
                'Mdl.patience': [5, 15, 30], 'Dat.inpFilLen': [3, 7, 15], 'Par.model': ['DNN', 'CNN', 'LSTM']}


def grid_of(setup):
    """Grid settings of a config (defaults filled in)."""
    return {**GRID_DEFAULT, **(setup.get('Grid') or {})}


def set_param(dicts, ref, val):
    """Sets 'Sec.var' (e.g. 'Par.window') in the dicts {'Exp': .., 'Dat': .., 'Par': .., 'Mdl': ..}."""
    sec, var = ref.split('.', 1)
    dicts[sec][var] = int(val) if (sec, var) in INTS and isinstance(val, float) and val.is_integer() else val


def couple_window(dicts):
    """Framing consistent with the window as in optiGrid.py: overlap = W-1, focus point = W/2."""
    W = int(dicts['Par']['window'])
    dicts['Par']['overlap'] = W - 1
    dicts['Par']['yFocus'] = W // 2
    if dicts['Par'].get('outseq', 0) > W:
        dicts['Par']['outseq'] = W


#######################################################################################################################
# Quick check (Run view): one epoch, few samples, small grids; trained models get the suffix _check
#######################################################################################################################
QUICK = {'epoch': 1, 'lim': 20000, 'kfold': 2, 'grid': 2}


def quick(exp, dat, par, mdl, mode):
    """Overrides of a quick check run (in place)."""
    mdl['epoch'] = QUICK['epoch'] if mode != 1 else max(QUICK['epoch'], 1)
    dat['lim'] = min(int(dat.get('lim') or QUICK['lim']), QUICK['lim'])
    exp['kfold'] = min(int(exp.get('kfold') or 2), QUICK['kfold'])
    if exp.get('train') == 1 or mode in (1, 2):                      # never overwrite a trained model
        exp['name'] = f"{exp.get('name', 'Default')}_check"


#######################################################################################################################
# Run
#######################################################################################################################
METRICS = ['ACC', 'F1', 'R2', 'TECA', 'RMSE', 'MAE', 'MAX', 'EST', 'TRUTH']   # columns of performance() (printResults)


def _label_list(x):
    try:
        return [str(v) for v in list(x)]
    except TypeError:
        return []


def run(setup, path, mode):
    """
    Runs main() / optiHyp() / optiGrid() and records the test results of every call of src.test.test():
    {'tests': [{'labels', 'result' (numOut x 9), 'avg' (9), 'yTrue', 'yPred', 'fs', 'name', 'feat'}],
     'train_logs': {name: DataFrame}, 'model': .., 'mode': ..}
    """
    import src.test as T
    import src.main as M

    exp, dat, par, mdl = split(setup)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(exp.get('warn', 3))
    exp['sim'] = int(mode)
    is_quick = bool(setup.get('_quick'))
    if is_quick:
        quick(exp, dat, par, mdl, mode)
        print(f"INFO: Quick check: {mdl['epoch']} epoch, at most {dat['lim']:,} samples, name {exp['name']}")
    rec = {'tests': [], 'train_logs': {}, 'mode': mode, 'model': par.get('model'), 'solver': par.get('solver'),
           'quick': is_quick, 'name': exp.get('name')}
    cur = {}

    # ------------------------------------------
    # Wrappers (recorded as deep copies, the toolkit keeps its own data)
    # ------------------------------------------
    orig = {'performance': T.performance, 'printResults': T.printResults, 'featureRank': T.featureRank}

    def performance(yPred, yTrue, yPred_L, yTrue_L, setupDat):
        cur['yPred'] = np.array(yPred, dtype=float, copy=True)
        cur['yTrue'] = np.array(yTrue, dtype=float, copy=True)
        cur['labels'] = _label_list(setupDat.get('outLabel', []))
        cur['fs'] = float(setupDat.get('fs', 0) or 0)
        return orig['performance'](yPred, yTrue, yPred_L, yTrue_L, setupDat)

    def featureRank(X, y, setupDat, setupMdl, setupPar):
        out = orig['featureRank'](X, y, setupDat, setupMdl, setupPar)
        try:
            cur['feat'] = (copy.deepcopy(out[0]), copy.deepcopy(out[1]))
        except Exception:
            cur['feat'] = None
        return out

    def printResults(resultsApp, resultsAvg, setupDat, setupExp):
        orig['printResults'](resultsApp, resultsAvg, setupDat, setupExp)
        if 'yPred' in cur:                                            # only the calls of test() (not k-fold summary)
            if mode == 2:                                             # grid: many runs, key figures only
                cur['yTrue'], cur['yPred'] = cur['yTrue'][:0], cur['yPred'][:0]
            rec['tests'].append({'grid': cur.get('grid'),
                                 'labels': cur.get('labels') or _label_list(setupDat.get('outLabel', [])),
                                 'result': np.array(resultsApp, dtype=float), 'avg': np.array(resultsAvg, dtype=float),
                                 'yTrue': cur.pop('yTrue'), 'yPred': cur.pop('yPred'), 'fs': cur.get('fs', 0.0),
                                 'feat': cur.pop('feat', None), 'name': str(setupExp.get('name', '')),
                                 'warnings': copy.deepcopy(setupExp.get('status', {}))})
        else:
            rec.setdefault('summaries', []).append({'result': np.array(resultsApp, dtype=float),
                                                    'avg': np.array(resultsAvg, dtype=float)})

    T.performance, T.printResults, T.featureRank = performance, printResults, featureRank
    try:
        if mode == 1:
            from src.optiHyp import optiHyp
            optiHyp(exp, dat, par, mdl, path)
            rec['opti'] = _read_opti(exp, par)
        elif mode == 2:
            _run_grid(setup, exp, dat, par, mdl, path, rec, cur, is_quick)
        else:
            M.main(exp, dat, par, mdl, path)
    finally:
        T.performance, T.printResults, T.featureRank = orig['performance'], orig['printResults'], orig['featureRank']

    # ------------------------------------------
    # Training logs of the TF models (mdl/mdl_<model>_<name>/training_log.csv)
    # ------------------------------------------
    if exp.get('train') == 1 and par.get('solver') == 'TF' and mode == 0:
        import glob
        import pandas as pd
        base = str(exp.get('name', ''))
        for f in glob.glob(os.path.join(ROOT, 'mdl', f"mdl_{par.get('model')}_{base}*", 'training_log.csv')):
            try:
                rec['train_logs'][os.path.basename(os.path.dirname(f))] = pd.read_csv(f)
            except Exception:
                pass
    return rec


def _read_opti(exp, par):
    """Result of the hyperparameter optimisation saved by trainMdlTFopti / trainMdlSKopti (JSON)."""
    import json
    f = os.path.join(ROOT, RESULTS_DIR, str(exp.get('name')), f"opti_{par.get('model')}_{exp.get('name')}.json")
    if not os.path.isfile(f):
        return None
    with open(f, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _run_grid(setup, exp, dat, par, mdl, path, rec, cur, is_quick):
    """Simulation over the grid cfg['Grid'] (one or two parameters), one main() call per grid point."""
    import shutil
    import time
    import src.main as M
    G = grid_of(setup)
    v1 = list(G['values1'] or [])
    v2 = list(G['values2'] or []) if G.get('var2') else [None]
    if is_quick:
        v1, v2 = v1[:QUICK['grid']], v2[:QUICK['grid']]
    base = exp['name']
    rec['grid'] = {'var1': G['var1'], 'values1': v1, 'var2': G.get('var2'), 'values2': v2 if G.get('var2') else [],
                   'points': []}
    n = len(v1) * len(v2)
    for i, a in enumerate(v1):
        for j, b in enumerate(v2):
            d = {'Exp': copy.deepcopy(exp), 'Dat': copy.deepcopy(dat), 'Par': copy.deepcopy(par),
                 'Mdl': copy.deepcopy(mdl)}
            set_param(d, G['var1'], a)
            if G.get('var2'):
                set_param(d, G['var2'], b)
            if G.get('couple', True):
                couple_window(d)
            d['Exp'].update(name=f"{base}_G{i}_{j}", sim=0, plot=0)
            k = len(rec['grid']['points']) + 1
            print(f"\n========== Grid point {k}/{n}: {G['var1']} = {a}"
                  + (f", {G['var2']} = {b}" if G.get('var2') else '') + " ==========\n")
            cur['grid'] = (i, j)
            t0, err, n0 = time.time(), None, len(rec['tests'])
            try:
                M.main(d['Exp'], d['Dat'], d['Par'], d['Mdl'], path)
            except (Exception, SystemExit) as e:                     # one failing point does not stop the grid
                err = f"{type(e).__name__}: {e}"
                print(f"ERROR: Grid point {k} failed: {err}")
            rec['grid']['points'].append({'i': i, 'j': j, 'v1': a, 'v2': b, 'time': time.time() - t0, 'error': err,
                                          'test': n0 if len(rec['tests']) > n0 else None,
                                          'name': d['Exp']['name']})
            if not G.get('keep_models', False):
                for f in os.listdir(os.path.join(ROOT, 'mdl')):
                    if f.startswith(f"mdl_{d['Par']['model']}_{d['Exp']['name']}"):
                        p = os.path.join(ROOT, 'mdl', f)
                        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
    cur.pop('grid', None)


def summary(cfg, mode):
    """Rows of the run summary (Run view)."""
    E, D, P, M = cfg['Exp'], cfg['Dat'], cfg['Par'], cfg['Mdl']
    meth = {0: '1-fold', 1: f"{E.get('kfold')}-fold", 2: 'transfer', 3: 'id based'}.get(E.get('method'), '?')
    todo = ' + '.join([s for s, on in (('training', E.get('train') == 1), ('testing', E.get('test') == 1)) if on])
    folder = D.get('folder') or ''
    rows = [('Simulation', f"{E.get('name')}  ·  {todo or 'nothing (train and test off)'}  ·  {meth}"),
            ('Dataset', f"data/{folder + '/' if folder else ''}{', '.join(map(str, D.get('train') or []))}."
                        f"{D.get('type')}" + (f"  →  test {D.get('test')}" if E.get('method') == 2 else '')),
            ('Appliances', ', '.join(map(str, D.get('out') or [])) if not D.get('outEnergy') else
             f"largest up to {100 * D['outEnergy']:g} % of the energy"),
            ('Model', f"{P.get('solver')} · {P.get('model')} · window {P.get('window')} · fs {D.get('fs'):.4g} Hz"),
            ('Model file', f"mdl/mdl_{P.get('model')}_{E.get('name')}")]
    if P.get('solver') in ('TF', 'PT') and E.get('train') == 1:
        rows.append(('Training', f"≤ {M.get('epoch')} epochs · batch {M.get('batch')} · {M.get('opt')} lr "
                                 f"{M.get('lr'):g}"))
    if mode == 1:
        rows.append(('Search', "Keras-Tuner Hyperband over the CNN search space (tfMdloptiRCNN), objective val_loss"
                     if P.get('solver') == 'TF' else f"GridSearchCV of {P.get('model')} (5-fold CV)"))
    if mode == 2:
        G = grid_of(cfg)
        rows.append(('Grid', f"{G['var1']} = {G['values1']}" + (f"  ×  {G['var2']} = "
                                                                 f"{[round(v, 6) if isinstance(v, float) else v for v in G['values2']]}"
                                                                 if G.get('var2') else '')))
    return rows


_GPU = []


def _has_gpu():
    """True if TensorFlow sees a GPU (checked once)."""
    if not _GPU:
        try:
            import tensorflow as tf
            _GPU.append(bool(tf.config.list_physical_devices('GPU')))
        except Exception:
            _GPU.append(False)
    return _GPU[0]


def warnings(cfg, mode):
    """[(level, text)] shown above the run button."""
    E, D, P = cfg['Exp'], cfg['Dat'], cfg['Par']
    out = []
    if int(E.get('gpu') or 0) == 1 and P.get('solver') in ('TF', 'PT', 'CU') and not _has_gpu():
        out.append(('info', "GPU is switched on (Setup › GPU = 1) but TensorFlow sees no GPU — the run falls back to "
                            "the CPU."))
    model = os.path.join(ROOT, 'mdl', f"mdl_{P.get('model')}_{E.get('name')}")
    if mode == 0 and E.get('train') != 1 and E.get('test') == 1 and P.get('solver') == 'TF' and \
            not os.path.isfile(model + '.weights.h5') and E.get('method') != 1:
        out.append(('error', f"Testing without training needs the trained model mdl/mdl_{P.get('model')}_"
                             f"{E.get('name')}.weights.h5, which does not exist — switch on Training or use the name "
                             f"of a trained model."))
    if mode == 0 and E.get('train') != 1 and E.get('test') != 1:
        out.append(('warning', "Training and testing are both off — the run does nothing."))
    if mode == 1 and P.get('solver') not in ('TF', 'SK'):
        out.append(('error', f"The hyperparameter optimisation supports the solvers TF and SK, not {P.get('solver')}."))
    if mode == 1 and P.get('solver') == 'SK':
        out.append(('info', "GridSearchCV fits every combination of its parameter grid 5 times (5-fold CV) — RF: 27 "
                            "combinations; reduce the sample limit for a first test."))
    if mode == 2:
        G = grid_of(cfg)
        n = max(len(G['values1']), 1) * (max(len(G['values2']), 1) if G.get('var2') else 1)
        if E.get('train') != 1:
            out.append(('error', "Training is off: every grid point needs its own model — switch on Training "
                                 "(Parameters › Experiment)."))
        out.append(('warning', f"The grid runs {n} complete simulations ({len(G['values1'])} × "
                               f"{len(G['values2']) if G.get('var2') else 1}), each with training and testing."))
    if E.get('method') == 1 and E.get('train') == 1:
        out.append(('info', f"k-fold trains {E.get('kfold')} models — the run takes about {E.get('kfold')} times "
                            f"as long as a 1-fold run."))
    if not D.get('lim') or D.get('lim', 0) > 2e5:
        if E.get('train') == 1 and P.get('solver') in ('TF', 'PT'):
            out.append(('info', "Training on the full data can take long on a CPU; reduce the sample limit or the "
                                "epochs for a first test."))
    return out


def compact(data, mode, max_points=20000):
    """Thins the test time series of a run before it is stored (gui_run.pkl)."""
    if not isinstance(data, dict):
        return data
    out = copy.deepcopy(data)
    for t in out.get('tests', []):
        n = len(t['yTrue'])
        if n > max_points:
            s = int(np.ceil(n / max_points))
            t['yTrue'], t['yPred'], t['step'] = t['yTrue'][::s], t['yPred'][::s], s
    return out


@contextlib.contextmanager
def session():
    """Context of a run: warnings off (as in start.py)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield
