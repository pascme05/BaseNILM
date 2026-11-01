#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         tutorials.py
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
Tutorials of the GUI (page Help › Tutorials, shown when TUTORIALS is not empty): guided scenarios per operating mode.
Every tutorial is a config (config/tutorials/<id>.json) made of a base config and 'changes' {'Sec.cat.var': value}
(run settings: '<RUN_SECTION>.var'). Write the configs with 'python -m gui.core.tutorials'. The 'look' texts must
come from runs of the tutorial configs — never invent the numbers.

Entry: {'id', 'section' (one of SECTIONS), 'mode', 'level' (Beginner | Intermediate | Advanced), 'minutes', 'base' (config name, default
'default'), 'title', 'story', 'goals' [..], 'changes' {..}, 'steps' [..], 'look' (markdown), 'exercise' [..]}
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
from gui import adapter as A
from gui.core.config import CONFIG_DIR, from_json, to_json, default_config, read_config

import copy
import os

TUT_DIR = os.path.join(CONFIG_DIR, 'tutorials')
RUN = {}                                                              # run settings of all tutorials (e.g. save off)

#######################################################################################################################
# Tutorials
#######################################################################################################################
# The tutorials are the start files of setup/tutorials (written to config/tutorials_<file>.json by build_schema.py);
# the 'changes' are computed against config/default.json on import. The 'look' texts describe what to inspect —
# reference numbers are not given because the tutorials run 5-fold on the full data (long CPU runs).
_TEXT = [
    # id, mode, level, minutes, title, story, goals, steps, look, exercise
    ('Tutorial_1_AMPds', 0, 'Beginner', 60, "Benchmark on AMPds2",
     "You want a first reference result of the default CNN on AMPds2 with the appliances that make up 80 % of the "
     "energy, evaluated with 5-fold cross-validation.",
     ["select appliances by energy share", "run a k-fold evaluation", "read TECA, F1 and the fold spread"],
     ["Open *Input data › Appliances*: the blue bars are the appliances that cover 80 % of the energy.",
      "Check *Model*: one input feature (P-agg), window 30 → input 30 × 1.",
      "Run the simulation (5 trainings) and open *Results › Folds*."],
     "The large loads (heat pump, dryer, fridge) reach high TECA; the spread between the folds shows the seasonal "
     "dependence of the heat pump.",
     ["Switch to 1-fold (method 0) and compare the run time and the scores.",
      "Use P-agg and Q-agg as inputs."]),
    ('Tutorial_1_ECO', 0, 'Beginner', 45, "Benchmark on ECO house 2",
     "Same benchmark on a Swiss household (ECO, 1 min).",
     ["switch datasets", "compare datasets"],
     ["Look at the energy shares under *Input data*.", "Run and compare the scores with the AMPds tutorial."],
     "Small consumer electronics dominate ECO house 2; expect lower average scores than for AMPds.",
     ["Try house 1 (eco1_1min)."]),
    ('Tutorial_1_REDD', 0, 'Beginner', 30, "Benchmark on REDD house 2",
     "Same benchmark on REDD house 2 (US, two weeks of data).", ["work with short datasets"],
     ["Check the duration under *Input data › Overview*.", "Run the 5-fold simulation."],
     "Few weeks of data: each fold holds only a few activations of the dishwasher and the washer — the scores "
     "vary strongly between folds.", ["Reduce the folds to 3 and compare."]),
    ('Tutorial_1_REFIT', 0, 'Beginner', 60, "Benchmark on REFIT house 2",
     "Same benchmark on REFIT house 2 (UK, 1 min, two years).", ["work with long datasets"],
     ["Check the split under *Input data › Split*.", "Run the 5-fold simulation."],
     "Long data with nine appliances; kettle and microwave are short events at 1 min resolution.",
     ["Limit the samples to one year and compare."]),
    ('Tutorial_2_ML', 0, 'Intermediate', 60, "Machine learning (CNN)",
     "Deep learning baseline for the comparison of model families: CNN on the active power with balanced frames.",
     ["understand balancing", "read the model architecture"],
     ["Open *Model* to see the CNN layers and parameters.", "Run the simulation."],
     "Reference for the pattern matching and source separation tutorials.", ["Switch the model to LSTM."]),
    ('Tutorial_2_PM', 0, 'Intermediate', 60, "Pattern matching (MVM)",
     "Minimum variance matching instead of a neural network (seq2seq on 60-sample windows).",
     ["use pattern matching", "use seq2seq outputs"],
     ["See *Model* (no network, parameters of MVM).", "Check *Input data › Pre-processing & framing*: the target is "
      "the whole window."], "Pattern matching needs no training epochs but is slow at test time.",
     ["Try DTW instead of MVM."]),
    ('Tutorial_2_SS', 0, 'Advanced', 60, "Source separation (DSC)",
     "Discriminative sparse coding on daily windows (1440 samples).", ["use source separation"],
     ["Note window = outseq = 1440 (one day).", "Run the simulation."],
     "Source separation models the whole day at once — good for energy shares, weaker for single events.",
     ["Change the model order SS_DSC_n."]),
    ('Tutorial_3_HF', 0, 'Advanced', 60, "High-frequency data (REDD-3 HF)",
     "Disaggregation from current/voltage waveforms with a 2D CNN.", ["use HF data", "4D model inputs"],
     ["Check *Input data › Overview* (HF features × samples).", "See *Model*: Conv2D with 4D input."],
     "Waveform features separate appliances with similar active power.", ["Compare with Tutorial 3 LF."]),
    ('Tutorial_3_LF', 0, 'Intermediate', 45, "Low-frequency data (REDD-3 LF)",
     "The same house and appliances as the HF tutorial with low-frequency data.", ["compare LF and HF"],
     ["Run and compare with the HF tutorial."], "The LF model sees only the power trajectory.",
     ["Change the window length."]),
    ('Tutorial_4_Seq2Point', 0, 'Intermediate', 45, "Seq2point",
     "Fridge disaggregation, one output value per 60-sample window.", ["understand seq2point"],
     ["Look at the framing preview (*Input data › Pre-processing & framing*): the focus point marks the target."],
     "One prediction per window position.", ["Move the focus point (yFocus)."]),
    ('Tutorial_4_Seq2Seq', 0, 'Intermediate', 45, "Seq2seq",
     "Fridge disaggregation, the whole window as output.", ["understand seq2seq"],
     ["Check the target sequence in the framing preview and the output shape under *Model*."],
     "The output layer has 60 units (one per sample of the window).", ["Compare with seq2point."]),
    ('Tutorial_4_Seq2SubSeq', 0, 'Intermediate', 45, "Seq2subseq",
     "Fridge disaggregation, the centre 30 samples of the window as output.", ["understand seq2subseq"],
     ["Check the shaded target sequence in the framing preview."],
     "A compromise between context and number of outputs.", ["Change outseq to 10."]),
    ('Tutorial_5_Feat_1D', 0, 'Intermediate', 45, "Statistical features (CNN)",
     "Frame-based statistical features (RMS, median, percentiles …) instead of raw windows.", ["use 1D features"],
     ["Enable features under *Model* parameters (Features · statistical).", "See the new input shape under *Model*."],
     "Features compress the window; the model input shrinks.", ["Switch single features on and off."]),
    ('Tutorial_5_Feat_1D_ML', 0, 'Intermediate', 30, "Statistical features (random forest)",
     "The same features with a random forest (scikit-learn).", ["use SK models"],
     ["Look at *Model* (RF parameters)."], "Fast training, no GPU needed.", ["Change the number of trees."]),
    ('Tutorial_5_Feat_2D', 0, 'Advanced', 60, "2D features",
     "PQ-plane images of each window as input of a 2D CNN.", ["use 2D features"],
     ["See *Model*: Conv2D on images.", "Note: Training is off in this file — switch it on for a first run."],
     "2D features need P and Q inputs.", ["Try the VI trajectory."]),
    ('Tutorial_5_Raw_1D', 0, 'Beginner', 45, "Raw data, one input",
     "Raw windows of the active power only.", ["compare inputs"], ["Run and note the scores."],
     "Reference for the raw 2D tutorial.", ["Add Q-agg as input."]),
    ('Tutorial_5_Raw_2D', 0, 'Beginner', 45, "Raw data, all inputs",
     "Raw windows of all inputs (P, I, Q, S).", ["compare inputs"], ["Run and compare with raw 1D."],
     "More inputs help for appliances with reactive power (motors, compressors).", ["Use only P and Q."]),
    ('Tutorial_6_Transfer', 0, 'Advanced', 30, "Transfer learning (REDD)",
     "Train on REDD house 1, validate on house 3 and test on house 2 (fridge, 3 s data).",
     ["use the transfer method", "check generalisation to an unseen house"],
     ["Check *Input data* for redd1 and redd2.", "Run the simulation."],
     "The fridge transfers well between houses; other appliances generalise worse.", ["Train on redd1 and redd3."]),
]


def _flat(c, p=''):
    out = {}
    for k, v in c.items():
        if k == '_meta':
            continue
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def _build():
    import json
    try:
        base = _flat(default_config())
    except Exception:
        return []
    out = []
    for tid, mode, level, minutes, title, story, goals, steps, look, exercise in _TEXT:
        f = os.path.join(CONFIG_DIR, f"tutorials_{tid}.json")
        if not os.path.isfile(f):
            continue
        with open(f, 'r', encoding='utf-8') as fh:
            cfg = _flat(json.load(fh))
        # sim is set by the mode; gpu = 1 of the files stops trainMdlTF on machines without GPU (kept at 0)
        changes = {k: v for k, v in cfg.items() if base.get(k) != v and k not in (f"{A.RUN_SECTION}.sim",
                                                                                   f"{A.RUN_SECTION}.gpu")}
        out.append({'id': tid, 'mode': mode, 'level': level, 'minutes': minutes, 'base': 'default', 'title': title,
                    'story': story + f" (setup/tutorials/{tid}.py)", 'goals': goals, 'changes': changes,
                    'steps': steps, 'look': look, 'exercise': exercise})
    return out


# Tutorials of the optimisation modes (GUI only, no start file): changes against config/default.json
_GUI = [
    {'id': 'Opti_1_Hyper_TF', 'mode': 1, 'level': 'Intermediate', 'minutes': 10, 'base': 'default',
     'title': "Hyperparameter optimisation with Keras-Tuner",
     'story': "You do not know how many convolution layers, filters and which learning rate suit AMPds2. Keras-Tuner "
              "Hyperband trains many small CNNs for a few epochs, keeps the best third and continues them "
              "(optiHyp.py → trainMdlTFopti.py, search space tfMdloptiRCNN in models.py).",
     'goals': ["run a Hyperband search", "read the trials and the parallel-coordinates plot",
               "transfer the result to the configuration"],
     'changes': {'Exp.name': 'Opti_Hyper_CNN', 'Exp.train': 1, 'Exp.plot': 0, 'Dat.lim': 50000, 'Mdl.epoch': 10,
                 'Mdl.patience': 4},
     'steps': ["Check *Inputs*: 50,000 samples (about 35 days) keep the search short.",
               "Open *Run*: the search trains about 30 small models with at most 10 epochs each (≈ 5–10 min on a "
               "CPU). Use *Quick check* first (1 epoch) to see that everything works.",
               "Open *Results*: the bar chart ranks the trials by their validation loss, the parallel-coordinates "
               "plot shows which hyperparameters the good trials share."],
     'look': "Look for hyperparameters on which the good trials agree (e.g. the learning rate) while others vary "
             "freely — the agreeing ones matter for this dataset, the others hardly. The final fit of the best "
             "trial shows at which epoch the validation loss was lowest.",
     'exercise': ["Increase the epochs to 30 and compare the best validation loss.",
                  "Apply the best learning rate (button) and run a normal simulation with it."]},
    {'id': 'Opti_2_Hyper_SK', 'mode': 1, 'level': 'Intermediate', 'minutes': 5, 'base': 'default',
     'title': "Grid search of a random forest (scikit-learn)",
     'story': "A random forest is a fast alternative to the CNN. GridSearchCV fits every combination of depth, "
              "minimum split size and number of trees five times (5-fold cross-validation, trainMdlSKopti.py).",
     'goals': ["run GridSearchCV", "understand cross-validation scores", "apply the best parameters"],
     'changes': {'Exp.name': 'Opti_Hyper_RF', 'Exp.train': 1, 'Exp.plot': 0, 'Par.solver': 'SK', 'Par.model': 'RF',
                 'Dat.lim': 5000},
     'steps': ["Check *Parameters › Model › Sklearn*: the current RF settings (depth, trees).",
               "Run: 27 combinations × 5 folds = 135 fits on 5,000 samples (≈ 2 min).",
               "Open *Results* and press *Apply to the configuration*."],
     'look': "The CV score is the R² of each fold — the folds are consecutive time blocks, so a fold with an "
             "appliance that was rarely on in the other folds can give a negative R². Compare the ranking, not the "
             "absolute value.",
     'exercise': ["Switch to KNN (Par › Model) and rerun.", "Run a simulation with the applied parameters."]},
    {'id': 'Opti_3_Grid_Window_fs', 'mode': 2, 'level': 'Intermediate', 'minutes': 15, 'base': 'default',
     'title': "Grid optimisation: window length × sampling frequency",
     'story': "The window of the CNN covers window / fs of the aggregate. Is it better to see a long time span "
              "coarsely or a short one in detail? The grid repeats training and testing for every combination "
              "(as optiGrid.py, reduced to 3 × 3).",
     'goals': ["select two grid variables", "read the heatmap", "relate the window to the appliance cycles"],
     'changes': {'Exp.name': 'Opti_Grid_W_fs', 'Exp.train': 1, 'Exp.plot': 0, 'Dat.lim': 100000, 'Mdl.epoch': 5,
                 'Grid.var1': 'Par.window', 'Grid.values1': [10, 20, 40], 'Grid.var2': 'Dat.fs',
                 'Grid.values2': [1 / 60, 1 / 300, 1 / 900], 'Grid.couple': True},
     'steps': ["Open *Parameters › Grid*: variable 1 is the window length, variable 2 the sampling frequency.",
               "Run (9 trainings of 5 epochs, ≈ 10–15 min) — or *Quick check* with 2 × 2 points first.",
               "Open *Results*: switch the metric between TECA and F1 and look at single appliances."],
     'look': "Check the appliances one by one: appliances with long cycles (heat pump, dryer) should profit from a "
             "long time span per window, short events (dishwasher heating phases) from a high sampling frequency. "
             "The best average cell is the compromise of both.",
     'exercise': ["Add the window 80.", "Replace the sampling frequency by the learning rate (Mdl.lr)."]},
    {'id': 'Opti_4_Grid_lr_batch', 'mode': 2, 'level': 'Advanced', 'minutes': 10, 'base': 'default',
     'title': "Grid optimisation: learning rate × batch size",
     'story': "Training hyperparameters interact: large batches need larger learning rates. The grid shows the "
              "combinations that converge within a fixed number of epochs.",
     'goals': ["sweep training hyperparameters", "relate the scores to the training curves"],
     'changes': {'Exp.name': 'Opti_Grid_lr_batch', 'Exp.train': 1, 'Exp.plot': 0, 'Dat.lim': 100000,
                 'Mdl.epoch': 5, 'Grid.var1': 'Mdl.lr', 'Grid.values1': [1e-2, 1e-3, 1e-4], 'Grid.var2': 'Mdl.batch',
                 'Grid.values2': [256, 1000, 4096], 'Grid.couple': False},
     'steps': ["Open *Parameters › Grid* and check the two variables.", "Run (9 trainings, ≈ 10 min).",
               "Compare the heatmap with the run time per grid point."],
     'look': "With only 5 epochs, combinations of a small learning rate and a large batch may not converge "
             "(low TECA) — compare the heatmap with the run time per grid point, which drops with the batch size.", 'exercise': ["Raise the epochs to 20 and compare the heatmap."]},
]


# Quick start (GUI only): the default configuration with the pretrained models mdl/mdl_<model>_Default.weights.h5
# (AMPds2, window 21, 1-fold). Training is off, so a run only loads the model and tests it (about a minute).
_QUICK_MDL = {
    'DNN': ("Feed-forward network (DNN)",
            "three dense layers of 32 units on the flattened window — the smallest and fastest of the three models",
            "The DNN has no notion of the order of the samples in the window; it recognises steady power levels "
            "well, but transitions and short events less sharply than the CNN and LSTM."),
    'CNN': ("Convolutional network (CNN)",
            "five 1D convolution layers that learn local patterns (edges, plateaus) in the window, followed by "
            "three dense layers",
            "The convolutions detect the switching edges of the appliances; expect sharper on/off transitions than "
            "with the DNN."),
    'LSTM': ("Recurrent network (LSTM)",
             "LSTM layers that read the window sample by sample and keep a memory of the trajectory",
             "The LSTM follows the time evolution of the load; it is the slowest of the three models at test time."),
}
_QUICK = [
    {'id': f"Quick_{i}_{m}", 'section': 'Quick start', 'mode': 0, 'level': 'Beginner', 'minutes': 1,
     'base': 'default', 'title': f"Pretrained {title}",
     'story': f"You want to see BaseNILM working without waiting for a training. This tutorial uses the default "
              f"setup (AMPds2, five appliances, 1-fold split) with the pretrained {m}: {desc}. Training is switched "
              f"off, so the run only loads mdl/mdl_{m}_Default.weights.h5 and disaggregates the test data "
              f"(about 15–30 s on a CPU).",
     'goals': ["run a simulation with a pretrained model", "read the disaggregation results",
               "compare the default model families"],
     'changes': {f"{A.RUN_SECTION}.train": 0, f"{A.RUN_SECTION}.test": 1, f"{A.RUN_SECTION}.method": 0,
                 f"{A.RUN_SECTION}.name": 'Default', 'Par.model': m, 'Par.window': 21, 'Par.overlap': 20,
                 'Par.yFocus': 10},
     'steps': ["Open *Model*: check the layers of the network — the input is a window of 21 samples.",
               "Run the simulation: no training, the pretrained weights are loaded and tested.",
               "Open *Results*: compare the predicted and true appliance power and the scores per appliance."],
     'look': look + " Keep the experiment name 'Default' and the window of 21 samples — the pretrained weights "
                    "only fit this setup.",
     'exercise': ["Load the other two quick-start tutorials and compare the scores.",
                  "Switch training on (*Run settings*), change the experiment name and train your own model."]}
    for i, (m, (title, desc, look)) in enumerate(_QUICK_MDL.items(), start=1)]

SECTIONS = ['Quick start', 'Toolkit', 'Optimisation']                 # sections of the tutorials page (in order)
TUTORIALS = _QUICK + [dict(t, section='Toolkit') for t in _build()] + [dict(t, section='Optimisation') for t in _GUI]


#######################################################################################################################
# Functions
#######################################################################################################################
def merge_changes(cfg, t):
    """Config of a tutorial: base config + run settings + changes 'Section.cat.var' (or '<RUN_SECTION>.var')."""
    cfg = copy.deepcopy(cfg)
    cfg[A.RUN_SECTION].update(RUN)
    for k, v in t['changes'].items():
        *path, last = k.split('.')
        d = cfg
        for p in path:
            d = d.setdefault(p, {})
        d[last] = copy.deepcopy(v)
    if A.MODE_VAR:
        cfg[A.RUN_SECTION][A.MODE_VAR] = t['mode']
    return cfg


def base(t):
    return default_config() if t.get('base', 'default') == 'default' else read_config(t['base'])


def config_file(tid):
    return os.path.join(TUT_DIR, f"{tid}.json")


def get(tid):
    return next(t for t in TUTORIALS if t['id'] == tid)


def load(tid):
    """Config of a tutorial (config/tutorials/<id>.json, generated from base and changes if the file is missing)."""
    f = config_file(tid)
    if os.path.isfile(f):
        with open(f, 'r', encoding='utf-8') as fh:
            return from_json(fh.read())
    return merge_changes(base(get(tid)), get(tid))


def write_all():
    """Writes config/tutorials/<id>.json for all tutorials."""
    os.makedirs(TUT_DIR, exist_ok=True)
    for t in TUTORIALS:
        with open(config_file(t['id']), 'w', encoding='utf-8') as fh:
            fh.write(to_json(merge_changes(base(t), t)))


if __name__ == '__main__':
    write_all()
