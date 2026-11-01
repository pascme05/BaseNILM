#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation (BaseNILM)
# Topic:        Non-intrusive load monitoring
# File:         help.py
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
This function holds the extended help of the parameters (shown in the question marks of the forms), the parameters of
the basic (non-expert) mode and the conditions under which a parameter or category is used at all. The texts follow
the implementation in src/ (not only the comments of start.py).

HELP:            'Section.var' (flat) or 'Section.dict.var' (feature dictionaries) -> markdown text
BASIC:           parameters shown when the expert mode is off
ACTIVE:          key -> function(cfg) that is False if the parameter has no effect with the current settings
RELEVANT:        section -> {category: function(cfg)} False if the category is not used with the current settings
CATEGORY_TITLES: category -> title of its expander
"""


#######################################################################################################################
# Helper (conditions)
#######################################################################################################################
def _sol(c):
    return c['Par'].get('solver')


def _nn(c):
    return _sol(c) in ('TF', 'PT')


def _frame(c):
    return c['Par'].get('frame') == 1


def _train(c):
    return c['Exp'].get('train') == 1


#######################################################################################################################
# Help texts
#######################################################################################################################
HELP = {
    # ==================================================================================================================
    # Experiment
    # ==================================================================================================================
    'Exp.method':
        "How training and testing data are separated (loadData.py).\n\n"
        "- **0) 1-fold**: the first *Training share* of the (limited) data trains the model, the remaining part tests "
        "it. The validation data are the first *Validation share* of the training part.\n"
        "- **1) k-fold**: k models are trained, each one tests on another fold; the results are averaged (mean and "
        "variance across the folds are printed).\n"
        "- **2) transfer**: train on the *training datasets*, validate on the *validation dataset* and test on the "
        "*testing dataset* (whole files, e.g. other houses).\n"
        "- **3) id based**: the column 'id' (e.g. the house) selects the *test ids* and *validation ids*.",
    'Exp.trainBatch':
        "0) all training data at once; 1) in chunks of *Batch size (data)* samples (for data that do not fit into "
        "memory, the model is retrained per chunk); 2) one training call per id.",
    'Exp.kfold': "Number of folds of the k-fold cross-validation. The run time grows linearly with k (k trainings "
                 "and tests). Typical: 5-10.",
    'Exp.train':
        "1: a new model is trained (an existing model of the same name is loaded and retrained). 0: no training, the "
        "model mdl/mdl_<model>_<name> is loaded for testing — the name of the simulation must match the trained "
        "model.",
    'Exp.test': "1: the model is tested on the testing data and the results (accuracy, F1, TECA, RMSE …) are "
                "computed and shown.",
    # ==================================================================================================================
    # Data
    # ==================================================================================================================
    'Dat.type': "File format of the datasets. MAT and PKL support 2D and 3D outputs; XLSX/CSV only 2D outputs; H5 "
                "(nilmtk) is partly supported. Only files of this format are offered as datasets.",
    'Dat.freq': "LF: low-frequency data (one value per time step, e.g. 1 s … 1 h). HF: high-frequency data, every "
                "time step holds a waveform window (e.g. REDD-3/5 HF); framing is replaced by the HF windows.",
    'Dat.dim': "2: one output feature per appliance. 3: several output features per appliance (e.g. AMPds: P, I, Q, "
               "S); the feature used as target is chosen by *Output feature axis*.",
    'Dat.batch': "Number of samples per training chunk if *Training batches* = 1 (fixed batch size).",
    'Dat.Shuffle': "Shuffles the samples before the split. Off keeps the time order (training before testing), which "
                   "is the realistic setting for NILM.",
    'Dat.rT': "Share of the data used for training with the 1-fold method (0 … 1); the rest is used for testing. "
              "Default 10/12 = 10 months training, 2 months testing for one year of data.",
    'Dat.rV': "Share of the training data used for validation (early stopping, learning-rate reduction). With the "
              "1-fold method the validation data are taken from the beginning of the training part.",
    'Dat.idT': "Ids (column 'id' of the data, e.g. houses) used for testing with the id-based method.",
    'Dat.idV': "Ids used for validation with the id-based method.",
    'Dat.folder': "Sub-folder of data/ that holds the dataset files (empty: data/ itself).",
    'Dat.house': "House number, only used for nilmtk HDF5 (.h5) files.",
    'Dat.train': "Training dataset(s). The 1-fold, k-fold and id methods split the **first** dataset; the transfer "
                 "method trains on all of them one after the other.",
    'Dat.test': "Testing dataset of the transfer method.",
    'Dat.val': "Validation dataset of the transfer method.",
    'Dat.inp': "Input features (columns of X, e.g. P-agg, Q-agg). Empty: all input features. Constant columns are "
               "removed by the loader.",
    'Dat.out': "Appliances to disaggregate (columns of y). Empty: all appliances. More appliances make the problem "
               "harder; small and rarely used loads reduce the average accuracy.",
    'Dat.outFeat': "For 3D data: index of the output feature used as target (AMPds: 0 = active power P, 1 = current "
                   "I, 2 = reactive power Q, 3 = apparent power S).",
    'Dat.outEnergy': "0: the appliances are taken from the list *Appliances*. 0 < x < 1: the largest appliances are "
                     "selected until they cover x of the total energy (e.g. 0.8 = 80 %).",
    'Dat.fs': "Sampling frequency the data are resampled to (Hz). 1/60 = one value per minute. Lower frequencies "
              "shorten the data and the run time but remove short events (kettle, microwave). For HF data this is "
              "the output (y) frequency.",
    'Dat.lim': "Maximum number of samples taken from the start of the data (0: all). 365·24·60 = one year at 1 min. "
               "Use small values (e.g. 50 000) for quick tests.",
    'Dat.weightNorm': "0: every input/output channel is normalised separately. 1: weighted normalisation (common "
                      "scale for all channels).",
    'Dat.inpNorm': "Normalisation of the inputs X: none, −1/+1, 0/1 (min/max), mean/std, quartiles (Q1/Q3) or "
                   "quantile transformation. Mean/std (3) is a robust default for neural networks.",
    'Dat.outNorm': "Normalisation of the outputs y; 0/1 (2) fits the linear output and the clipping to "
                   "*Output minimum/maximum*.",
    'Dat.inpNoise': "Gaussian noise added to the inputs (SNR in dB, 0: none) — robustness tests.",
    'Dat.outNoise': "Gaussian noise added to the outputs (SNR in dB, 0: none).",
    'Dat.inpFil': "Median filter of the inputs (removes spikes).",
    'Dat.outFil': "Median filter of the outputs.",
    'Dat.inpFilLen': "Length of the input median filter (samples, odd).",
    'Dat.outFilLen': "Length of the output median filter (samples, odd).",
    'Dat.threshold': "ON threshold (in the unit of the outputs, e.g. W). Turns the power into ON/OFF states for "
                     "the finite-state metrics accuracy and F1 (postprocess.py) and for the balancing. 0: no "
                     "threshold.",
    'Dat.balance': "0: no balancing. 1: class weights (classification). 2: frames are balanced between ON and OFF "
                   "states using the threshold (regression).",
    'Dat.ghost': "Ghost power = aggregate minus the sum of the selected appliances. 0: ignored, 1: modelled as an "
                 "additional appliance, 2: ideal data (aggregate replaced by the sum of the appliances; only for 1D "
                 "inputs with the same quantity as the outputs).",
    # ==================================================================================================================
    # General
    # ==================================================================================================================
    'Par.method': "0: regression (power of each appliance, linear output). 1: classification (ON/OFF states, sigmoid "
                  "output).",
    'Par.solver': "Family of the model: TF (TensorFlow/Keras), PT (PyTorch), SK (scikit-learn), PM (pattern "
                  "matching: DTW, MVM …), SS (source separation: NMF, DSC), CU (own model in "
                  "trainMdlCU/testMdlCU).",
    'Par.model': "Model of the selected solver. TF: DNN, CNN, CNN2, LSTM, TRA (transformer), DAE, INF (informer). "
                 "SK: RF, SVM, KNN. PM: DTW, MVM, GAK, sDTW. SS: NMF, DSC. The architecture of TF models is shown in "
                 "the view *Model*.",
    'Par.modelInpDim': "3: model input samples × window × features (Conv1D, LSTM, Dense). 4: samples × window × "
                       "features × 1 (Conv2D, model CNN2).",
    'Par.frame': "1: the time series is cut into windows of *Window length* samples (sliding windows). 0: every time "
                 "step is one sample (no temporal context — only sensible with features or SK models).",
    'Par.feat': "Input of the model: 0) raw windows; 1) statistical features per window (mean, RMS, …, see Features "
                "below); 2) rolling features of the input (EWMA, differences); 3) both; 4) 2D features (PQ plane, VI "
                "trajectory, recurrence plot …).",
    'Par.window': "Length of the input window (samples). Longer windows see whole appliance cycles (better for "
                  "long-running loads such as heat pumps) but cost memory and time. At 1 min: 10-60.",
    'Par.overlap': "Overlap of consecutive windows during training (samples). window − 1 gives one window per "
                   "sample (maximum data). −1: no overlap during testing.",
    'Par.outseq': "0: seq2point — one output value per window (at the *Focus point*). x > 0: seq2subseq — x output "
                  "samples in the centre of the window (x = window: seq2seq).",
    'Par.yFocus': "seq2point: sample of the window whose output is predicted (window/2 = centre, typical). −1: the "
                  "average output over the window.",
    'Par.nDim': "Dimension of the input after the feature extraction (adaptDim.py): 1D flattened, 2D (window × "
                "features, default for TF), 3D (window × features × channels).",
    'Par.ranking': "1: the input features are ranked with a random forest after testing (shown under Results).",
    'Par.outMin': "Predictions below this value are clipped (postprocess.py), e.g. 0 W — no negative power.",
    'Par.outMax': "Predictions above this value are clipped, e.g. the rated power of the largest appliance.",
    # ==================================================================================================================
    # Model
    # ==================================================================================================================
    'Mdl.batch': "Mini-batch size of the training (samples per gradient step). Larger batches train faster per "
                 "epoch, smaller ones generalise slightly better. 256-2000.",
    'Mdl.epoch': "Maximum number of training epochs; early stopping ends earlier after *Patience* epochs without "
                 "improvement of the validation loss.",
    'Mdl.patience': "Epochs without improvement of the validation loss before the training stops (the learning rate "
                    "is reduced after half of it).",
    'Mdl.valsteps': "Validation batches evaluated per epoch.",
    'Mdl.shuffle': "Shuffling of the training batches (Keras fit argument, kept as the text 'False'/'True' as in "
                   "mdlPara.py).",
    'Mdl.verbose': "Training output on the console: 0 silent, 1 progress bar, 2 one line per epoch (best for the "
                   "GUI log).",
    'Mdl.loss': "Loss minimised by the training. mae is robust against the large peaks of appliance power; mse "
                "weights large errors more.",
    'Mdl.metric': "Metric shown during training. TECA = total energy correctly assigned (lossMetric in models.py).",
    'Mdl.opt': "Optimiser of the training. Note: trainMdlTF checks the key **'SDG'** for SGD — with 'SGD' it would "
               "fall back to Adam.",
    'Mdl.lr': "Learning rate of the optimiser. Adam: 1e-4 … 1e-3. Too large: unstable training, too small: slow.",
    'Mdl.beta1': "Decay of the first moment (Adam).",
    'Mdl.beta2': "Decay of the second moment (Adam).",
    'Mdl.eps': "Small constant for numerical stability (Adam, RMSprop).",
    'Mdl.rho': "Discounting factor of the gradient history (RMSprop).",
    'Mdl.mom': "Momentum (RMSprop, SGD).",
    'Mdl.SK_RF_depth': "Maximum depth of the trees of the random forest.",
    'Mdl.SK_RF_state': "Random state of the random forest (reproducibility).",
    'Mdl.SK_RF_estimators': "Number of trees of the random forest.",
    'Mdl.SK_SVM_kernel': "Kernel function of the support vector machine.",
    'Mdl.SK_SVM_C': "Regularisation of the SVM (larger: less regularisation).",
    'Mdl.SK_SVM_gamma': "Kernel coefficient of the SVM (rbf, poly, sigmoid).",
    'Mdl.SK_SVM_epsilon': "Width of the ε-insensitive tube of the SVM regression.",
    'Mdl.SK_KNN_neighbors': "Number of neighbours of the k-nearest-neighbour model.",
    'Mdl.SK_KNN_weights': "Weighting of the neighbours: 'uniform' averages the k neighbours equally, 'distance' weighs "
                          "closer neighbours more (1/distance).",
    'Mdl.PM_Gen_cDTW': "Constraint of the pattern matching on the size of the model (share of the training data).",
    'Mdl.PM_DTW_metric': "Distance metric of the warping path of DTW.",
    'Mdl.PM_DTW_const': "Global constraint of the warping path (Sakoe-Chiba band, Itakura parallelogram).",
    'Mdl.PM_GAK_sigma': "Bandwidth of the global alignment kernel.",
    'Mdl.PM_sDTW_gamma': "Smoothing of the soft-DTW alignment (0: hard DTW).",
    'Mdl.PM_MVM_steps': "Number of samples minimum variance matching may skip.",
    'Mdl.PM_MVM_metric': "Distance metric of the minimum variance matching.",
    'Mdl.PM_MVM_const': "Constraint of the minimum variance matching.",
    'Mdl.SS_Gen_lr': "Learning rate of the source separation.",
    'Mdl.SS_DSC_n': "Model order (number of basis functions) of the discriminative sparse coding.",
    'Mdl.feat_roll.EWMA': "Spans of exponentially weighted moving averages added as input features (list, 0: none).",
    'Mdl.feat_roll.EWMS': "Spans of exponentially weighted moving standard deviations (list, 0: none).",
    'Mdl.feat_roll.diff': "1: the first difference of the inputs is added as feature (switching events).",
    'Mdl.feat2D.PQ': "2D image of the PQ plane: 1) product of V and I as input, 2) P and Q as input. Select only one "
                     "2D feature.",
}
_FEAT = {'Mean': 'mean value', 'Std': 'standard deviation', 'RMS': 'RMS value', 'Peak2Rms': 'peak-to-RMS ratio',
         'Median': 'median', 'Min': 'minimum', 'Max': 'maximum', 'Per25': '25 % percentile', 'Per75': '75 % percentile',
         'Energy': 'energy (sum)', 'Var': 'variance', 'Range': 'range (max − min)', '3rdMoment': 'skewness',
         '4thMoment': 'kurtosis'}
for _k, _v in _FEAT.items():
    HELP[f'Mdl.feat.{_k}'] = f"1: the {_v} of every window and input channel is a feature (features1D.py)."
for _k, _v in {'VI': 'VI trajectory', 'REC': 'recurrence plot', 'GAF': 'Gramian angular field',
               'MKF': 'Markov transition field'}.items():
    HELP[f'Mdl.feat2D.{_k}'] = f"1: the {_v} of each window is the 2D input (features2D.py). Select only one."
HELP['Mdl.feat2D.DFIA'] = "FFT image of each window: 1) amplitudes, 2) phases, 3) amplitudes, phases and absolute " \
                          "values. Select only one 2D feature."

#######################################################################################################################
# Basic parameters (expert mode off)
#######################################################################################################################
BASIC = {
    'Exp.method', 'Exp.kfold', 'Exp.train', 'Exp.test',
    'Dat.rT', 'Dat.idT', 'Dat.idV', 'Dat.train', 'Dat.test', 'Dat.val', 'Dat.out', 'Dat.inp', 'Dat.fs', 'Dat.lim',
    'Dat.inpNorm', 'Dat.outNorm', 'Dat.threshold',
    'Par.method', 'Par.solver', 'Par.model', 'Par.frame', 'Par.feat', 'Par.window', 'Par.outseq', 'Par.yFocus',
    'Par.outMax',
    'Mdl.batch', 'Mdl.epoch', 'Mdl.patience', 'Mdl.loss', 'Mdl.opt', 'Mdl.lr',
    'Mdl.SK_RF_depth', 'Mdl.SK_RF_estimators', 'Mdl.SK_SVM_kernel', 'Mdl.SK_SVM_C', 'Mdl.SK_KNN_neighbors',
    'Mdl.PM_DTW_metric', 'Mdl.PM_MVM_steps', 'Mdl.SS_DSC_n',
} | {f'Mdl.feat.{k}' for k in _FEAT} | {'Mdl.feat2D.PQ', 'Mdl.feat2D.VI', 'Mdl.feat2D.REC', 'Mdl.feat2D.GAF',
                                       'Mdl.feat2D.MKF', 'Mdl.feat2D.DFIA', 'Mdl.feat_roll.EWMA',
                                       'Mdl.feat_roll.EWMS', 'Mdl.feat_roll.diff'}

#######################################################################################################################
# Activity rules
#######################################################################################################################
MODELS = {'TF': ['DNN', 'CNN', 'CNN2', 'LSTM', 'TRA', 'DAE', 'INF'], 'PT': ['DNN', 'CNN', 'LSTM'],
          'SK': ['RF', 'SVM', 'KNN'], 'PM': ['DTW', 'MVM', 'GAK', 'sDTW'], 'SS': ['NMF', 'DSC'], 'CU': []}
ACTIVE = {
    'Exp.kfold': lambda c: c['Exp']['method'] == 1,
    'Exp.trainBatch': _train,
    'Dat.batch': lambda c: c['Exp'].get('trainBatch') == 1,
    'Dat.rT': lambda c: c['Exp']['method'] == 0,
    'Dat.rV': lambda c: c['Exp']['method'] in (0, 1) and _train(c),
    'Dat.Shuffle': lambda c: c['Exp']['method'] in (0, 1),
    'Dat.idT': lambda c: c['Exp']['method'] == 3,
    'Dat.idV': lambda c: c['Exp']['method'] == 3,
    'Dat.test': lambda c: c['Exp']['method'] == 2,
    'Dat.val': lambda c: c['Exp']['method'] == 2,
    'Dat.house': lambda c: c['Dat']['type'] == 'h5',
    'Dat.outFeat': lambda c: c['Dat']['dim'] == 3,
    'Dat.out': lambda c: not c['Dat'].get('outEnergy'),
    'Dat.inpFilLen': lambda c: c['Dat']['inpFil'] != 0,
    'Dat.outFilLen': lambda c: c['Dat']['outFil'] != 0,
    'Par.window': _frame,
    'Par.overlap': _frame,
    'Par.outseq': _frame,
    'Par.yFocus': lambda c: _frame(c) and c['Par'].get('outseq', 0) == 0,
    'Par.modelInpDim': _nn,
    'Mdl.beta1': lambda c: c['Mdl']['opt'] == 'Adam',
    'Mdl.beta2': lambda c: c['Mdl']['opt'] == 'Adam',
    'Mdl.rho': lambda c: c['Mdl']['opt'] == 'RMSprop',
    'Mdl.mom': lambda c: c['Mdl']['opt'] in ('RMSprop', 'SDG'),
    'Mdl.eps': lambda c: c['Mdl']['opt'] in ('Adam', 'RMSprop'),
    'Mdl.feat2D.VI': lambda c: c['Par']['feat'] == 4, 'Mdl.feat2D.PQ': lambda c: c['Par']['feat'] == 4,
}

#######################################################################################################################
# Categories
#######################################################################################################################
_SK = lambda c: _sol(c) == 'SK'                                       # noqa: E731
_PM = lambda c: _sol(c) == 'PM'                                       # noqa: E731
_SS = lambda c: _sol(c) == 'SS'                                       # noqa: E731
RELEVANT = {
    'Mdl': {
        'Hyperparameters': lambda c: _sol(c) in ('TF', 'PT', 'CU'),
        'Solver Parameters': lambda c: _sol(c) in ('TF', 'PT', 'CU'),
        'Sklearn': _SK,
        'Pattern matching': _PM,
        'Source separation': _SS,
        'feat_roll': lambda c: c['Par']['feat'] in (2, 3),
        'feat': lambda c: c['Par']['feat'] in (1, 3),
        'feat2D': lambda c: c['Par']['feat'] == 4,
    },
}
# Sub-headings inside a category (schema 'sub'): used only by the selected model -> hidden in basic mode otherwise
SUB_ACTIVE = {sub: (lambda m: (lambda c: c['Par']['model'] == m))(sub)
              for sub in ('RF', 'SVM', 'KNN', 'DTW', 'GAK', 'sDTW', 'MVM', 'NMF', 'DSC')}
CATEGORY_TITLES = {'Sklearn': 'Sklearn (RF, SVM, KNN)', 'Pattern matching': 'Pattern matching (DTW, GAK, sDTW, MVM)',
                   'Source separation': 'Source separation (NMF, DSC)', 'feat_roll': 'Rolling features (input based)', 'feat': 'Statistical features (frame based)',
                   'feat2D': '2D features (select only one)', 'Hyperparameters': 'Training (TF / PT)',
                   'Solver Parameters': 'Loss and optimiser (TF / PT)', 'Input/ Output Mapping': 'Inputs and outputs',
                   'Training/Testing': 'Training and testing', 'Datasets': 'Datasets', 'Pre-processing':
                   'Pre-processing', 'Framing and Features': 'Framing and features', 'Postprocessing':
                   'Post-processing'}


#######################################################################################################################
# Helper
#######################################################################################################################
def key(section, p):
    """'Section.var' for flat parameters, 'Section.dict.var' for the nested feature dictionaries."""
    return f"{section}.{p['cat']}.{p['var']}" if p.get('nested') else f"{section}.{p['var']}"


def is_basic(section, p):
    return not BASIC or key(section, p) in BASIC


def is_active(section, p, cfg):
    rule = ACTIVE.get(key(section, p))
    if rule is None:
        return True
    try:
        return bool(rule(cfg))
    except (KeyError, TypeError, ValueError):
        return True
