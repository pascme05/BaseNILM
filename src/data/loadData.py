#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         loadData
# Date:         23.05.2024
# Author:       Dr. Pascal A. Schirmer
# Version:      V.1.0
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Function Description
#######################################################################################################################
"""
This function loads the data from \data using different data formats. The data is processed and formatted into training
testing and validation data.
Inputs:     1) filename:    name of the datafile to be loaded without file extension
            2) setupDat:    includes all simulation variables
            3) method:      method for loading data, e.g. k-fold or transfer learning
            4) train:       if 1 training, if 2 testing, if 0 validation
            5) fold:        number of folds for loading data
Outputs:    1) data:        loaded data
            2) setup:       modified setup files
"""

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================
from src.general.helpFnc import normVal
from src.general.featuresRoll import featuresRoll
from src.general.helpFnc import warnMsg

# ==============================================================================
# External
# ==============================================================================
import pandas as pd
import pickle
import numpy as np
# from nilmtk import DataSet
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import copy
import scipy.io


#######################################################################################################################
# External Functions
#######################################################################################################################
# ==============================================================================
# Excel and CSV
# ==============================================================================
def loadXlsx(filename, data, units):
    try:
        data['X'] = pd.read_excel(filename, sheet_name='input')
        data['y'] = pd.read_excel(filename, sheet_name='output')
        unitsRaw = pd.read_excel(filename, sheet_name='units')
        units['Input'] = pd.DataFrame(columns=data['X'].columns)
        units['Input'].loc[0] = unitsRaw['Input'].dropna().values
        units['Output'] = pd.DataFrame(columns=data['y'].columns)
        units['Output'].loc[0] = unitsRaw['Output'].dropna().values
        print("INFO: Xlsx data file loaded")
    except:
        print("ERROR: Data file could not be loaded")

    return [data, units]


# ==============================================================================
# Matlab
# ==============================================================================
def loadMat(filename, setupDat, data, units):
    try:
        # ------------------------------------------
        # Loading
        # ------------------------------------------
        raw = scipy.io.loadmat(filename)

        # ------------------------------------------
        # Loading
        # ------------------------------------------
        if setupDat['freq'] == 'HF':
            # Init
            W = raw['input'].shape[1] - 2
            F = raw['input'].shape[2]
            temp = np.zeros((raw['input'].shape[0], W * raw['input'].shape[2] + 2))
            temp[:, 0] = raw['output'][:, 0]
            temp[:, 1] = raw['output'][:, 1]

            # Assign
            for i in range(0, raw['input'].shape[2]):
                temp[:, i * W + 2:i * W + 2 + (raw['input'].shape[1] - 2)] = raw['input'][:, 2:, i]

            # Output
            raw['input'] = temp

            # HF Information
            setupDat['HF_W'] = W
            setupDat['HF_F'] = F

        # ------------------------------------------
        # Output axis
        # ------------------------------------------
        if setupDat['dim'] == 3:
            try:
                raw['output'] = raw['output'][:, :, setupDat['outFeat']]
                print("INFO: 3D datafile loaded using axis: %d" % setupDat['outFeat'])
            except:
                print("ERROR: 3D error data file could not be loaded")

        # ------------------------------------------
        # Processing
        # ------------------------------------------
        # Labels
        for i in range(0, len(raw['labelInp'])):
            raw['labelInp'][i] = raw['labelInp'][i].rstrip()
        for i in range(0, len(raw['labelOut'])):
            raw['labelOut'][i] = raw['labelOut'][i].rstrip()

        # Data
        if setupDat['freq'] == 'HF':
            col = ['time', 'id']
            for i in range(0, raw['input'].shape[1] - 2):
                col.append('Inp' + str(i))
            data['X'] = pd.DataFrame(data=raw['input'], columns=col)
        else:
            data['X'] = pd.DataFrame(data=raw['input'], columns=raw['labelInp'])
        data['y'] = pd.DataFrame(data=raw['output'], columns=raw['labelOut'])

        # Units
        units['Input'] = pd.DataFrame(columns=raw['labelInp'])
        units['Input'].loc[0] = raw['unitInp']
        units['Output'] = pd.DataFrame(columns=raw['labelOut'])
        units['Output'].loc[0] = raw['unitOut']

        # Msg
        print("INFO: Mat data file loaded")
    except:
        print("ERROR: Data file could not be loaded")

    return [data, units, setupDat]


# ==============================================================================
# Pickle
# ==============================================================================
def loadPkl(filename, data, units):
    try:
        data, units = pickle.load(open(filename, "rb"))
        print("INFO: Pkl data file loaded")
    except:
        print("ERROR: Data file could not be loaded")

    return [data, units]


# ==============================================================================
# H5 (configured for REDD, adapt manually)
# ==============================================================================
def loadH5(filename, data, units, setupDat):
    try:
        # ------------------------------------------
        # Loading
        # ------------------------------------------
        # raw = DataSet(filename)
        elec = []
        # elec = raw.buildings[setupDat['house']].elec
        app = next(elec[1].load(sample_period=int(1 / setupDat['fs'])))

        # ------------------------------------------
        # Init
        # ------------------------------------------
        # Data
        data['X'] = pd.DataFrame(np.zeros((len(app), 5)), columns=['time', 'id', 'P-agg', 'P1-agg', 'P2-agg'])
        data['y'] = pd.DataFrame(np.zeros((len(app), len(setupDat['out']) + 2)),
                                 columns=pd.concat(['time', 'id', setupDat['out']]))

        # Units
        units['Input'] = pd.DataFrame(columns=data['X'].columns)
        units['Input'].loc[0] = ['sec', '-', 'W', 'W', 'W']
        units['Output'] = pd.DataFrame(columns=data['y'].columns)
        strs = ['W' for x in range(len(setupDat['out']))]
        units['Output'].loc[0] = np.concatenate((['sec', '-', strs]))

        # ------------------------------------------
        # Input data
        # ------------------------------------------
        data['X']['time'] = np.linspace(0, len(app) * int(1 / setupDat['fs']) - int(1 / setupDat['fs']), len(app))
        data['X']['id'] = 1
        data['X']['P1-agg'] = next(elec[1].load(sample_period=int(1 / setupDat['fs'])))
        data['X']['P2-agg'] = next(elec[2].load(sample_period=int(1 / setupDat['fs'])))
        data['X']['P-agg'] = data['X']['P1-agg'] + data['X']['P2-agg']

        # ------------------------------------------
        # Output data
        # ------------------------------------------
        data['y'][setupDat['out'][0]] = next(elec[setupDat['out'][0]].load(sample_period=int(1 / setupDat['fs'])))
        for i in range(1, len(setupDat['out'])):
            data['y'][setupDat['out'][i]] = next(elec[setupDat['out'][i]].load(sample_period=int(1 / setupDat['fs'])))

        # ------------------------------------------
        # Msg
        # ------------------------------------------
        print("INFO: H5 data file loaded")
    except:
        print("ERROR: Data file could not be loaded")

    return [data, units]


#######################################################################################################################
# Function
#######################################################################################################################
def loadData(setupExp, setupDat, setupPar, setupMdl, setupPath, name, method, train, fold):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Loading Dataset")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    shu = setupDat['Shuffle']

    # ==============================================================================
    # Variables
    # ==============================================================================
    data = {}
    units = {}

    # ==============================================================================
    # Path
    # ==============================================================================
    # ------------------------------------------
    # File Extension
    # ------------------------------------------
    name = name + '.' + setupDat['type']
    path = setupPath['datPath']

    # ------------------------------------------
    # Path
    # ------------------------------------------
    if setupDat['folder'] == "":
        filename = pjoin(path, name)
        print("INFO: Loading dataset from head-folder: \data")
    else:
        try:
            filename = pjoin(path, setupDat['folder'], name)
            print("INFO: Loading dataset from sub-folder: \data\ " + str(setupDat['folder']))
        except:
            filename = pjoin(path, name)
            msg = "WARN: Sub-folder not found trying head-folder: \data"
            setupExp = warnMsg(msg, 1, 1, setupExp)

    ###################################################################################################################
    # Loading
    ###################################################################################################################
    # ==============================================================================
    # Excel
    # ==============================================================================
    if setupDat['type'] == 'xlsx' or setupDat['type'] == 'csv':
        [data, units] = loadXlsx(filename, data, units)

    # ==============================================================================
    # Mat-file
    # ==============================================================================
    elif setupDat['type'] == 'mat':
        [data, units, setupDat] = loadMat(filename, setupDat, data, units)

    # ==============================================================================
    # Pkl-file
    # ==============================================================================
    elif setupDat['type'] == 'pkl':
        [data, units] = loadPkl(filename, data, units)

    # ==============================================================================
    # h5-file (tbi)
    # ==============================================================================
    elif setupDat['type'] == 'h5':
        [data, units] = loadH5(filename, data, units, setupDat)

    # ==============================================================================
    # Default
    # ==============================================================================
    else:
        print("ERROR: Data format not available")

    ###################################################################################################################
    # Pre-Processing
    ###################################################################################################################
    # ==============================================================================
    # Selecting Input and Output
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    if len(setupDat['inp']) != 0:
        try:
            inp = copy.deepcopy(setupDat['inp'])
            inp.append('time')
            inp.append('id')
            data['X'] = data['X'][inp]
            units['Input'] = units['Input'][inp]
            print("INFO: Input features selected")
        except:
            print("INFO: Selecting input features failed")

    # ------------------------------------------
    # Output List based
    # ------------------------------------------
    # Input List
    if len(setupDat['out']) != 0 and setupDat['outEnergy'] == 0:
        try:
            out = copy.deepcopy(setupDat['out'])
            setupDat['numOut'] = len(out)
            out.append('time')
            out.append('id')
            data['y'] = data['y'][out]
            units['Output'] = units['Output'][out]
            print("INFO: Output features selected")
        except:
            print("INFO: Selecting output features failed")

    # All Inputs
    elif len(setupDat['out']) == 0 and setupDat['outEnergy'] == 0:
        setupDat['numOut'] = data['y'].shape[1] - 2

    # Energy based
    else:
        # Energy
        energySel = 0
        energy = data['y'].drop(['time', 'id'], axis=1)
        energy = energy.sum(axis=0)
        energy = energy.sort_values(ascending=False)
        energyTotal = energy.sum()

        # Select
        for i in range(0, len(energy)):
            energySel = energySel + energy[i]
            if energySel/energyTotal > setupDat['outEnergy']:
                try:
                    data['y'] = data['y'].drop(energy.index[i], axis=1)
                except:
                    print("INFO: Selecting output features failed")
        setupDat['numOut'] = data['y'].shape[1] - 2

        # Msg
        print("INFO: Selected appliance with a total energy amount of ", str(int(setupDat['outEnergy']*100)), "%")

    # ==============================================================================
    # Limiting
    # ==============================================================================
    if setupDat['lim'] != 0:
        data['X'] = data['X'].head(setupDat['lim'])
        data['y'] = data['y'].head(setupDat['lim'])
        print("INFO: Data limited to ", setupDat['lim'], " samples")
    else:
        print("INFO: Data samples not limited")

    # ==============================================================================
    # Removing Constant Features
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for col in data['X'].columns:
        if np.sum(abs(np.diff(data['X'][col]))) == 0 and col != 'time' and col != 'id' and setupDat['freq'] != 'HF':
            # Calc
            data['X'] = data['X'].drop([col], axis=1)

            # Unit
            units['Input'] = units['Input'].drop([col], axis=1)

            # Warn
            msg = "WARN: Constant column in X data " + str(col) + " will be removed"
            setupExp = warnMsg(msg, 1, 1, setupExp)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for col in data['y'].columns:
        if np.sum(abs(np.diff(data['y'][col]))) == 0 and col != 'time' and col != 'id':
            # Calc
            data['y'] = data['y'].drop([col], axis=1)

            # Unit
            units['Output'] = units['Output'].drop([col], axis=1)

            # Warn
            msg = "WARN: Constant column in X data " + str(col) + " will be removed"
            setupExp = warnMsg(msg, 1, 1, setupExp)

    ###################################################################################################################
    # Calculating
    ###################################################################################################################
    # ==============================================================================
    # 1-Fold
    # ==============================================================================
    if method == 0:
        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            data['X'], _ = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            _, data['X'] = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)
            _, data['y'] = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=shu)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # Split
            X, _ = train_test_split(data['X'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=False)
            y, _ = train_test_split(data['y'], test_size=(1 - setupDat['rT']), random_state=None, shuffle=False)

            # Extract
            data['X'], _ = train_test_split(X, test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(y, test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)

    # ==============================================================================
    # k-Fold
    # ==============================================================================
    elif method == 1:
        # ------------------------------------------
        # Init
        # ------------------------------------------
        kfX = KFold(n_splits=setupExp['kfold'])
        kfX.get_n_splits(data['X'])
        kfy = KFold(n_splits=setupExp['kfold'])
        kfy.get_n_splits(data['y'])

        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            # X
            iter = 0
            for idx, _ in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx, :]
                    break

            # y
            iter = 0
            for idx, _ in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx, :]
                    break

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            # X
            iter = 0
            for idx1, idx2 in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx2, :]
                    break

            # y
            iter = 0
            for idx1, idx2 in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx2, :]
                    break

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # X
            iter = 0
            for idx, _ in kfX.split(data['X']):
                iter = iter + 1
                if iter == fold:
                    data['X'] = data['X'].iloc[idx, :]
                    break

            # y
            iter = 0
            for idx, _ in kfy.split(data['y']):
                iter = iter + 1
                if iter == fold:
                    data['y'] = data['y'].iloc[idx, :]
                    break

            # Extract
            data['X'], _ = train_test_split(data['X'], test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)
            data['y'], _ = train_test_split(data['y'], test_size=(1 - setupDat['rV']), random_state=None, shuffle=shu)

    # ==============================================================================
    # Transfer
    # ==============================================================================
    elif method == 2:
        data['X'] = data['X']
        data['y'] = data['y']

    # ==============================================================================
    # IDs
    # ==============================================================================
    else:
        # ------------------------------------------
        # Init
        # ------------------------------------------
        idTest = setupDat['idT']
        idVal = setupDat['idV']

        # ------------------------------------------
        # Training
        # ------------------------------------------
        if train == 1:
            # Split
            for sel in idTest:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

        # ------------------------------------------
        # Testing
        # ------------------------------------------
        elif train == 2:
            # Init
            idTrain = [sel for sel in data['X']['id'].unique() if sel not in idTest]

            # Split
            for sel in idTrain:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

        # ------------------------------------------
        # Validation
        # ------------------------------------------
        else:
            # Init
            idTrain = [sel for sel in data['X']['id'].unique() if sel not in idVal]

            # Split
            for sel in idTrain:
                data['X'].drop(data['X'][data['X']['id'] == sel].index, inplace=True)
                data['y'].drop(data['y'][data['y']['id'] == sel].index, inplace=True)

    ###################################################################################################################
    # Post-Processing
    ###################################################################################################################
    # ==============================================================================
    # Rolling input features
    # ==============================================================================
    if setupPar['feat'] == 2 or setupPar['feat'] == 3:
        tempTime = copy.deepcopy(data['X']['time'])
        tempID = copy.deepcopy(data['X']['id'])
        data['X'] = featuresRoll(data['X'].drop(['time', 'id'], axis=1), setupMdl)
        data['X']['time'] = tempTime
        data['X']['id'] = tempID

    # ==============================================================================
    # Norm
    # ==============================================================================
    [maxX, maxY, minX, minY, uX, uY, sX, sY] = normVal(data['X'].drop(['time', 'id'], axis=1),
                                                       data['y'].drop(['time', 'id'], axis=1))

    # ==============================================================================
    # Labels
    # ==============================================================================
    setupDat['inpLabel'] = data['X'].columns
    setupDat['inpLabel'] = setupDat['inpLabel'].drop(['time', 'id'])
    setupDat['outLabel'] = data['y'].columns
    setupDat['outLabel'] = setupDat['outLabel'].drop(['time', 'id'])
    setupDat['inpUnits'] = units['Input'].drop(['time', 'id'], axis=1)
    setupDat['outUnits'] = units['Output'].drop(['time', 'id'], axis=1)

    # ==============================================================================
    # Sampling Time
    # ==============================================================================
    setupDat['Ts_raw_X'] = data['X']['time'].iloc[1] - data['X']['time'].iloc[0]
    setupDat['fs_raw_X'] = 1 / setupDat['Ts_raw_X']
    setupDat['Ts_raw_y'] = data['y']['time'].iloc[1] - data['y']['time'].iloc[0]
    setupDat['fs_raw_y'] = 1 / setupDat['Ts_raw_y']

    # ==============================================================================
    # Interpolation
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for names in data['X']:
        if pd.isna(data['X'][names]).any():
            # Calc
            data['X'][names] = data['X'][names].interpolate(limit_direction='both')

            # Msg
            msg = "WARN: NaN in X data column " + str(names) + " detected using interpolation"
            setupExp = warnMsg(msg, 1, 0, setupExp)
            print("WARN: NaN in X data column %s detected using interpolation", names)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        if pd.isna(data['y'][names]).any():
            # Calc
            data['y'][names] = data['y'][names].interpolate(limit_direction='both')

            # Msg
            msg = "WARN: NaN in X data column " + str(names) + " detected using interpolation"
            setupExp = warnMsg(msg, 1, 0, setupExp)
            print("WARN: NaN in y data column %s detected using interpolation", names)

    # ==============================================================================
    # Removing NaN/Inf
    # ==============================================================================
    # ------------------------------------------
    # Input
    # ------------------------------------------
    for names in data['X']:
        data['X'][names] = data['X'][names].fillna(0)
        data['X'][names].replace([np.inf, -np.inf], 0, inplace=True)

    # ------------------------------------------
    # Output
    # ------------------------------------------
    for names in data['y']:
        data['y'][names] = data['y'][names].fillna(0)
        data['y'][names].replace([np.inf, -np.inf], 0, inplace=True)

    # ==============================================================================
    # Normalisation Values
    # ==============================================================================
    setupDat['normMaxX'] = maxX
    setupDat['normMaxY'] = maxY
    setupDat['normMinX'] = minX
    setupDat['normMinY'] = minY
    setupDat['normAvgX'] = uX
    setupDat['normAvgY'] = uY
    setupDat['normVarX'] = sX
    setupDat['normVarY'] = sY

    ###################################################################################################################
    # Return
    ###################################################################################################################
    return [data, setupDat, setupExp]
