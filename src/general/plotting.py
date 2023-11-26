#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         plotting
# Date:         21.11.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################
# Import libs
#######################################################################################################################
# ==============================================================================
# Internal
# ==============================================================================

# ==============================================================================
# External
# ==============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, kstest


#######################################################################################################################
# Function
#######################################################################################################################
def plotting(dataRaw, data, dataPred, resultsAvg, feaScore, feaError, setupDat):
    ###################################################################################################################
    # MSG IN
    ###################################################################################################################
    print("INFO: Start Plotting")

    ###################################################################################################################
    # Initialisation
    ###################################################################################################################
    # ==============================================================================
    # Parameters
    # ==============================================================================
    accLabels1 = ['ACC', 'F1', 'R2', 'TECA']
    accLabels2 = ['RMSE', 'MAE', 'SAE']
    outLabel = setupDat['outLabel']
    inpLabel = dataRaw['T']['X'].columns
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    col_max = 20
    row_max = 7
    cols = len(dataRaw['T']['X'].axes[1])

    # ==============================================================================
    # Variables
    # ==============================================================================
    t = np.linspace(0, len(data['y']), len(data['y'])) / 3600 / setupDat['fs']
    traw = np.linspace(0, len(dataRaw['T']['y']), len(dataRaw['T']['y'])) / 3600 / setupDat['fs']
    SAE = abs(resultsAvg[8] - resultsAvg[7]) / resultsAvg[8]
    accResults = [resultsAvg[0], resultsAvg[1], resultsAvg[2], resultsAvg[3], resultsAvg[4], resultsAvg[5], SAE]

    ###################################################################################################################
    # Preprocessing
    ###################################################################################################################
    # ==============================================================================
    # Limit Cols
    # ==============================================================================
    if row_max > cols:
        row_max = cols

    # ==============================================================================
    # Limit Rows
    # ==============================================================================
    if cols > col_max:
        cols = col_max

    ###################################################################################################################
    # Calculation
    ###################################################################################################################
    # ==============================================================================
    # Input Analysis
    # ==============================================================================
    # ------------------------------------------
    # General
    # ------------------------------------------
    plt.figure()
    txt = "Input Feature Analysis using Distribution, Heatmap, and Feature Ranking"
    plt.suptitle(txt, size=18)
    plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

    # ------------------------------------------
    # Violin Plot
    # ------------------------------------------
    plt.subplot(2, 1, 1)
    df_std = (dataRaw['T']['X'].iloc[:, :cols] - dataRaw['T']['X'].iloc[:, :cols].mean()) / dataRaw['T']['X'].iloc[:, :cols].std()
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(inpLabel, rotation=90)
    plt.grid('on')
    plt.title('Input Feature Distribution')
    plt.xlabel('')

    # ------------------------------------------
    # Heatmap
    # ------------------------------------------
    plt.subplot(2, 2, 3)
    dataHeat = pd.concat([dataRaw['T']['X'].iloc[:, :cols], dataRaw['T']['y']], axis=1)
    sns.heatmap(dataHeat.corr(), annot=True, cmap="Blues")
    plt.title("Heatmap of Input and Output Features")

    # ------------------------------------------
    # Feature Ranking
    # ------------------------------------------
    plt.subplot(2, 2, 4)
    feaScore[:cols].plot.bar(yerr=feaError[:cols])
    plt.title("Feature Ranking using RF")
    plt.ylabel("Mean Accuracy")
    plt.grid('on')

    # ==============================================================================
    # Input Features Time Domain
    # ==============================================================================
    # ------------------------------------------
    # General
    # ------------------------------------------
    plt.figure()
    txt = "Time-domain plots of input features"
    plt.suptitle(txt, size=18)
    plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

    # ------------------------------------------
    # Plotting
    # ------------------------------------------
    for i in range(0, row_max):
        plt.subplot(row_max, 1, i+1)
        plt.plot(traw, dataRaw['T']['X'].iloc[:, i])
        plt.title('Input Feature: ' + inpLabel[i])
        plt.xticks([])
        try:
            plt.ylabel(inpLabel[i] + ' (' + setupDat['inpUnits'][inpLabel[i]][0] + ')')
        except:
            plt.ylabel(inpLabel[i] + ' (-)')
        plt.grid('on')

    # ==============================================================================
    # Average Performance
    # ==============================================================================
    # ------------------------------------------
    # General
    # ------------------------------------------
    plt.figure()
    txt = "Evaluation of Average Performance including Error Distributions"
    plt.suptitle(txt, size=18)
    plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

    # ------------------------------------------
    # Scattering
    # ------------------------------------------
    plt.subplot(2, 2, 1)
    for i in range(0, setupDat['numOut']):
        plt.scatter(data['y'][:, i]/np.max(data['y'][:, i]), dataPred['y'][:, i]/np.max(dataPred['y'][:, i]))
        plt.plot(range(1), range(1))
    plt.grid('on')
    plt.xlabel('True Values ' + '(' + setupDat['outUnits'][outLabel[0]][0] + ')')
    plt.ylabel('Pred Values ' + '(' + setupDat['outUnits'][outLabel[0]][0] + ')')
    plt.title("Scattering Prediction and Residuals")
    plt.legend(outLabel)

    # ------------------------------------------
    # Error Distribution
    # ------------------------------------------
    plt.subplot(2, 2, 3)
    xminG = +np.Inf
    xmaxG = -np.Inf
    for i in range(0, setupDat['numOut']):
        idx = (data['y'][:, i] != 0)
        mu, std = norm.fit((data['y'][idx, i] - dataPred['y'][idx, i]))
        [_, _] = kstest((data['y'][idx, i] - dataPred['y'][idx, i]), 'norm')
        plt.hist((data['y'][idx, i] - dataPred['y'][idx, i]), bins=25, density=True, alpha=0.6)
        xmin, xmax = plt.xlim()
        if xmin < xminG:
            xminG = xmin
        if xmax > xmaxG:
            xmaxG = xmax
        x = np.linspace(xminG, xmaxG, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, linewidth=2, color=colors[i])
    plt.title("Fit Error Gaussian Distribution")
    plt.xlabel('Error ' + '(' + setupDat['outUnits'][outLabel[0]][0] + ')')
    plt.ylabel("Density")
    plt.legend(outLabel)
    plt.grid('on')

    # ------------------------------------------
    # Classical Learning
    # ------------------------------------------
    plt.subplot(2, 2, 2)
    plt.bar(accLabels1, accResults[0:4])
    plt.title('Average Accuracy Values')
    plt.ylabel('Accuracy (%)')
    plt.grid('on')
    plt.subplot(2, 2, 4)
    plt.bar(accLabels2, accResults[4:7])
    plt.title('Average Error Rates')
    plt.ylabel('Error ' + '(' + setupDat['outUnits'][outLabel[0]][0] + ')')
    plt.grid('on')

    # ==============================================================================
    # Temporal Performance
    # ==============================================================================
    for i in range(0, setupDat['numOut']):
        ii = i + 1
        plt.figure()
        plt.subplot(411)
        plt.plot(t, data['y'][:, i])
        plt.plot(t, dataPred['y'][:, i])
        plt.title('Values prediction ' + outLabel[ii - 1])
        plt.xticks([])
        plt.ylabel(outLabel[ii - 1] + ' (' + setupDat['outUnits'][outLabel[ii - 1]][0] + ')')
        plt.grid('on')
        plt.legend(['True', 'Pred'])

        plt.subplot(412)
        plt.plot(t, data['y'][:, i]-dataPred['y'][:, i])
        plt.title('Error prediction ' + outLabel[ii - 1])
        plt.xticks([])
        plt.ylabel(outLabel[ii - 1] + ' (' + setupDat['outUnits'][outLabel[ii - 1]][0] + ')')
        plt.grid('on')

        plt.subplot(413)
        plt.plot(t, 2 * data['L'][:, i], color=colors[0])
        plt.plot(t, dataPred['L'][:, i], color=colors[1])
        plt.title('States labels ' + outLabel[ii - 1])
        plt.xticks([])
        plt.ylabel(outLabel[ii - 1] + ' (On/Off)')
        plt.legend(['True', 'Pred'])
        plt.grid('on')

        plt.subplot(414)
        plt.plot(t, data['L'][:, i] - dataPred['L'][:, i])
        plt.title('Error labels ' + outLabel[ii - 1])
        plt.xlabel('time (hrs)')
        plt.ylabel(outLabel[ii - 1] + ' (On/Off)')
        plt.grid('on')

    plt.show()
