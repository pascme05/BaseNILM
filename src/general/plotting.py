#######################################################################################################################
#######################################################################################################################
# Title:        BaseNILM toolkit for energy disaggregation
# Topic:        Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File:         plotting
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
This function plots the time domain results as well as the accuracy results.
"""

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
import os


#######################################################################################################################
# Function
#######################################################################################################################
def plotting(dataRaw, data, dataPred, resultsAvg, feaScore, feaError, setupDat, setupPar, setupExp):
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
    plt.grid(True)
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
    plt.grid(True)

    # ==============================================================================
    # Input Features Time Domain
    # ==============================================================================
    # ------------------------------------------
    # Plotting
    # ------------------------------------------
    fig, axs = plt.subplots(row_max, 1, sharex=True)
    txt = "Time-domain plots of input features"
    plt.suptitle(txt, size=18)
    plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

    for i in range(0, row_max):
        axs[i].plot(traw, dataRaw['T']['X'].iloc[:, i])
        axs[i].set_title('Input Feature: ' + inpLabel[i])
        axs[i].grid(True)
        try:
            axs[i].set_ylabel(inpLabel[i] + ' (' + setupDat['inpUnits'][inpLabel[i]][0] + ')')
        except:
            axs[i].set_ylabel(inpLabel[i] + ' (-)')

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
    plt.grid(True)
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
    plt.grid(True)

    # ------------------------------------------
    # Classical Learning
    # ------------------------------------------
    plt.subplot(2, 2, 2)
    plt.bar(accLabels1, accResults[0:4])
    plt.title('Average Accuracy Values')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.bar(accLabels2, accResults[4:7])
    plt.title('Average Error Rates')
    plt.ylabel('Error ' + '(' + setupDat['outUnits'][outLabel[0]][0] + ')')
    plt.grid(True)

    # ==============================================================================
    # Convergence
    # ==============================================================================
    try:
        # ------------------------------------------
        # Loading
        # ------------------------------------------
        save_dir = 'mdl/mdl_' + setupPar['model'] + '_' + setupExp['name']
        log_file = os.path.join(save_dir, 'training_log.csv')
        df = pd.read_csv(log_file)

        # ------------------------------------------
        # Plotting
        # ------------------------------------------
        # Plot training and validation loss and accuracy with logarithmic y-axis
        plt.figure()
        txt = "Convergence for Training and Validation and Learning Rate Update"
        plt.suptitle(txt, size=18)
        plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

        # Change Epoch
        df['epoch'] = np.linspace(1, df['epoch'].shape[0], df['epoch'].shape[0])

        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(df['epoch'], df['loss'], label='Training Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Metric plot (assuming 'accuracy' is the metric)
        plt.subplot(1, 3, 2)
        plt.plot(df['epoch'], df.iloc[:, 3], label='Training Accuracy')
        plt.plot(df['epoch'], df.iloc[:, 5], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.yscale('log')
        plt.title('Training and Validation Metric')
        plt.legend()
        plt.grid(True)

        # Learning rate plot
        plt.subplot(1, 3, 3)
        plt.plot(df['epoch'], df['learning_rate'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate')
        plt.grid(True)

    except:
        print("WARN: Could not load training log file.")

    # ==============================================================================
    # Temporal Performance
    # ==============================================================================
    for i in range(0, setupDat['numOut']):
        ii = i + 1
        fig, axs = plt.subplots(4, 1, sharex=True)
        txt = "Power Consumption and State Prediction for Appliance: " + outLabel[ii - 1]
        plt.suptitle(txt, size=18)
        plt.subplots_adjust(hspace=0.35, wspace=0.35, left=0.075, right=0.925, top=0.90, bottom=0.075)

        axs[0].plot(t, data['y'][:, i])
        axs[0].plot(t, dataPred['y'][:, i])
        axs[0].set_title('Values prediction ' + outLabel[ii - 1])
        axs[0].set_ylabel(outLabel[ii - 1] + ' (' + setupDat['outUnits'][outLabel[ii - 1]][0] + ')')
        axs[0].grid(True)
        axs[0].legend(['True', 'Pred'])

        axs[1].plot(t, data['y'][:, i] - dataPred['y'][:, i])
        axs[1].set_title('Error prediction ' + outLabel[ii - 1])
        axs[1].set_ylabel(outLabel[ii - 1] + ' (' + setupDat['outUnits'][outLabel[ii - 1]][0] + ')')
        axs[1].grid(True)

        axs[2].plot(t, 2 * data['L'][:, i], color=colors[0])
        axs[2].plot(t, dataPred['L'][:, i], color=colors[1])
        axs[2].set_title('States labels ' + outLabel[ii - 1])
        axs[2].set_ylabel(outLabel[ii - 1] + ' (On/Off)')
        axs[2].legend(['True', 'Pred'])
        axs[2].grid(True)

        axs[3].plot(t, data['L'][:, i] - dataPred['L'][:, i])
        axs[3].set_title('Error labels ' + outLabel[ii - 1])
        axs[3].set_xlabel('time (hrs)')
        axs[3].set_ylabel(outLabel[ii - 1] + ' (On/Off)')
        axs[3].grid(True)

        plt.tight_layout()

    plt.show()
