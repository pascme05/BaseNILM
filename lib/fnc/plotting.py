#######################################################################################################################
#######################################################################################################################
# Title: Baseline NILM Architecture
# Topic: Non-intrusive load monitoring utilising machine learning, pattern matching and source separation
# File: plotting
# Date: 23.10.2021
# Author: Dr. Pascal A. Schirmer
# Version: V.0.0
# Copyright: Pascal A. Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import matplotlib.pyplot as plt
import numpy as np


#######################################################################################################################
# Function
#####################################################################################################################
def plotting(Y_test, Y_Pred, Y_testLabel, Y_PredLabel, resultsApp, resultsAvg, setup_Data, setup_Exp):
    ####################################################################################################################
    # Welcome Message
    ####################################################################################################################
    print("Running NILM tool: Plotting Results")

    ####################################################################################################################
    # Plotting
    ####################################################################################################################
    # ------------------------------------------
    # init
    # ------------------------------------------
    time = np.linspace(0, len(Y_test),  len(Y_test))
    accLabels = ['ACC', 'F1', 'EACC', 'RMSE', 'MAE', 'SAE']
    SAE = abs(resultsAvg[6] - resultsAvg[5]) / resultsAvg[6]
    accResults = [resultsAvg[0], resultsAvg[1], resultsAvg[2], resultsAvg[3], resultsAvg[4], SAE]
    appLabel = setup_Data['labels']

    # ------------------------------------------
    # plotting performance
    # ------------------------------------------
    if setup_Exp['plotting'] >= 1:
        fig1, ax1 = plt.subplots(2, 2)
        ax1[0, 0].pie(resultsApp[:, 6], labels=appLabel, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1[0, 0].set(aspect="equal", title='Groundtruth Energy Consumption')
        ax1[0, 1].pie(resultsApp[:, 5], labels=appLabel, autopct='%1.1f%%', shadow=True, startangle=90)
        ax1[0, 1].set(aspect="equal", title='Predicted Energy Consumption')
        ax1[1, 0].bar(accLabels, accResults)
        ax1[1, 0].set(title='Average Performance')
        ax1[1, 1].bar(appLabel, resultsApp[:, 2])
        ax1[1, 1].set(title='Appliance Performance (EACC)')
        plt.show()

    # ------------------------------------------
    # plotting time series
    # ------------------------------------------
    if setup_Exp['plotting'] > 1:
        for i in range(0, setup_Data['numApp']):
            ii = i + 1
            plt.figure(ii)
            plt.subplot(211)
            plt.plot(time, Y_test[:, i], 'k', time, Y_Pred[:, i], 'b')
            plt.title('Active Power Consumption of Device ' + appLabel[ii-1])
            plt.ylabel('Power P in [W]')

            plt.subplot(212)
            plt.plot(time, 2*Y_testLabel[:, i], 'k', time, Y_PredLabel[:, i], 'b')
            plt.title('Appliance On-Sets of Device ' + appLabel[ii-1])
            plt.xlabel('Time t in [s]')
            plt.ylabel('On/Off')
        plt.show()
