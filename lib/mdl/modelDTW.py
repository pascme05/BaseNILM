# Load Libaries
import numpy as np
from tslearn import metrics


def modelDTW(X_train, Y_train, X_test, setup_Data, setup_Para):
    # Init Variables
    Y_Pred = np.zeros((X_test.shape[0], X_test.shape[1], setup_Data['numApp']))

    # Fit regression model
    dist = np.zeros((X_test.shape[0], X_train.shape[0]))
    for i in range(0, X_test.shape[0]):
        for ii in range(0, X_train.shape[0]):
            dist[i, ii] = metrics.dtw(X_test[i, :, :], X_train[ii, :, :])
        sel = np.argmin(dist[i, :])
        Y_Pred[i, :, :] = Y_train[sel, :, :]

    # Post-Processing
    Y_Pred = Y_Pred.reshape((Y_Pred.shape[0] * Y_Pred.shape[1], Y_Pred.shape[2]))
    X_Pred = np.sum(Y_Pred, axis=1)

    return [X_Pred, Y_Pred]

