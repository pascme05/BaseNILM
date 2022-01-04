# Load Libaries
import numpy as np
from scipy.optimize import lsq_linear


def NMF(X_train, Y_train, Y_train2, X_test, setup_Data, setup_Para):
    # Init Variables
    Y_Pred = np.zeros((X_test.shape[0], setup_Data['numApp']))
    MTest = 24*7

    # Reshape data
    D = Y_train2[0:MTest, :, :, 0]
    for i in range(1, setup_Data['numApp']):
        D = np.append(D, Y_train2[0:MTest, :, :, i], axis=0)
    D = np.reshape(D, (D.shape[0], D.shape[1]*D.shape[2]))
    D = np.transpose(D)
    P = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

    # Fit regression model
    for i in range(0, X_test.shape[0]):
        A = lsq_linear(D, P[i, :], bounds=(0, 1))

    # Post-Processing
    X_Pred = np.sum(Y_Pred, axis=1)

    return [X_Pred, Y_Pred]
