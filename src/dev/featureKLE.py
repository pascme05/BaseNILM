import numpy as np
import scipy
from scipy.linalg import toeplitz
from numpy import linalg as LA
import math
from scipy import signal


def featureKLE(data, setup_Para):
    W = setup_Para['framelength']
    a = np.zeros(W)
    f = np.zeros(W)
    phi = np.zeros(W)
    x = np.zeros((W, W))
    A = np.zeros((len(np.transpose(data)), W))
    F = np.zeros((len(np.transpose(data)), W))
    Phi = np.zeros((len(np.transpose(data)), W))
    for i in range(0, len(np.transpose(data))):
        X = data[:, i]
        xc = scipy.correlate(X, X, mode='full')
        r = xc[W-1:]
        rxx = toeplitz(r)
        w, v = LA.eigh(rxx)
        v = v*(-1)
        v = np.asmatrix(v)
        for ii in range(0, W):
            x[:, ii] = np.transpose(v[:, ii] * X * v[:, ii])
            a[ii] = np.mean(np.abs(x[:, ii]))
            w, h = scipy.signal.freqz(x[:, ii])
            id = np.argmax(abs(h))
            f[ii] = id/len(h)*math.pi
            phi[ii] = np.angle(h[id])
        A[i, :] = a
        F[i, :] = f
        Phi[i, :] = phi

    return [A, F, Phi]
