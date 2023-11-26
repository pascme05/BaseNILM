#######################################################################################################################
#######################################################################################################################
# Title:        EVOptiDrive (Optimal Drivetrain Design for Electrical Vehicles)
# Topic:        Electrical Vehicle (EV) Simulation
# File:         template
# Date:         19.08.2023
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
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
import numpy as np
from scipy.interpolate import interp1d


#######################################################################################################################
# Function
#######################################################################################################################
def zoh(x, xp, yp, left=np.nan, assume_sorted=False):
    r"""
    Interpolates a function by holding at the most recent value.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    left: int or float, optional, default is np.nan
        Value to use for any value less that all points in xp
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by DStauffman in July 2020.

    Examples
    --------
    >>> import numpy as np
    >>> xp = np.array([0., 111., 2000., 5000.])
    >>> yp = np.array([0, 1, -2, 3])
    >>> x = np.arange(0, 6001, dtype=float)
    >>> y = zoh(x, xp, yp)

    """
    # force arrays
    x  = np.asanyarray(x)
    xp = np.asanyarray(xp)
    yp = np.asanyarray(yp)
    # find the minimum value, as anything left of this is considered extrapolated
    xmin = xp[0] if assume_sorted else np.min(xp)
    # check that xp data is sorted, if not, use slower scipy version
    if assume_sorted or np.all(xp[:-1] <= xp[1:]):
        ix = np.searchsorted(xp, x, side='right') - 1
        return np.where(np.asanyarray(x) < xmin, left, yp[ix])
    func = interp1d(xp, yp, kind='zero', fill_value='extrapolate', assume_sorted=False)
    return np.where(np.asanyarray(x) < xmin, left, func(x))
