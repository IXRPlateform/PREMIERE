import numpy as np

from scipy.interpolate import RBFInterpolator
from scipy.interpolate import UnivariateSpline
# from scipy.interpolate import Rbf

# def buildRBF ( Xm, Ym, function='linear', epsilon=.01, smooth=.01):
#     rbfi = Rbf(Xm, Ym, function=function, epsilon=epsilon, smooth=smooth)  # radial basis function interpolator instance
#     return rbfi

def buildInterpolator ( Xm, Ym, rbfkernel='linear', rbfepsilon=1.0, rbfsmooth=.01):
    # print ("Building interpolator with kernel: ", rbfkernel)
    # print ("Xm: ", Xm)
    # print ("Ym: ", Ym)
    if rbfkernel == 'univariatespline':
        interpolator = UnivariateSpline(Xm, Ym, k=3, s=rbfsmooth)
    else:
        points = np.column_stack((Xm, np.zeros_like(Xm)))  # Assuming 1D data for Xm
        values = Ym
        interpolator = RBFInterpolator(points, values, kernel=rbfkernel, epsilon=rbfepsilon, smoothing=rbfsmooth)
    return interpolator
