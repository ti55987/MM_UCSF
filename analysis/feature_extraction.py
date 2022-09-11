from scipy import stats
import numpy as np

AXIS = 0

# mean is the average of the data
def mean(data):
    return np.mean(data,axis=AXIS)

# std is the standard deviation
def std(data):
    return np.std(data,axis=AXIS)

# ptp indicates peak to peak
def ptp(data):
    return np.ptp(data,axis=AXIS)

# var is the variance of the data
def var(data):
        return np.var(data,axis=AXIS)

# The minimum of the data
def minim(data):
      return np.min(data,axis=AXIS)

# The maximum of the data
def maxim(data):
      return np.max(data,axis=AXIS)

# Not used. The indices of the minimum values
def argminim(data):
      return np.argmin(data,axis=AXIS)

# Not used. The indices of the maximum values
def argmaxim(data):
      return np.argmax(data,axis=AXIS)

def mean_square(data):
      return np.mean(data**2,axis=AXIS)

# root mean square
def rms(data):
      return  np.sqrt(np.mean(data**2,axis=AXIS))

def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data,axis=AXIS)),axis=AXIS)

# skewness is a measure of the asymmetry of the probability distribution
# of a real-valued random variable about its mean.
def skewness(data):
    return stats.skew(data,axis=AXIS)

# kurtosis is a measure of the "tailedness" of the probability distribution
# of a real-valued random variable.
def kurtosis(data):
    return stats.kurtosis(data,axis=AXIS)

def concatenate_features(data):
    features = (mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),
                          mean_square(data),rms(data),abs_diffs_signal(data),
                          skewness(data),kurtosis(data))
    if data.ndim == 1:
        return features

    return np.concatenate(features,axis=AXIS)
