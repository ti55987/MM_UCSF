from scipy import stats
import numpy as np

AXIS = 0

EEG_BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta1": (12, 20),
    "Beta2": (20, 30),
    "Gamma": (30, 50),
}

# mean is the average of the data
def mean(data):
    return np.mean(data, axis=AXIS)


# std is the standard deviation
def std(data):
    return np.std(data, axis=AXIS)


# ptp indicates peak to peak
def ptp(data):
    return np.ptp(data, axis=AXIS)


# var is the variance of the data
def var(data):
    return np.var(data, axis=AXIS)


# The minimum of the data
def minim(data):
    return np.min(data, axis=AXIS)


# The maximum of the data
def maxim(data):
    return np.max(data, axis=AXIS)


# Not used. The indices of the minimum values
def argminim(data):
    return np.argmin(data, axis=AXIS)


# Not used. The indices of the maximum values
def argmaxim(data):
    return np.argmax(data, axis=AXIS)


def mean_square(data):
    return np.mean(data**2, axis=AXIS)


# root mean square
def rms(data):
    return np.sqrt(np.mean(data**2, axis=AXIS))


def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data, axis=AXIS)), axis=AXIS)


# skewness is a measure of the asymmetry of the probability distribution
# of a real-valued random variable about its mean.
def skewness(data):
    return stats.skew(data, axis=AXIS)


# kurtosis is a measure of the "tailedness" of the probability distribution
# of a real-valued random variable.
def kurtosis(data):
    return stats.kurtosis(data, axis=AXIS)


# get_frequency_idx returns the indexes given the frequency bands
def get_frequency_idx(sz, srate):
    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(sz, 1.0 / srate)
    freq_ix = dict()
    for band in EEG_BANDS:
        freq_ix[band] = np.where(
            (fft_freq >= EEG_BANDS[band][0]) & (fft_freq <= EEG_BANDS[band][1])
        )[0]
    return freq_ix


def get_spectral_power(data, srate):
    # Get real amplitudes of FFT (only in postive frequencies)
    fft_vals = np.absolute(np.fft.rfft(data))
    freq_ix = get_frequency_idx(len(data), srate)
    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()
    for band, indices in freq_ix.items():
        eeg_band_fft[band] = np.mean(fft_vals[indices])

    return eeg_band_fft


# data is (num_data_points, num_channels)
def process_spectral_power_for_channels(data, srate):
    features = []
    for channel in range(data.shape[1]):
        eeg_band_fft = get_spectral_power(data[:, channel], srate)
        fft_values = list(eeg_band_fft.values())
        features.append(fft_values)

    return np.concatenate(features)


# data is (num_data_points, num_channels)
# return features:  (num_features x num_channels)
def concatenate_features(data):
    features = (
        mean(data),  # (1, num_channels)
        std(data),
        ptp(data),
        var(data),
        minim(data),
        maxim(data),
        mean_square(data),
        rms(data),
        abs_diffs_signal(data),
        skewness(data),
        kurtosis(data),
    )
    if data.ndim == 1:
        return features

    return np.concatenate(features, axis=AXIS)
