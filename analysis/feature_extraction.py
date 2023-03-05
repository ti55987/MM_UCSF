from enum import Enum

import numpy as np
from features.psd import avg_welch_bandpower, welch_bandpower
from scipy import stats


class Feature(Enum):
    DELTA = 1
    THETA = 2
    ALPHA = 3
    BETA1 = 4
    BETA2 = 5
    GAMMA = 6
    MEAN = 7
    STD = 8
    PTP = 9
    VAR = 10
    MINIM = 11
    MAXIM = 12
    MEAN_SQUARE = 13
    RMS = 14
    ABS_DIFF = 15
    SKEWNESS = 16
    KURTOSIS = 17


AXIS = 0

EEG_BANDS = {
    Feature.DELTA: (1, 4),
    Feature.THETA: (4, 8),
    Feature.ALPHA: (8, 12),
    Feature.BETA1: (12, 20),
    Feature.BETA2: (20, 30),
    Feature.GAMMA: (30, 50),
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
    return np.mean(data ** 2, axis=AXIS)


# root mean square.
def rms(data):
    return np.sqrt(np.mean(data ** 2, axis=AXIS))


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


FEATURE_TO_FUNC = {
    Feature.MEAN: mean,
    Feature.STD: std,
    Feature.PTP: ptp,
    Feature.VAR: var,
    Feature.MINIM: minim,
    Feature.MAXIM: maxim,
    Feature.MEAN_SQUARE: mean_square,
    Feature.RMS: rms,
    Feature.ABS_DIFF: abs_diffs_signal,
    Feature.SKEWNESS: skewness,
    Feature.KURTOSIS: kurtosis,
}

STAT_FEATURES = FEATURE_TO_FUNC.keys()


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


def get_mean_spectral_power(data, srate):
    bandpower = dict()
    freqs, psd = welch_bandpower(data, srate, None, 2)
    for band, values in EEG_BANDS.items():
        bandpower[band] = avg_welch_bandpower(freqs, psd, values)
    return bandpower


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


def get_feature_by_name(
    all_blocks: np.ndarray,
    feature_name: Feature,
    channel: int = 0,
):
    all_epoch_data = np.swapaxes(
        all_blocks, 0, -1
    )  # (num_channels, num_data_points, num_epochs) => (num_epochs, num_data_points, num_channels)

    all_blocks_features = []
    for data in all_epoch_data:
        val = 0
        if data.ndim > 1:
            data = data[:, channel]

        if feature_name in EEG_BANDS.keys():
            eeg_band_fft = get_mean_spectral_power(data, 512)
            # (TODO) this should be optimized
            val = eeg_band_fft[feature_name]
        else:
            func = FEATURE_TO_FUNC[feature_name]
            val = func(data)

        all_blocks_features.append(val)

    return all_blocks_features


def get_spectral_power_features(
    all_blocks: np.ndarray,
    channel: int = 0,
    get_raw: bool = False,
):
    print(f"Extracting PSD features for {channel}...")

    name_to_features = {}
    all_epoch_data = np.swapaxes(
        all_blocks, 0, -1
    )  # (num_channels, num_data_points, num_epochs) => (num_epochs, num_data_points, num_channels)

    for data in all_epoch_data:
        if data.ndim > 1:
            data = data[:, channel]

        eeg_band_fft = get_mean_spectral_power(data, 512)
        for f in EEG_BANDS.keys():
            if f not in name_to_features:
                name_to_features[f] = []

            name_to_features[f].append(eeg_band_fft[f])

    return name_to_features


def get_all_blocks_features_by_channel(
    all_blocks: np.ndarray,
    features: list,
    channel_num: int = 0,
) -> dict:
    name_to_features = {}
    for feature_name in features:
        name_to_features[feature_name] = get_feature_by_name(
            all_blocks=all_blocks, feature_name=feature_name, channel=channel_num
        )
    return name_to_features
