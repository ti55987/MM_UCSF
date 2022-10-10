import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import EEG, EMG, EOG
from feature_extraction import (
    get_feature_by_name,
    Feature,
    EEG_BANDS,
)


MULTIPLE_CHANNELS_SIGNAL = [EEG.__name__, EMG.__name__, EOG.__name__]


def get_pearson_corr_with_stats_features(
    all_data: np.ndarray, labels: list, channel: int = 0
):
    pearson_corr = []
    features = []
    for f in Feature:
        if f in EEG_BANDS.keys():
            continue

        spf = get_feature_by_name(all_blocks=all_data, feature_name=f, channel=channel)
        corr = pearsonr(spf, labels)
        pearson_corr.append(corr[0])
        features.append(f.name)

    return pearson_corr, features


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_eeg_spectral_pearson_correlation(
    all_blocks: np.ndarray,
    labels: list,
    eeg_band: Feature,
    num_channel: int = 128,
    num_blocks: int = 10,
):
    pearson_corr = np.zeros((num_channel, num_blocks))
    labels_chunks = list(divide_chunks(labels, num_blocks))
    for ch in range(num_channel):
        spf = get_feature_by_name(
            all_blocks=all_blocks, feature_name=eeg_band, channel=ch
        )

        spf_chunks = list(divide_chunks(spf, num_blocks))
        for i in range(0, num_blocks):
            corr = pearsonr(spf_chunks[i], labels_chunks[i])
            pearson_corr[ch][i] = corr[0]

    return pearson_corr
