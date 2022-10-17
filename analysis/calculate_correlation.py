import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import EEG, EMG, EOG
from feature_extraction import (
    get_feature_by_name,
    Feature,
    EEG_BANDS,
)

from data_utils import (
    get_sorted_behavior_labels,
    get_sorted_block_to_data_by_marker,
)


MULTIPLE_CHANNELS_SIGNAL = [EEG.__name__, EMG.__name__, EOG.__name__]
EEG_BANDS_LIST = [
    Feature.DELTA,
    Feature.THETA,
    Feature.ALPHA,
    Feature.BETA1,
    Feature.BETA2,
    Feature.GAMMA,
]

EEG_BANDS_NAMES = [
    Feature.DELTA.name,
    Feature.THETA.name,
    Feature.ALPHA.name,
    Feature.BETA1.name,
    Feature.BETA2.name,
    Feature.GAMMA.name,
]

STAT_FEATURES = [
    Feature.STD,
    Feature.PTP,
    Feature.VAR,
    Feature.MINIM,
    Feature.MAXIM,
    Feature.MEAN_SQUARE,
    Feature.RMS,
    Feature.ABS_DIFF,
    Feature.SKEWNESS,
    Feature.KURTOSIS,
]


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


def get_pearson_correlation_by_feature(
    all_blocks: np.ndarray,
    labels: list,
    f: Feature,
    num_channel: int = 128,
    num_blocks: int = 10,
):
    pearson_corr = np.zeros((num_channel, num_blocks))
    labels_chunks = list(divide_chunks(labels, num_blocks))
    for ch in range(num_channel):
        spf = get_feature_by_name(all_blocks=all_blocks, feature_name=f, channel=ch)

        spf_chunks = list(divide_chunks(spf, num_blocks))
        for i in range(0, num_blocks):
            corr = pearsonr(spf_chunks[i], labels_chunks[i])
            pearson_corr[ch][i] = corr[0]

    return pearson_corr


def get_eeg_features_means(
    feature_to_pc: np.ndarray, all_block_names: list, k: int = 1
):
    means = np.zeros((len(EEG_BANDS_LIST), len(all_block_names) + 1))
    i = 0
    for f in EEG_BANDS_LIST:
        # mean = np.mean(feature_to_pc[f], axis=0)
        avg = 0
        for nb in range(len(all_block_names)):
            top_k = np.partition(feature_to_pc[f][:, nb], -k)[-k:]
            # bottom_k = np.partition(feature_to_pc[f][:,nb], k)[:k]
            rounded_mean = np.round_(np.mean(top_k), decimals=3)
            means[i][nb] = rounded_mean
            avg += rounded_mean

        means[i][-1] = np.round_(avg / len(all_block_names), decimals=3)
        i += 1

    return means


def get_feature_to_pearson_correlation(
    all_blocks: np.ndarray,
    labels: list,
    features: list,
    num_channel: int = 128,
) -> dict:
    feature_to_pc = {}
    for feature_name in features:
        feature_to_pc[feature_name] = get_pearson_correlation_by_feature(
            all_blocks, labels, feature_name, num_channel
        )
    return feature_to_pc


def get_all_conditions_feature_to_pc_by_markers(
    all_data: dict, marker: str, features: list, num_channel: int = 128
) -> dict:
    all_block_names = list(all_data.keys())
    all_block_names.sort()
    print(all_block_names)

    all_blocks = get_sorted_block_to_data_by_marker(all_data, marker, all_block_names)

    condition_to_feature = {}
    for condition in ["valence", "arousal", "attention"]:
        labels = get_sorted_behavior_labels(all_data, condition, all_block_names)
        feature_to_pc = get_feature_to_pearson_correlation(
            all_blocks, labels, features, num_channel
        )
        condition_to_feature[condition] = feature_to_pc
        print(f"Complete computing {condition} features")

    return condition_to_feature
