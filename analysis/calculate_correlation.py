from collections import defaultdict

import numpy as np
from scipy.stats import combine_pvalues
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import BEHAVIOR_LIST
from feature_extraction import (
    get_feature_by_name,
    Feature,
)

from data_utils import (
    get_sorted_behavior_labels,
    get_sorted_block_to_data_by_marker,
)

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

# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def get_correlation_by_feature(
    all_blocks: np.ndarray,
    labels: list,
    f: Feature,
    num_channel: int = 128,
    num_blocks: int = 10,
):
    # calculate all blocks
    if num_blocks == 0:
        return _get_all_blocks_correlation_by_feature(
            all_blocks, labels, f, num_channel
        )

    pearson_corr = np.zeros((num_channel, num_blocks))
    labels_chunks = list(divide_chunks(labels, num_blocks))
    for ch in range(num_channel):
        spf = get_feature_by_name(all_blocks=all_blocks, feature_name=f, channel=ch)

        spf_chunks = list(divide_chunks(spf, num_blocks))
        for i in range(0, num_blocks):
            corr = pearsonr(spf_chunks[i], labels_chunks[i])
            pearson_corr[ch][i] = corr[0]

    return pearson_corr


def _get_all_blocks_correlation_by_feature(
    all_blocks: np.ndarray,
    labels: list,
    f: Feature,
    num_channel: int = 128,
):
    pearson_corr = np.zeros((num_channel, 2))
    spearman_corr = np.zeros((num_channel, 2))
    for ch in range(num_channel):
        spf = get_feature_by_name(all_blocks=all_blocks, feature_name=f, channel=ch)
        spearman_corr[ch] = spearmanr(spf, labels)
        pearson_corr[ch] = pearsonr(spf, labels)

    return {"pearson": pearson_corr, "spearman": spearman_corr}


def get_eeg_features_means(
    feature_to_pc: np.ndarray, all_block_names: list, k: int = 1
):
    """Get top k eeg pearson correlation means

    Parameters
    ----------
    feature_to_pc : map
        key: feature name
        value: (num_channels, num_blocks)
    all_block_names: the names of the blocks
    k: int
        Top k positive and top k negative.

    Returns
    -------
    data : array
        eeg top k channel pearson correlation means
    """
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


def get_feature_to_correlation(
    all_blocks: np.ndarray,
    labels: list,
    features: list,
    num_channel: int = 128,
    num_blocks: int = 10,
) -> dict:
    feature_to_pc = {}
    for feature_name in features:
        feature_to_pc[feature_name] = get_correlation_by_feature(
            all_blocks, labels, feature_name, num_channel, num_blocks
        )
    return feature_to_pc


def get_all_behaviors_feature_to_pc_by_markers(
    all_data: dict,
    marker: str,
    features: list,
    num_channel: int = 128,
    num_blocks: int = 10,
) -> dict:
    all_block_names = list(all_data.keys())
    all_block_names.sort()
    print(all_block_names)

    all_blocks = get_sorted_block_to_data_by_marker(all_data, marker, all_block_names)

    condition_to_feature = {}
    for condition in BEHAVIOR_LIST:
        labels = get_sorted_behavior_labels(all_data, condition, all_block_names)
        feature_to_pc = get_feature_to_correlation(
            all_blocks, labels, features, num_channel, num_blocks
        )
        condition_to_feature[condition] = feature_to_pc
        print(f"Complete computing {condition} features correlation")

    return condition_to_feature


def get_all_trials_average_rp_values(
    dir_name_to_cf: dict,
    features: list,
    corr_name: str,
):
    """Get average r values and combined p values with Fisher's method

    Parameters
    ----------
    dir_name_to_cf : a map
        key: dir name
        value: map[behavior]: (
                map[feature]: {"pearson": (num_channel, [r_value, p_value]), "spearman": (num_channel, [r_value, p_value])}
    features: list of features
    Returns
    -------
    behavior_to_avg_rp : map
        key: behavior name
        value: map[feature]: (num_channel, [r_value, p_value])
    """
    behavior_to_avg_rp = defaultdict()
    all_dirs = list(dir_name_to_cf.keys())
    for behavior in BEHAVIOR_LIST:
        behavior_to_avg_rp[behavior] = defaultdict()
        for f in features:
            all_rv = np.asarray(
                [
                    dir_name_to_cf[dir_name][behavior][f][corr_name][:, 0]
                    for dir_name in all_dirs
                ]
            )
            all_pv = np.asarray(
                [
                    dir_name_to_cf[dir_name][behavior][f][corr_name][:, 1]
                    for dir_name in all_dirs
                ]
            )
            all_pv = np.swapaxes(all_pv, 0, -1)

            combined_pv = [np.round_(combine_pvalues(p)[1], decimals=3) for p in all_pv]
            mean_rv = np.round_(np.mean(all_rv, axis=0), decimals=3)
            behavior_to_avg_rp[behavior][f] = np.swapaxes(
                np.asarray([mean_rv, combined_pv]), 0, -1
            )

    return behavior_to_avg_rp
