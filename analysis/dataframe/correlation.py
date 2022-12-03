from collections import defaultdict

from scipy.stats import kendalltau, pearsonr, spearmanr
from scipy.stats import combine_pvalues

import pandas as pd
import numpy as np

from biomarkers import (
    EEG_CHANEL_NAMES,
)
from feature_extraction import (
    Feature,
)


def get_eeg_channel_num_from_feature_name(feature):
    """Get eeg channel number from {channel}_{feature} format

    Parameters
    ----------
    feature : str {channel}_{feature} format

    Returns
    -------
    channel number : int
    """
    for i in range(len(EEG_CHANEL_NAMES)):
        channel = feature.split("_", 1)[0]
        if EEG_CHANEL_NAMES[i] == channel.strip():
            return i
    return -1


def get_feature_name(feature):
    """Get feature name from {channel}_{feature} format

    Parameters
    ----------
    feature : str {channel}_{feature} format

    Returns
    -------
    feature_name : str
    """
    return feature.split("_", 1)[1].strip()


def get_feature_to_corr_by_behavior(
    result: pd.DataFrame, behavior: str, feature_names: list
):
    """Get feature correlation from all subjects

    Parameters
    ----------
    result : load from csv data
    behavior: [Valance, Attention, Arousal]
    feature_names: list of the features

    Returns
    -------
    feature_to_corr : map
        key: feature name
        value: np.array(channel, subject, [r, p])
    """
    feature_to_corr = {}
    num_subjects = len(result.Subject.unique())
    for i, s in enumerate(result.Subject.unique()):
        subject_features = result[result["Subject"] == s]
        labels = subject_features[behavior].values
        for f in feature_names:
            score = pearsonr(subject_features[f].values, labels)
            channel_num = get_eeg_channel_num_from_feature_name(f)

            if channel_num == -1:
                continue

            name = get_feature_name(f)
            if name not in feature_to_corr:
                feature_to_corr[name] = np.empty((128, num_subjects, 2))

            # first is correlation , second is p_value
            feature_to_corr[name][channel_num][i] = [score[0], score[1]]
    return feature_to_corr


def get_behavior_to_average_corr(behavior_to_rp):
    """Get average correlation from all subjects

    Parameters
    ----------
    behavior_to_rp : map
    key: [Valance, Attention, Arousal]
    value: a map
        key: feature name
        value: np.array(channel, subject, [r, p])

    Returns
    -------
    avg_condition_to_features : map
        key: behavior name
        value: map
            key: feature
            value: np.array(channel, [r, p])
    """
    avg_condition_to_features = defaultdict()
    for b, feature_to_corr in behavior_to_rp.items():
        avg_condition_to_features[b] = defaultdict()
        for f, corr in feature_to_corr.items():
            mean_rv = np.round_(np.mean(corr[:, :, 0], axis=1), decimals=3)
            combined_pv = [
                np.round_(combine_pvalues(p)[1], decimals=3) for p in corr[:, :, 1]
            ]
            avg_condition_to_features[b][Feature[f]] = np.swapaxes(
                np.asarray([mean_rv, combined_pv]), 0, -1
            )
    return avg_condition_to_features
