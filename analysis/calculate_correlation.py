import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import EEG, EMG, EOG
from feature_extraction import (
    get_feature_by_name,
    Feature,
    EEG_BANDS,
)


MULTIPLE_CHANNELS_SIGNAL = [EEG.__name__, EMG.__name__, EOG.__name__]


def get_pearson_corr_with_stats_features(
    all_data: dict, labels: list, channel: int = 0
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
