import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import BioMarkers, EEG, EMG, EOG, ALL_MARKERS
from feature_extraction import (
    get_feature_by_name,
    Feature,
    EEG_BANDS,
)


MULTIPLE_CHANNELS_SIGNAL = [EEG.__name__, EMG.__name__, EOG.__name__]


def calculate_pearson_all_features(
    all_data: dict, labels: list, marker_name: str, channel: int = 1
):
    pearson_corr = []
    features = []
    for f in Feature:
        if f in EEG_BANDS.keys():
            continue

        spf = get_feature_by_name(
            all_blocks=all_data,
            marker_name=marker_name,
            feature_name=f,
            channel=channel,
        )
        corr = pearsonr(spf, labels)
        # print(f"{marker_name} channel {channel} {feature_name} pearson correlation {corr[0]}")
        pearson_corr.append(corr[0])
        features.append(f.name)

    return pearson_corr, features
