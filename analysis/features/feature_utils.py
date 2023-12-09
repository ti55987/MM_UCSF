import numpy as np

from features.psd import get_eeg_psd_by_channel_band, get_psd_by_channel
from features.constants import Feature, MARKER_TO_FEATURE
from feature_extraction import EEG_BANDS

from resample.resample import (
    slice_data_by_seconds,
)

from constants import AUDIO_BLOCKS

def get_features(block_data, marker, channel_type: str, srate: int, feature):
    if marker == "EEG":
        return get_eeg_psd_by_channel_band(block_data, channel_type, srate, feature)
    if feature == Feature.ECG_HF or feature == Feature.EGG_FILTERED:
        return block_data[:, 0, :]
    elif feature == Feature.ECG_LF or feature == Feature.EGG_PHASE:
        return block_data[:, 1, :]
    elif feature == Feature.ECG_LFHF or feature == Feature.EGG_AMPLITUDE:
        return block_data[:, 2, :]


def get_block_features(
    blocks, subject_data, marker, channel, feature, with_sliced: bool = False
):
    features = []

    for b in blocks:
        block_data = subject_data[b]
        if with_sliced:
            srate = block_data.get_srate(marker)
            sliced_data = slice_data_by_seconds(
                block_data.get_all_data()[marker], srate, 4
            )
            psd_data = get_features(sliced_data, marker, channel, srate, feature)
        else:
            psd_data = get_psd_by_channel(block_data, marker, channel, feature)

        features = np.vstack((psd_data, features)) if len(features) > 0 else psd_data

    return features


def get_eeg_channel_feature_to_data(subject_data, marker: str = "EEG"):
    sliced_channel_feature_to_data = {"A": {}, "B": {}, "C": {}, "D": {}}
    for c in sliced_channel_feature_to_data.keys():
        for f in EEG_BANDS.keys():
            raw_data = get_block_features(
                AUDIO_BLOCKS, subject_data, marker, c, f, True
            )
            sliced_channel_feature_to_data[c][f] = raw_data

    return sliced_channel_feature_to_data


def get_feature_to_data(subject_data, block_list: list=AUDIO_BLOCKS, marker: str = "EEG"):
    if marker == "EEG":
        return get_eeg_channel_feature_to_data(subject_data, marker)

    sliced_feature_to_data = {marker: {f: {} for f in MARKER_TO_FEATURE[marker]}}
    for f in sliced_feature_to_data[marker].keys():
        raw_data = get_block_features(block_list, subject_data, marker, "", f, True)
        sliced_feature_to_data[marker][f] = raw_data

    return sliced_feature_to_data