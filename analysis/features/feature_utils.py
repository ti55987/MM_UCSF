import numpy as np

from features.psd import get_eeg_psd_by_channel_band, get_psd_by_channel
from features.time_series import get_time_series_by_channel
from features.constants import Feature, MARKER_TO_FEATURE
from feature_extraction import EEG_BANDS

from resample.resample import (
    slice_data_by_seconds,
)

from constants import AUDIO_BLOCKS

def get_features(block_data, channel_type: str, srate: int, feature):
    if feature in list(EEG_BANDS.keys()):
        return get_eeg_psd_by_channel_band(block_data, channel_type, srate, feature)
    elif feature == Feature.ECG_HF or feature == Feature.EGG_FILTERED:
        return block_data[:, 0, :]
    elif feature == Feature.ECG_LF or feature == Feature.EGG_PHASE:
        return block_data[:, 1, :]
    elif feature == Feature.ECG_LFHF or feature == Feature.EGG_AMPLITUDE:
        return block_data[:, 2, :]
    else:
        return get_time_series_by_channel(block_data, channel_type)

def get_block_raw_data_by_marker(subject_data, blocks, marker, second_per_slice):
    from resample.resample import (
        slice_data_by_seconds,
    )

    block_to_data = {}
    for b in blocks:
        block_data = subject_data[b]
        srate =  block_data.get_srate(marker)
        sliced_data = slice_data_by_seconds(block_data.get_all_data()[marker], srate, second_per_slice)
        sliced_data = sliced_data[:, 0, :]
        block_to_data[b] = sliced_data
    
    return block_to_data  

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
            psd_data = get_features(sliced_data, channel, srate, feature)
        else:
            psd_data = get_psd_by_channel(block_data, marker, channel, feature)

        features = np.vstack((psd_data, features)) if len(features) > 0 else psd_data

    return features


def get_eeg_channel_feature_to_data(subject_data, block_list, feature_list):
    sliced_channel_feature_to_data = {"A": {}, "B": {}, "C": {}, "D": {}}
    for c in sliced_channel_feature_to_data.keys():
        for f in feature_list: #EEG_BANDS.keys():
            raw_data = get_block_features(
                block_list, subject_data, 'EEG', c, f, True
            )
            sliced_channel_feature_to_data[c][f.name] = raw_data

    return sliced_channel_feature_to_data


def get_feature_to_data(subject_data, block_list: list, feature_list: list, marker: str = "EEG"):
    if marker == "EEG":
        return get_eeg_channel_feature_to_data(subject_data, block_list, feature_list)

    sliced_feature_to_data = {}
    for f in feature_list:
        raw_data = get_block_features(block_list, subject_data, marker, "", f, True)
        sliced_feature_to_data[f.name] = raw_data

    return sliced_feature_to_data