from collections import defaultdict

import mat73
import glob
import numpy as np
import scipy.io as sio
from typing import Tuple

from feature_extraction import (
    concatenate_features,
    get_all_blocks_features_by_channel,
    process_spectral_power_for_channels,
)
from biomarkers import BEHAVIOR_LIST, SIOBioMarkers, Mat73BioMarkers, BioMarkersInterface, EEG

ALL_DIRS = ["../2000_CleanData", "../2001_CleanData", "../1004_CleanData"]


def load_data_from_file(file_name: str) -> BioMarkersInterface:
    try:
        raw_data = mat73.loadmat(file_name)
        signal = raw_data["Signal"]

        print(f"Complete loading {len(signal)} markers")
        return Mat73BioMarkers(signal)
    except Exception as ex:
        raw_data = sio.loadmat(file_name)
        signal = raw_data["Signal"]

        print(f"Complete loading {len(signal.dtype)} markers")
        return SIOBioMarkers(signal)


def load_data_from_dir(dir_name: str) -> dict:
    # All files and directories ending with .mat and that don't begin with a dot:
    all_files = glob.glob(dir_name + "/*.mat")
    all_data = {}
    for f in all_files:
        markers = load_data_from_file(f)
        block_name = markers.get_block_name()
        print(f"Loaded {markers.get_block_name()} block")
        all_data[block_name] = markers

    return all_data


def transform_to_marker_to_all_block(all_data: dict) -> dict:
    marker_to_all_block = {}
    for _, markers in all_data.items():
        marker_to_data = markers.get_all_data()
        for marker, data in marker_to_data.items():
            if marker not in marker_to_all_block:
                marker_to_all_block[marker] = data
            else:
                marker_to_all_block[marker] = np.concatenate(
                    (marker_to_all_block[marker], data), axis=-1
                )

    return marker_to_all_block


def get_features(all_epochs, get_spectral=False):
    all_epochs = np.swapaxes(
        all_epochs, 0, -1
    )  # (num_channels, num_data_points, num_epochs) => (num_epochs, num_data_points, num_channels)

    features = []
    for data in all_epochs:
        if get_spectral:
            spf = process_spectral_power_for_channels(data, 512)
            features.append(spf)
        else:
            features.append(concatenate_features(data))

    features = np.array(features)
    return features


def get_features_in_block(marker_to_data: dict) -> np.array:
    all_features = np.array([])
    for marker, data in marker_to_data.items():
        marker_features = get_features(data)
        print(f"{marker} feature shape {marker_features.shape}")

        if all_features.ndim > 1:
            all_features = np.concatenate((all_features, marker_features), axis=1)
        else:
            all_features = marker_features

    # extract spectral data in eeg
    spetral_features = get_features(marker_to_data[EEG.__name__], True)
    print(f"EEG spetral features shape {spetral_features.shape}")
    all_features = np.concatenate((all_features, spetral_features), axis=1)
    return all_features


def get_all_features_by_marker(
    all_data: dict,
    marker: str,
    features: list,
    channel_num: int = 0,
) -> dict:
    all_block_names = list(all_data.keys())
    all_block_names.sort()

    all_blocks = get_sorted_block_to_data_by_marker(all_data, marker, all_block_names)

    return get_all_blocks_features_by_channel(all_blocks, features, channel_num)


def get_all_behaviors_labels(
    all_data: dict,
) -> dict:
    all_block_names = list(all_data.keys())
    all_block_names.sort()

    behavior_to_labels = {}
    for b in BEHAVIOR_LIST:
        behavior_to_labels[b] = get_sorted_behavior_labels(all_data, b, all_block_names)

    return behavior_to_labels


def get_sorted_behavior_labels(all_data, label_name, sorted_blocks: list):
    all_labels = np.array([])
    for block in sorted_blocks:
        y = all_data[block].get_labels(label_name)
        all_labels = np.concatenate((all_labels, y), axis=None)

    print(f"All labels shape: {all_labels.shape}")
    return all_labels


def get_sorted_block_to_data_by_marker(
    all_data: dict, marker_name: str, sorted_blocks: list
) -> dict:
    all_blocks = np.array([])
    for block in sorted_blocks:
        marker_to_data = all_data[block].get_all_data()
        data = marker_to_data[marker_name]
        all_blocks = (
            data
            if all_blocks.ndim == 1
            else np.concatenate((all_blocks, data), axis=-1)
        )

    return all_blocks

def extract_features_by_channel(marker: str, dir_to_data: dict, features: list, channel_num: int):
    dir_name_to_features = {}
    for dir_name, all_data in dir_to_data.items():
        feature_to_value = get_all_features_by_marker(
            all_data, marker, features, channel_num
        )

        dir_name_to_features[dir_name] = feature_to_value

    features_to_trials = defaultdict()
    all_data = dir_to_data["../2000_CleanData"]
    channel_name = all_data["audio_hvla"].get_chanlocs(marker)[channel_num]

    for dir_name, fv in dir_name_to_features.items():
        for f, v in fv.items():
            key = f'{channel_name}_{f.name}'
            if key not in features_to_trials:
                features_to_trials[key] = defaultdict()
            if dir_name not in features_to_trials[key]:
                features_to_trials[key][dir_name] = defaultdict()

            features_to_trials[key][dir_name] = v
    return features_to_trials

def extract_labels(dir_to_data: dict, all_dir: list=ALL_DIRS):
    """Get labels

    Parameters
    ----------
    dir_to_data: dict
        the key is dir name and the data is the biomarkers
    all_dir: list
        the list of trials
    Returns
    -------
    behavior_to_label_array : dict
        key: behavior
        value: (num_labels)
    """
    behavior_to_label_array = {"valence": [], "arousal": [], "attention": []}
    for dir_name in all_dir:
        all_data = dir_to_data[dir_name]
        b_to_l = get_all_behaviors_labels(all_data)
        for b, labels in b_to_l.items():
            behavior_to_label_array[b].append(labels)

    behavior_to_label_array = {b: np.concatenate(labels) for b, labels in behavior_to_label_array.items()}
    return behavior_to_label_array

def extract_features(marker: str, dir_to_data: dict, features: list, all_dir: list=ALL_DIRS):
    """Get a marker's features.

    Parameters
    ----------
    marker : str
    dir_to_data: dict
        the key is dir name and the data is the biomarkers
    features: list
        the list of features
    Returns
    -------
    f_array : (num_trials, num_features) e.g (130, 10)
    feature_names: the list of feature names e.g. VEOG_STD
    """
    all_channels = dir_to_data[all_dir[0]]["audio_hvla"].get_chanlocs(marker)
    all_features_array = []
    for dir_name in all_dir:
        all_data = dir_to_data[dir_name]
        f_array = []
        for i, ch in enumerate(all_channels):
            feature_to_value = get_all_features_by_marker(
                all_data, marker, features, i
            ) # feature_to_value: feature_name to num_blocks_values

            f_array.append(_get_feature_values(feature_to_value, features))

        all_features_array.append(np.concatenate(f_array, axis=0))

    all_features_array = np.concatenate(all_features_array, axis=-1)

    feature_names = []
    for i, ch in enumerate(all_channels):
        f_names = [f'{ch}_{f.name}' for f in features]
        feature_names.extend(f_names)

    return np.swapaxes(all_features_array, 0, -1), feature_names

def _get_feature_values(features_to_trials: dict, features: list):
    """Get a marker's features.

    Parameters
    ----------
    features_to_trials: dict
        the key is feature name and the data is the feature values of all blocks
    features: list
        the list of features
    Returns
    -------
    f_array : (num_features, num_trials) e.g (10, 130)
    """
    f_array = [features_to_trials[f] for f in features]
    return np.array(f_array)

# should not concatenate all data.
def concatenate_all_data(dir_to_data: dict, marker: str) -> Tuple[np.ndarray, dict]:
    all_participants_data = np.array([])
    condition_to_labels = {"valence": [], "arousal": [], "attention": []}
    for _, data in dir_to_data.items():
        block_names = list(data.keys())
        block_names.sort()

        sorted_data = get_sorted_block_to_data_by_marker(data, marker, block_names)
        sorted_data = np.expand_dims(sorted_data, axis=0) if sorted_data.ndim == 2 else sorted_data
        all_participants_data = (
            sorted_data
            if all_participants_data.ndim == 1
            else np.concatenate((all_participants_data, sorted_data), axis=-1)
        )

        for condition, _ in condition_to_labels.items():
            labels = get_sorted_behavior_labels(data, condition, block_names)
            condition_to_labels[condition].extend(labels)

    return all_participants_data, condition_to_labels
