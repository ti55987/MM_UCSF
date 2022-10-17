import mat73
import glob
import numpy as np
import scipy.io as sio

from feature_extraction import (
    concatenate_features,
    process_spectral_power_for_channels,
)
from biomarkers import SIOBioMarkers, Mat73BioMarkers, BioMarkersInterface, EEG


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


def get_features_in_all_blocks(all_blocks: dict) -> np.array:
    all_blocks_features = []
    for block_name, markers in all_blocks.items():
        marker_to_data = markers.get_all_data()
        block_features = get_features_in_block(marker_to_data)
        print(f"{block_name} block has features: {block_features.shape}...")
        all_blocks_features.append(block_features)
    return np.concatenate(all_blocks_features)


def get_behavior_labels(all_data, label_name):
    all_labels = np.array([])
    for block, markers in all_data.items():
        y = markers.get_labels(label_name)
        all_labels = np.concatenate((all_labels, y), axis=None)

    print(f"All labels shape: {all_labels.shape}")
    return all_labels


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
