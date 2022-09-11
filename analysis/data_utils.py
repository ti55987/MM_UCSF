import mat73
import glob
import numpy as np

from feature_extraction import concatenate_features
from biomarkers import BioMarkers

def load_data_from_file(file_name: str) -> BioMarkers:
    raw_data = mat73.loadmat(file_name)
    signal = raw_data["Signal"]

    print(f"Complete loading {len(signal)} markers")
    return BioMarkers(signal)

def load_data_from_dir(dir_name: str) -> dict:
    # All files and directories ending with .mat and that don't begin with a dot:
    all_files = glob.glob(dir_name + "/*.mat")
    all_block = {}
    for f in all_files:
        markers = load_data_from_file(f)
        block_name = markers.get_block_name()
        print(f'Loaded {markers.get_block_name()} block')
        all_block[block_name] = markers

    return all_block

def get_features(data_array):
    features=[]
    for data in data_array:
        features.append(concatenate_features(data))

    features=np.array(features)
    return features

def get_features_in_block(marker_to_data: dict) -> np.array:
    all_features = np.array([])
    for marker, data in marker_to_data.items():
        data = np.swapaxes(data,0,-1) # (num_features, num_blocks) => (num_blocks, num_features)

        print(f'get {marker} features {data.shape}...')
        marker_features = get_features(data)
        print(f"feature shape {marker_features.shape}")

        if all_features.ndim > 1:
            all_features = np.concatenate((all_features, marker_features), axis=1)
        else:
            all_features = marker_features

    return all_features

def get_features_in_all_blocks(all_blocks: dict) -> np.array:
    all_features = []
    for block_name, markers in all_blocks.items():
        marker_to_data = markers.get_all_data()
        block_features = get_features_in_block(marker_to_data)
        print(f'{block_name} block has features: {block_features.shape}...')
        all_features.append(all_features, block_features)

    print(f"All feature shape {all_features.shape}")
    return np.array(all_features)
