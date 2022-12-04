import pandas as pd
import numpy as np

from data_utils import (
    extract_features,
)
from calculate_correlation import (
    EEG_BANDS_LIST,
)
from biomarkers import (
    EEG,
)


def extract_features_by_markers(
    markers: list, dir_to_data: dict, dir_to_extracted: list
):
    """
    extract features from physiological signals
    """
    feature_names = []
    all_feature_array = []
    for m in markers:
        f_array, names = (
            extract_features(m, dir_to_data, EEG_BANDS_LIST, dir_to_extracted)
            if m == EEG.__name__
            else extract_features(m, dir_to_data, all_dir=dir_to_extracted)
        )
        print(f"extracted {m} stats or PSD features")
        all_feature_array.append(f_array)
        feature_names.extend(names)

    all_feature_array = np.concatenate(all_feature_array, axis=-1)

    return pd.DataFrame(all_feature_array, columns=feature_names)
