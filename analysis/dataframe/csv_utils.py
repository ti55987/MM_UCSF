import glob

import pandas as pd
from labels import get_categorical_labels
from sklearn.preprocessing import StandardScaler  # used for 'Feature Scaling'


def load_data_from_csv(dir_name: str):
    all_files = glob.glob(dir_name + "/*.csv")
    frames = []
    for fn in all_files:
        df = pd.read_csv(fn)
        if fn == "extracted_features/three_subjects_eeg_ecg_emg.csv":
            # get rid of 1004
            mask = df["Subject"].isin(["../2000_CleanData", "../2001_CleanData"])
            df = df[mask]
        frames.append(df)

    return pd.concat(frames)


def get_labels_from_result(result: pd.DataFrame, valence_threshold=0.6):
    all_label_array = {
        "valence": result["Valence"].values,
        "arousal": result["Arousal"].values,
        "attention": result["Attention"].values,
    }
    label_list = get_categorical_labels(
        all_label_array, valence_threshold=valence_threshold
    )
    return all_label_array, label_list


def get_features_from_result(
    result: pd.DataFrame,
    dropped_columns=["Subject", "Unnamed: 0", "Valence", "Arousal", "Attention"],
    should_drop_beta: bool = True,
):
    all_feature_array = result.drop(dropped_columns, axis=1)

    if should_drop_beta:
        all_feature_array = all_feature_array[
            all_feature_array.columns.drop(list(all_feature_array.filter(regex="BETA")))
        ]

    feature_names = all_feature_array.columns
    return all_feature_array, feature_names


def get_filtered_data(
    result: pd.DataFrame, subjects: list, pattern: str = ".*(?<!BETA2)$"
):
    if len(subjects) > 0:
        mask = result["Subject"].isin(subjects)
        result = result[mask]

    _, label_list = get_labels_from_result(result, 0.65)
    all_feature_array, feature_names = get_features_from_result(
        result, ["Subject", "Valence", "Arousal", "Attention"], False
    )
    # all_feature_array = all_feature_array.drop(["index"], axis=1)
    feature_names = all_feature_array.columns
    print(all_feature_array.shape, len(feature_names), len(label_list))

    # Initialize our scaler
    scaler = StandardScaler()
    # Scale each column in numer
    normalized_eeg_features = pd.DataFrame(
        scaler.fit_transform(all_feature_array.filter(regex=pattern))
    )
    return normalized_eeg_features, result, label_list
