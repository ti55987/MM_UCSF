import glob
import pandas as pd

from labels import get_categorical_labels


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


def get_labels_from_result(result: pd.DataFrame):
    all_label_array = {
        "valence": result["Valence"].values,
        "arousal": result["Arousal"].values,
        "attention": result["Attention"].values,
    }
    label_list = get_categorical_labels(all_label_array, valence_threshold=0.65)
    return all_label_array, label_list


def get_features_from_result(result: pd.DataFrame):
    all_feature_array = result.drop(
        ["Subject", "Unnamed: 0", "Valence", "Arousal", "Attention"], axis=1
    )
    all_feature_array = all_feature_array[
        all_feature_array.columns.drop(list(all_feature_array.filter(regex="BETA")))
    ]

    feature_names = all_feature_array.columns
    return all_feature_array, feature_names
    print(all_feature_array.shape, len(feature_names), len(label_list))
