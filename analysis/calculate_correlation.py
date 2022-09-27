import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr

from biomarkers import EEG, EMG, EOG
from data_utils import load_data_from_file, get_features_in_block

MULTIPLE_CHANNELS_SIGNAL = [EEG.__name__, EMG.__name__, EOG.__name__]


def calculateEEGPearson(eegData, labels):
    arrax = np.mean(eegData, axis=1)  # the eegData is in shape (128, 12288, 13)
    pearson_corr = []
    for i in range(arrax.shape[0]):
        corr = pearsonr(arrax[i, :], labels)
        print(f"channel {i} pearson correlation {corr[0]}")
        pearson_corr.append(corr[0])

    plt.plot(pearson_corr)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="the path to the .mat file")
    args = parser.parse_args()
    markers = load_data_from_file(args.file_name)

    while True:
        should_continue = input("Continue? [y/n]: ") == "y"
        if not should_continue:
            exit()

        try:
            y = markers.get_labels()
            print(f"lables {y}")
            marker_to_data = markers.get_all_data()

            all_features = get_features_in_block(marker_to_data)

            print(f"get all features: {all_features.shape}...")
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
