import pandas as pd
import matplotlib.pyplot as plt

from calculate_correlation import (
    get_pearson_corr_with_stats_features,
    get_eeg_spectral_pearson_correlation,
)
from feature_extraction import EEG_BANDS


def get_pearson_correlation_series(
    marker_to_block: dict, markers: list, labels: list, num_channel: int = 1
):
    ser_list = []
    for marker_name in markers:
        data = marker_to_block[marker_name]
        for ch in range(num_channel):
            peaerson_corr, features = get_pearson_corr_with_stats_features(
                data, labels, ch
            )
            ser = pd.Series(data=peaerson_corr, index=features)

            name = marker_name if num_channel == 1 else f"{marker_name}:{ch}"
            ser_list.append({"marker_name": name, "value": ser})

    return ser_list


def get_eeg_pearson_correlation_series(
    all_block: dict, labels: list, num_channel: int = 128
):
    ser_list = []
    index = list(range(1, num_channel + 1, 1))
    for f in list(EEG_BANDS.keys()):
        pearson_corr = get_eeg_spectral_pearson_correlation(all_block, labels, f)
        ser = pd.Series(data=pearson_corr, index=index)
        print(f"The mean {ser.mean()} over feature {f.name}")

        ser_list.append(
            {
                "marker_name": f"EEG {f.name}",
                "value": pd.concat([ser.nlargest(5), ser.nsmallest(5)]),
            }
        )

    return ser_list


def plot_series(nrow: int, ncol: int, ser_list):
    # make a list of all dataframes
    fig, axes = plt.subplots(nrow, ncol)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.set_size_inches(15, 15)
    # plot counter
    count = 0
    for r in range(nrow):
        for c in range(ncol):
            marker_name = ser_list[count]["marker_name"]
            ser_list[count]["value"].plot(
                ax=axes[r][c], kind="barh", title=f"{marker_name} Spearmanr Correlation"
            )
            count += 1
