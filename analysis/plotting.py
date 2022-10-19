import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

from biomarkers import (
    EEG_CHANEL_NAMES,
    EEG_NUM_CHANNELS,
    EEG_MONTAGES,
)
from calculate_correlation import (
    EEG_BANDS_NAMES,
    EEG_BANDS_LIST,
    get_pearson_corr_with_stats_features,
    get_eeg_features_means,
)
from feature_extraction import EEG_BANDS, Feature


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


def get_eeg_pearson_correlation_series_by_block(
    feature_to_pc: np.ndarray,
    all_block_names: list,
    block_num: int = 0,
    num_channel: int = 128,
):
    ser_list = []
    block_name = all_block_names[block_num]
    index = list(range(1, num_channel + 1, 1))
    for f in EEG_BANDS_LIST:
        pearson_corr = feature_to_pc[f][:, block_num]
        ser = pd.Series(data=pearson_corr, index=index)
        ser_list.append(
            {
                "marker_name": f"EEG {f.name} at block {block_name}",
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
                ax=axes[r][c], kind="barh", title=f"{marker_name} Pearson Correlation"
            )
            count += 1


def plot_pearson_correlation_table(
    label: str, feature_to_pc: dict, all_block_names: list, k: int = 1
):
    means = get_eeg_features_means(feature_to_pc, all_block_names, k)

    col_lables = []
    col_lables.extend(all_block_names)
    col_lables.append("average")
    _plot_table(f"{label} Pearson Correlation", means, EEG_BANDS_NAMES, col_lables)


def plot_pearson_correlation_table_by_features(
    label: str,
    feature_to_pc: dict,
    all_block_names: list,
    features: list,
    channel_num: int = 0,
):
    means = np.zeros((len(features), len(all_block_names) + 1))
    i = 0
    row_labels = []
    for f in features:
        row_labels.append(f.name)
        data = np.round_(feature_to_pc[f][channel_num, :], decimals=3)
        # means[i] = np.round_(feature_to_pc[f][channel_num, :], decimals=3)
        avg = np.round_(np.mean(data), decimals=3)
        data = np.append(data, avg)
        means[i] = data
        i += 1

    col_lables = []
    col_lables.extend(all_block_names)
    col_lables.append("average")
    _plot_table(f"{label} Pearson Correlation", means, row_labels, col_lables)


def _plot_table(title: str, cell_values: list, row_labels: list, col_lables: list):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold")
    cellcolours = np.empty_like(cell_values, dtype="object")
    for i, cl in enumerate(row_labels):
        for j, _ in enumerate(col_lables):
            if cell_values[i][j] > 0.5:
                cellcolours[i, j] = "mistyrose"
            elif cell_values[i][j] < -0.5:
                cellcolours[i, j] = "skyblue"
            else:
                cellcolours[i, j] = "w"

    table = ax.table(
        cellText=cell_values,
        rowLabels=row_labels,
        colLabels=col_lables,
        cellColours=cellcolours,
        rowColours=["palegreen"] * cell_values.shape[0],
        colColours=["palegreen"] * cell_values.shape[1],
        cellLoc="center",
        loc="upper left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    plt.show()


def plot_eeg_topomap_one_block(
    condition: str,
    spectral_feature: Feature,
    feature_to_pc: dict,
    all_block_names: list,
    num_epochs: int = 13,
):
    fig, axes = plt.subplots(
        1,
        11,
        figsize=(30, 5),
        gridspec_kw={"width_ratios": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]},
    )

    fig.suptitle(
        f"{spectral_feature.name} <> {condition} Pearson Correlation",
        fontsize=40,
        y=1.3,
    )

    _plot_eeg_topomap(
        feature_to_pc[spectral_feature], all_block_names, axes, num_epochs
    )


def plot_eeg_topomap_all_blocks(
    condition: str,
    feature_to_pc: dict,
    num_epochs: int = 130,
):
    fig, axes = plt.subplots(
        1,
        7,
        figsize=(30, 5),
        gridspec_kw={"width_ratios": [3, 3, 3, 3, 3, 3, 1]},
    )

    fig.suptitle(
        f"EEG spectral feature vs {condition} Pearson Correlation", fontsize=40, y=1.3
    )

    all_features = []
    all_block_pc_mean = np.array([])
    for f in list(EEG_BANDS.keys()):
        all_features.append(f.name)
        pearson_corr = np.mean(feature_to_pc[f], axis=1)
        all_block_pc_mean = (
            pearson_corr
            if len(all_block_pc_mean) == 0
            else np.vstack((all_block_pc_mean, pearson_corr))
        )

    all_block_pc_mean = np.swapaxes(all_block_pc_mean, 0, 1)

    _plot_eeg_topomap(all_block_pc_mean, all_features, axes, num_epochs)


def _plot_eeg_topomap(data, xlables, axes, num_epochs):
    for i in range(len(xlables)):
        axes[i].set_xlabel(xlables[i], fontsize=25)
        axes[i].xaxis.set_label_position("top")

    sampling_freq = 1  # in Hertz
    info = mne.create_info(
        ch_names=EEG_CHANEL_NAMES,
        sfreq=sampling_freq,
        ch_types=["eeg"] * EEG_NUM_CHANNELS,
    )
    info.set_montage(EEG_MONTAGES)

    evoked_array = mne.EvokedArray(
        data,
        info,
        tmin=1,
        nave=num_epochs,
        comment="simulated",
    )
    efig = evoked_array.plot_topomap(
        axes=axes, time_format="", ch_type="eeg", units="score", scalings=1, show=False
    )
    efig.axes[-1].set_title("Score", fontsize=25)
    efig.axes[-1].tick_params(labelsize=25)
    plt.show()
