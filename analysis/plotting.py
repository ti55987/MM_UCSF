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
    get_eeg_features_means,
)
from feature_extraction import EEG_BANDS, Feature
from biomarkers import BioMarkersInterface


def plot_time_series_by_epoch(
    biomarker: BioMarkersInterface, marker: str, block: str, ch: int
):
    fig, axes = plt.subplots(
        13,
        1,
        figsize=(10, 30),
    )

    times = biomarker.get_times(marker)
    data = biomarker.get_all_data()[marker]
    channel_name = biomarker.get_chanlocs(marker)[ch]
    axes[0].set_title(f"{channel_name} {block}")
    for i in range(data.shape[2]):
        axes[i].plot(times, data[ch, :, i])
        axes[i].set_ylabel(f"Epoch {i+1}")
        if i < data.shape[2] - 1:
            axes[i].get_xaxis().set_visible(False)

    axes[-1].set_xlabel("time (ms)", visible=True)

    plt.show()


def get_eeg_pearson_correlation_series_all_blocks(
    feature_to_rp: np.ndarray,
    channel_names: list,
    k: int = 10,
):
    """Get eeg pd series for plotting.

    Parameters
    ----------
    feature_to_rp : map
        key: feature name
        value: (num_channels, [R_value, P_value])
    channel_names: the names of the channel
    k: int
        Top k positive and top k negative.

    Returns
    -------
    data : array
        eeg pd series
    """
    ser_list = []
    for f in list(EEG_BANDS.keys()):
        pearson_corr = feature_to_rp[f][:, 0]
        ser = pd.Series(data=pearson_corr, index=channel_names)

        value = ser
        if k > 0:
            value = pd.concat([ser.nlargest(int(k / 2)), ser.nsmallest(int(k / 2))])

        ser_list.append(
            {
                "marker_name": f"EEG {f.name} Pearson Correlation R value",
                "value": value,
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


def plot_eeg_pearson_correlation_table(
    label: str, feature_to_pc: dict, all_block_names: list, k: int = 1
):
    """Plot top k eeg pearson correlation means table

    Parameters
    ----------
    label: str
        The condition
    feature_to_pc : map
        key: feature name
        value: (num_channels, num_blocks)
    all_block_names: list
        The names of the blocks
    k: int
        Top k positive and top k negative.
    """
    means = get_eeg_features_means(feature_to_pc, all_block_names, k)

    col_lables = []
    col_lables.extend(all_block_names)
    col_lables.append("average")
    plot_table(f"{label} Pearson Correlation", means, EEG_BANDS_NAMES, col_lables)


def plot_top_chaneels_by_p_value(
    feature_to_rp: dict,
    channel_names: list,
    features: list,
    condition: str,
    k: int = 10,
):
    for f in features:
        top_k_by_r = np.argpartition(feature_to_rp[f][:, 0], -k)[-k:]
        # bottom_k = np.argpartition(feature_to_rp[f][:, 1], k)[:k]
        channels = [channel_names[i] for i in top_k_by_r]
        data = np.take(feature_to_rp[f], top_k_by_r, 0)
        data = np.round_(data, decimals=3)
        plot_table(
            f"{f.name} {condition} Pearson Correlation",
            data,
            channels,
            ["R-Value", "P-Value"],
            True,
            2,
        )


def plot_pearson_correlation_table_by_channel(
    label: str,
    feature_to_pc: dict,
    col_lables: list,
    features: list,
    channel_num: int = 0,
    with_pr_value: bool = False,
):
    """Plot top k eeg pearson correlation means table

    Parameters
    ----------
    label: str
        The condition
    feature_to_pc : map
        key: feature name
        value: (num_channels, num_blocks) or (num_channels, [r_value, p_value])
    features: list
        The names of the features
    channel_num: int
        The number of channel
    with_pr_value: bool
        If the feature_to_pc contains p values
    """
    means = np.zeros((len(features), len(col_lables)))
    i = 0
    row_labels = []
    for f in features:
        row_labels.append(f.name)
        means[i] = np.round_(feature_to_pc[f][channel_num, :], decimals=3)
        i += 1

    f_width = 2 if with_pr_value else 12
    plot_table(
        f"{label} Pearson Correlation",
        means,
        row_labels,
        col_lables,
        with_pr_value,
        f_width,
    )


def _get_pr_colors(cell_values, row_labels, col_lables):
    cellcolours = np.empty_like(cell_values, dtype="object")
    for i, cl in enumerate(row_labels):
        cellcolours[i, 0] = _color_r_value(cell_values[i][0])
        cellcolours[i, 1] = _color_p_value(cell_values[i][1])

    return cellcolours


def _get_r_colors(cell_values, row_labels, col_lables):
    cellcolours = np.empty_like(cell_values, dtype="object")
    for i, cl in enumerate(row_labels):
        for j, _ in enumerate(col_lables):
            cellcolours[i, j] = _color_r_value(cell_values[i][j])

    return cellcolours


def plot_table(
    title: str,
    cell_values: list,
    row_labels: list,
    col_lables: list,
    with_p_value: bool = False,
    fig_width: int = 12,
):
    fig, ax = plt.subplots(figsize=(fig_width, 2))
    ax.set_axis_off()
    ax.set_title(title, fontweight="bold")

    cellcolours = (
        _get_pr_colors(cell_values, row_labels, col_lables)
        if with_p_value
        else _get_r_colors(cell_values, row_labels, col_lables)
    )

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


def _color_p_value(value) -> str:
    return "mistyrose" if value < 0.05 else "w"


def _color_r_value(value) -> str:
    if value > 0.5:
        return "mistyrose"
    elif value < -0.5:
        return "skyblue"
    else:
        return "w"


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

    plot_eeg_topomap(feature_to_pc[spectral_feature], all_block_names, axes, num_epochs)


def plot_eeg_topomap_all_blocks(
    condition: str,
    feature_to_rp: dict,
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
        pearson_corr = feature_to_rp[f][:, 0]
        all_block_pc_mean = (
            pearson_corr
            if len(all_block_pc_mean) == 0
            else np.vstack((all_block_pc_mean, pearson_corr))
        )

    all_block_pc_mean = np.swapaxes(all_block_pc_mean, 0, 1)

    plot_eeg_topomap(all_block_pc_mean, all_features, axes, num_epochs)


def plot_eeg_topomap(data, xlables, axes, num_epochs):
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
