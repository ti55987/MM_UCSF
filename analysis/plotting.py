import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import seaborn as sns

from biomarkers import (
    EEG_CHANEL_NAMES,
    EEG_NUM_CHANNELS,
    EEG_MONTAGES,
    BEHAVIOR_LIST,
)
from calculate_correlation import (
    EEG_BANDS_NAMES,
    get_eeg_features_means,
)
from feature_extraction import EEG_BANDS, Feature
from biomarkers import BioMarkersInterface

# basic color from https://matplotlib.org/stable/gallery/color/named_colors.html
COLORS = ["b", "c", "y", "m", "g", "r"]


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


def plot_pd_scatter_by_marker(marker: str, result: pd.DataFrame, all_trials: list = []):
    # basic color from https://matplotlib.org/stable/gallery/color/named_colors.html
    colors = COLORS
    feature_names = result.filter(regex=(f"{marker}.*")).columns
    feature_names = feature_names[:10]
    if len(all_trials) == 0:
        all_trials = result["Subject"].unique()[: len(colors)]

    for b in BEHAVIOR_LIST:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7))
        fig.suptitle(f"{marker} features vs {b}", fontsize=14, y=1)

        fidx = 0
        r, c = 0, 0
        for f in feature_names:
            cidx = 0
            r = int(fidx % 2)
            c = int(fidx / 2)

            for trial in all_trials:
                trial_name = trial.strip("../_CleanData").strip("../")
                df = result[result["Subject"] == trial]

                axes[r][c].scatter(
                    df[f].values,
                    df[b.capitalize()],
                    c=colors[cidx],
                    label=trial_name,
                    alpha=0.3,
                )
                axes[r][c].set_xlabel(f, fontsize=10)
                cidx += 1

            axes[r][c].legend()
            fidx += 1

        plt.show()


def plot_scatter_by_marker(
    marker: str, features_to_trials: dict, dir_name_to_labels: dict, data_name: str
):
    colors = COLORS
    for b in BEHAVIOR_LIST:
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 7))
        fig.suptitle(f"{marker} {data_name} features vs {b}", fontsize=14, y=1)

        fidx = 0
        r, c = 0, 0
        for f, trials in features_to_trials.items():
            cidx = 0
            r = int(fidx % 2)
            c = int(fidx / 2)

            for trial, val in trials.items():
                trial_name = trial.strip("../_CleanData").strip("../")
                labels = dir_name_to_labels[trial][b]
                axes[r][c].scatter(
                    val, labels, c=colors[cidx], label=trial_name, alpha=0.3
                )  # norm=mpl.colors.Normalize(vmin=0, vmax=2)
                axes[r][c].set_xlabel(f, fontsize=10)
                cidx += 1

            axes[r][c].legend()
            fidx += 1

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


def plot_k_chaneels_by_r_value(
    feature_to_rp: dict,
    channel_names: list,
    features: list,
    condition: str,
    is_top: bool = True,
    k: int = 10,
):
    for f in features:
        k_by_r = np.argpartition(feature_to_rp[f][:, 0], -k)[-k:]
        title = f"Top {k} {f.name} {condition} Pearson Correlation"
        if not is_top:
            k_by_r = np.argpartition(feature_to_rp[f][:, 0], k)[:k]
            title = f"Bottom {k} {f.name} {condition} Pearson Correlation"

        channels = [channel_names[i] for i in k_by_r]
        data = np.take(feature_to_rp[f], k_by_r, 0)
        data = np.round_(data, decimals=3)
        plot_table(
            title,
            data,
            channels,
            ["R-Value", "P-Value"],
            True,
            2,
        )


def plot_correlation_table_by_channel(
    label: str,
    feature_to_pc: dict,
    col_lables: list,
    features: list,
    channel_num: int = 0,
    with_pr_value: bool = False,
):
    """Plot pearson correlation means table

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

    f_width = 4 if with_pr_value else 12
    plot_table(
        f"{label} Correlation",
        means,
        row_labels,
        col_lables,
        with_pr_value,
        f_width,
    )


def _get_pr_colors(cell_values, row_labels, col_lables):
    cellcolours = np.empty_like(cell_values, dtype="object")
    for i, cl in enumerate(row_labels):
        for j, _ in enumerate(col_lables):
            if j % 2 == 0:
                cellcolours[i, j] = _color_r_value(cell_values[i][j])
            else:
                cellcolours[i, j] = _color_p_value(cell_values[i][j])

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
    all_features = list(feature_to_rp.keys())
    width_ratios = [3] * len(all_features)
    width_ratios.append(1)

    fig, axes = plt.subplots(
        1,
        len(all_features) + 1,
        figsize=(30, 5),
        gridspec_kw={"width_ratios": width_ratios},
    )

    fig.suptitle(
        f"EEG spectral feature vs {condition} Pearson Correlation", fontsize=40, y=1.3
    )

    all_block_pc_mean = np.array([])
    for f in all_features:
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

def subplot_confusion_matrix(cf,
                          ax=None,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent='all',
                          cbar=True,
                          cbar_ax=None,
                          xyticks=True,
                          vmin=0, vmax=3000,
                          cmap='Blues'):

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent == 'all':
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    elif percent == 'by_row':
        group_percentages = []
        cf_matrix = []
        for cfg in cf:
            values = ["{0:.2%}".format(value) for value in cfg/np.sum(cfg)]
            group_percentages.extend(values)
            cf_matrix.append([float("{0:.2}".format(value)) for value in cfg/np.sum(cfg)])
        
        cf = np.array(cf_matrix)
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    return sns.heatmap(
      cf,
      annot=box_labels,
      fmt="",
      cmap=cmap,
      cbar=cbar,
      cbar_ax=cbar_ax,
      vmin=vmin, vmax=vmax,
      xticklabels=categories,
      yticklabels=categories, 
      ax=ax)

def plot_roc_curve(predicted_labels, true_labels, method, label_type, channel, f):
    from sklearn.metrics import RocCurveDisplay, auc

    n_splits = len(predicted_labels)
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))

    tprs, aucs = [], []
    for fold in range(n_splits):
        viz = RocCurveDisplay.from_predictions(
            y_true=true_labels[fold],
            y_pred=predicted_labels[fold],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{method} {label_type} - {channel}:{f.name} \nMean ROC curve with variability",
    )
    ax.axis("square")
    ax.legend(loc="lower right")
    
    return fig

def set_pane_axis(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])

def plot_3d_embeddings(list_embedding_tuple, title):
    n_row, n_col = (2, 3) if len(list_embedding_tuple) == 6 else (1, 4)
    fig, axes = plt.subplots(
        nrows=n_row,
        sharey=True,
        ncols=n_col,
        figsize=(n_col * 5, n_row * 5),
        subplot_kw=dict(projection="3d"),
    )
    idx1, idx2, idx3 = (0, 1, 2)
    for idx, (title, embeddings, embedding_labels) in enumerate(list_embedding_tuple):
        y = axes.flat[idx].scatter(
            embeddings[:, idx1],
            embeddings[:, idx2],
            embeddings[:, idx3],
            cmap="cool",
            c=embedding_labels,
            s=5,
            vmin=0,
            vmax=1,
        )
        axes.flat[idx].set_title(title)
        yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(0, 1, 9))
        yc.ax.tick_params(labelsize=10)
        yc.ax.set_title("score", fontsize=10)    
        set_pane_axis(axes.flat[idx])        
        
    fig.suptitle(f'{title} Latents: (1,2,3)')


def plot_umap_embeddings(list_embedding_tuple):
    from umap import UMAP
    for (title, embeddings, embedding_labels) in list_embedding_tuple:
        if 'GAMMA' in title:
            components = embeddings
            color = embedding_labels
            break

    fig, axes = plt.subplots(
        nrows=3,
        sharey=True,
        ncols=2,
        figsize=(2 * 5, 3 * 5),
    )
    a = [0.0001, 0.001, 0.1, 1, 10, 50]
    for idx, ax in enumerate(axes.flat):
        umap2d = UMAP(n_components=2, a=0.0001, b=2)
        proj_2d = umap2d.fit_transform(components)
        y = ax.scatter(
            proj_2d[:, 0],
            proj_2d[:, 1],
            cmap="cool",
            c=color,
            s=5,
            vmin=0,
            vmax=1,
        )
        ax.set_title(title)
        yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(0, 1, 9))
        yc.ax.tick_params(labelsize=10)
        yc.ax.set_title("score", fontsize=10)

def umap_visualization(components):
    from umap import UMAP
    umap2d = UMAP(n_components=2, a=0.0001, b=2)
    return umap2d.fit_transform(components)

def tsne_visualization(components):
    from sklearn.manifold import TSNE
    tsne2d = TSNE(n_components=2, random_state=0) 
    return tsne2d.fit_transform(components) 

def plot_embeddings(list_embedding_tuple, method, label_type, visualization_func):
    n_row, n_col = (2, 3) if len(list_embedding_tuple) == 6 else (1, 4)
    fig, axes = plt.subplots(
        nrows=n_row,
        sharey=True,
        ncols=n_col,
        figsize=(n_col * 5, n_row * 5),
    )
    for idx, (title, embeddings, embedding_labels) in enumerate(list_embedding_tuple):
        proj_2d = visualization_func(embeddings)
        y = axes.flat[idx].scatter(
            proj_2d[:, 0],
            proj_2d[:, 1],
            cmap="cool",
            c=embedding_labels,
            s=5,
            vmin=0,
            vmax=1,
        )
        axes.flat[idx].set_title(title)
        yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(0, 1, 9))
        yc.ax.tick_params(labelsize=10)
        yc.ax.set_title("score", fontsize=10)    
        
    fig.suptitle(f'{visualization_func.__name__}:{method} embedding - {label_type}')   