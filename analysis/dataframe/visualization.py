import time

import pandas as pd

# plotly imports
import plotly.graph_objs as go
import seaborn as sns
from plotly.offline import iplot

# sklearn imports
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.manifold import TSNE  # T-Distributed Stochastic Neighbor Embedding
from umap import UMAP


def umap_2d(features: pd.DataFrame, colors: list, legend: str, title: str):
    import plotly.express as px

    umap2d = UMAP(n_components=2, init="random", random_state=0)
    proj_2d = umap2d.fit_transform(features)
    fig_2d = px.scatter(
        proj_2d, x=0, y=1, color=colors, labels={"color": legend}, title=title
    )
    fig_2d.show()


def pca_1d(features: pd.DataFrame, num_cluster: int, colors: list, title: str):
    # PCA with one principal component
    pca_1d = PCA(n_components=1)

    # This DataFrame holds that single principal component mentioned above
    s_1d = pd.DataFrame(pca_1d.fit_transform(features.drop(["Cluster"], axis=1)))
    # Rename the columns
    s_1d.columns = ["PC1_1d"]
    plotX = pd.concat([features, s_1d], axis=1, join="inner")

    # Used for 1-D
    plotX["dummy"] = 0

    data = []
    col_name = s_1d.columns[0]
    for i in range(num_cluster):
        c = plotX[plotX["Cluster"] == i]
        # trace is for 'Cluster i'
        trace1 = go.Scatter(
            x=c[col_name],
            y=c["dummy"],
            mode="markers",
            name=f"Cluster {i}",
            marker=dict(color=colors[i]),
            text=None,
        )
        data.append(trace1)

    layout = dict(
        title=title,
        xaxis=dict(title=col_name, ticklen=5, zeroline=False),
        yaxis=dict(title="", ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)

    iplot(fig)


def pca_2d(
    features: pd.DataFrame,
    num_cluster: int,
    colors: list,
    title: str,
    enabled_block: bool = False,
    **kwargs,
):
    # PCA with one principal component
    pca = PCA(n_components=2)

    # This DataFrame holds that single principal component mentioned above
    f = (
        features.drop(["Cluster", "Block"], axis=1)
        if enabled_block
        else features.drop(["Cluster"], axis=1)
    )
    sd = pd.DataFrame(pca.fit_transform(f))
    # Rename the columns
    sd.columns = ["PC1_2d", "PC2_2d"]
    plotX = pd.concat([features, sd], axis=1, join="inner")

    data = []
    col_name0 = sd.columns[0]
    col_name1 = sd.columns[1]
    for i in range(num_cluster):
        c = plotX[plotX["Cluster"] == i]
        # trace is for 'Cluster i'
        trace1 = go.Scatter(
            x=c[col_name0],
            y=c[col_name1],
            mode=kwargs["mode"] if "mode" in kwargs else "markers",
            name=f"Cluster {i}",
            marker=dict(
                color=colors[i],
                symbol=kwargs["symbol"] if "symbol" in kwargs else "diamond",
            ),
            text=c["Block"] if enabled_block else None,
            textfont=kwargs["textfont"] if "textfont" in kwargs else None,
        )
        data.append(trace1)

    layout = dict(
        title=title,
        xaxis=dict(title=col_name0, ticklen=5, zeroline=False),
        yaxis=dict(title=col_name1, ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)

    iplot(fig)


def pca_3d(features: pd.DataFrame, num_cluster: int, colors: list, title: str):
    # PCA with one principal component
    pca = PCA(n_components=3)

    # This DataFrame holds that single principal component mentioned above
    sd = pd.DataFrame(pca.fit_transform(features.drop(["Cluster"], axis=1)))
    # Rename the columns
    sd.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
    plotX = pd.concat([features, sd], axis=1, join="inner")

    data = []
    col_name0 = sd.columns[0]
    col_name1 = sd.columns[1]
    col_name2 = sd.columns[2]
    for i in range(num_cluster):
        c = plotX[plotX["Cluster"] == i]
        # trace is for 'Cluster i'
        trace1 = go.Scatter3d(
            x=c[col_name0],
            y=c[col_name1],
            z=c[col_name2],
            mode="markers",
            name=f"Cluster {i}",
            marker=dict(color=colors[i]),
            text=None,
        )
        data.append(trace1)

    layout = dict(
        title=title,
        xaxis=dict(title=col_name0, ticklen=5, zeroline=False),
        yaxis=dict(title=col_name1, ticklen=5, zeroline=False),
    )

    fig = dict(data=data, layout=layout)

    iplot(fig)


def tsne_2d(
    features: pd.DataFrame,
    title: str = "",
):
    # reference: https://builtin.com/data-science/tsne-python
    # visualization via t-SNE
    # Set our perplexity
    perplexity = 50
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=300)
    tsne_results = pd.DataFrame(tsne.fit_transform(features.drop(["Cluster"], axis=1)))

    print("t-SNE done! Time elapsed: {} seconds".format(time.time() - time_start))
    tsne_results.columns = ["TSNE1_2d", "TSNE2_2d"]
    plotX = pd.concat([features, tsne_results], axis=1, join="inner")

    sns.scatterplot(
        x="TSNE1_2d",
        y="TSNE2_2d",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=plotX,
        legend="full",
        alpha=0.3,
    )
