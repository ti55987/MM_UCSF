
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from cebra import CEBRA, KNNDecoder
import cebra

def model_fit(
    neural_data,
    out_dim,
    num_hidden_units,
    behavioral_labels,
    max_iterations: int = 10000,
):
    single_cebra_model = CEBRA(
        # model_architecture = "offset10-model",
        batch_size=512,
        temperature_mode="auto",
        output_dimension=out_dim,
        max_iterations=max_iterations,
        num_hidden_units=num_hidden_units,
        verbose = False,
    )

    if behavioral_labels is None:
        single_cebra_model.fit(neural_data)
    else:
        single_cebra_model.fit(neural_data, behavioral_labels)
    
    cebra.plot_loss(single_cebra_model)
    return single_cebra_model

def plot_umap_embeddings(list_embedding_tuple, method, label_type):
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

def tsne_visualization(components):
    from sklearn.manifold import TSNE
    tsne2d = TSNE(n_components=2, random_state=0) 
    return tsne2d.fit_transform(components) 

def umap_visualization(components):
    from umap import UMAP
    umap2d = UMAP(n_components=2, a=0.0001, b=2)
    return umap2d.fit_transform(components)

def get_embeddings(
    train_data,
    val_data,
    train_labels,
    use_pca: bool = False,
    out_dim: int = 8,
    num_hidden_units: int = 256,
    max_iterations: int = 100,
):
    if use_pca:
        # Run PCA
        pca = PCA(n_components=out_dim)
        pca = pca.fit(train_data)
        return pca.transform(train_data), pca.transform(val_data)

    single_cebra_model = model_fit(train_data, out_dim, num_hidden_units, train_labels, max_iterations)

    # Calculate embedding
    embedding = single_cebra_model.transform(train_data)
    val_embedding = single_cebra_model.transform(val_data)
    return embedding, val_embedding

def run_knn_decoder(
    dataset,
    method,
    threshold,
    output_dim,
    max_hidden_units,
):
    y_pred, y_pred_cat, all_embeddings = [], [], []
    for _, (train_data, train_labels, val_data, _) in enumerate(dataset):
        embedding, val_embedding = get_embeddings(
            train_data=train_data,
            val_data=val_data,
            train_labels=train_labels,
            use_pca=(method == "PCA"),
            out_dim= 6 if method == "PCA" else output_dim,
            num_hidden_units=max_hidden_units,
        )
        all_embeddings.append(embedding)
        # 4. Train the decoder on training embedding and labels
        # train_true_cat = get_label_category(train_labels, label_type)
        decoder = KNNDecoder()
        decoder.fit(embedding, np.array(train_labels))

        # score = decoder.score(val_embedding, np.array(val_labels))
        prediction = decoder.predict(val_embedding)
        y_pred.append(prediction)
        y_pred_cat.append([0 if p < threshold else 1 for p in prediction])

    return y_pred, y_pred_cat, all_embeddings