
import numpy as np

from sklearn.decomposition import PCA
from cebra import CEBRA, KNNDecoder

def model_fit(
    neural_data,
    out_dim,
    num_hidden_units,
    behavioral_labels,
    max_iterations: int = 10,
    max_adapt_iterations: int = 10,
):
    single_cebra_model = CEBRA(
        # model_architecture = "offset10-model",
        batch_size=512,
        output_dimension=out_dim,
        max_iterations=max_iterations,
        num_hidden_units=num_hidden_units,
        max_adapt_iterations=max_adapt_iterations,
    )

    if behavioral_labels is None:
        single_cebra_model.fit(neural_data)
    else:
        single_cebra_model.fit(neural_data, behavioral_labels)
    # cebra.plot_loss(single_cebra_model)
    return single_cebra_model


def get_embeddings(
    train_data,
    val_data,
    train_labels,
    use_pca: bool = False,
    out_dim: int = 8,
    num_hidden_units: int = 256,
):
    if use_pca:
        # Run PCA
        pca = PCA(n_components=out_dim)
        pca = pca.fit(train_data)
        return pca.transform(train_data), pca.transform(val_data)

    single_cebra_model = model_fit(train_data, out_dim, num_hidden_units, train_labels)

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
            out_dim=output_dim,
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