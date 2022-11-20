from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    LeakyReLU,
    MaxPool1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    AveragePooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from labels import get_tranformed_labels


def cnn_model(input_x: int, input_y: int):
    clear_session()
    model = Sequential(
        [
            Conv1D(filters=5, kernel_size=3, strides=1, input_shape=(input_x, input_y)),
            BatchNormalization(),
            LeakyReLU(),
            MaxPool1D(pool_size=2, strides=2),
            Conv1D(filters=5, kernel_size=3, strides=1),
            LeakyReLU(),
            MaxPool1D(pool_size=2, strides=2),
            Dropout(0.5),
            Conv1D(filters=5, kernel_size=3, strides=1),
            LeakyReLU(),
            AveragePooling1D(pool_size=2, strides=2),
            Dropout(0.5),
            Conv1D(filters=5, kernel_size=3, strides=1),
            LeakyReLU(),
            AveragePooling1D(pool_size=2, strides=2),
            Conv1D(filters=5, kernel_size=3, strides=1),
            LeakyReLU(),
            GlobalAveragePooling1D(),
            Dense(4, activation="sigmoid"),
        ]
    )

    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_with_cnn(
    input_x: int, input_y: int, data_array, label_array, group_array
) -> list:
    accuracy = []
    gkf = GroupKFold()
    for train_index, val_index in gkf.split(
        data_array, label_array, groups=group_array
    ):
        train_features, train_labels = data_array[train_index], label_array[train_index]
        val_features, val_labels = data_array[val_index], label_array[val_index]

        scaler = StandardScaler()
        train_features = scaler.fit_transform(
            train_features.reshape(-1, train_features.shape[-1])
        ).reshape(train_features.shape)
        val_features = scaler.transform(
            val_features.reshape(-1, val_features.shape[-1])
        ).reshape(val_features.shape)

        model = cnn_model(input_x, input_y)
        model.fit(
            train_features,
            train_labels,
            epochs=13,
            batch_size=50,
            validation_data=(val_features, val_labels),
        )
        accuracy.append(model.evaluate(val_features, val_labels)[1])
    return accuracy


def train_with_logistic(all_features, label_name, condition_to_labels, group_array):
    name_to_transformed = get_tranformed_labels(condition_to_labels)
    clf = LogisticRegression(max_iter=200)
    gkf = GroupKFold(n_splits=5)
    param_grid = {"classifier__C": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
    pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
    gscv = GridSearchCV(pipe, param_grid, cv=gkf, n_jobs=16)
    gscv.fit(all_features, name_to_transformed[label_name], groups=group_array)

    best_model = gscv.best_estimator_
    print(f"best model: {best_model}")
    print(gscv.best_score_)

    return best_model


def train_with_gradient_boosting_regressor(
    all_feature_array, all_label_array, bahavior
):
    X, y = all_feature_array, all_label_array[bahavior]
    print(X.shape, y.shape)

    # (TODO) Using sklearn StandardScaler to normlize the data
    # train_test_split
    X_train, X_test = X[:312], X[312:]
    y_train, y_test = y[:312], y[312:]
    # lightGBM
    # Plot confusion metrics
    est = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss="ls"
    ).fit(X_train, y_train)
    print(mean_squared_error(y_test, est.predict(X_test)))

    # feat_importances = pd.Series(est.feature_importances_, index=feature_names)
    # feat_importances.nlargest(10).plot(
    #     kind="barh", title=f"{bahavior} Feature Importance"
    # )
    return est
