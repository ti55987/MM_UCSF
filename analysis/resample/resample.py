import random
import numpy as np

random.seed(33)

def upsample_by_attention(attention, num_output):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 100))
    weights = scaler.fit_transform(attention.reshape(-1, 1))

    resampled_features = random.choices(
        np.arange(0, len(attention), 1), weights=weights, k=num_output
    )

    return np.array(resampled_features)

def get_resampled_list_index(train_indexes, attention_labels):
    # exclude validation
    train_attention_labels = np.array(attention_labels)[train_indexes]
    # up sample 4 times by attention labels
    resampled_list = upsample_by_attention(
        train_attention_labels, len(train_attention_labels) * 4
    )
    return resampled_list

def get_consecutive_validation_indexes(
    num_train_set: int = 52,
    num_block: int = 4,
    num_slice_per_trial: int = 1,
    start_trial_in_block: int = 1,
    n_step_trial: int = 3.0,
):
    # generate random integer values
    indexes = []
    # extract 3 trials per block
    num_trial_per_block = int(num_train_set / num_block)
    for b in range(0, num_train_set, num_trial_per_block):
        start = b + start_trial_in_block * num_slice_per_trial
        end = start + n_step_trial * num_slice_per_trial
        val_indexes = np.arange(start, end, dtype=int)
        indexes.extend(val_indexes)

    return indexes

# generate random validation index
def get_validation_indexes(num_train_set: int = 52, num_block: int = 4):
    # generate random integer values
    indexes = []
    # extract 3 samples per block
    k = 3
    num_trial_per_block = int(num_train_set / num_block)
    for i in range(num_block):
        start = i * num_trial_per_block
        end = (i + 1) * num_trial_per_block
        population = list(range(start, end))

        indexes.extend(random.sample(population, k))

    return indexes