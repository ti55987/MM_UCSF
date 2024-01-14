import numpy as np

from sklearn.utils import shuffle

class DatasetBuilder():
    def __init__(self, num_labels: int, val_indexes_group: list):
        self.shuffled_val_indexes = []
        self.shuffled_train_indexes = []
        self.no_shuffled_val_indxes = []
        self.no_shuffled_train_indxes = []

        for val_indexes in val_indexes_group:
            train_indexes = list(set(range(num_labels)) - set(val_indexes))
            self.shuffled_train_indexes.append(shuffle(train_indexes, random_state=0))
            self.shuffled_val_indexes.append(shuffle(val_indexes, random_state=0))
            self.no_shuffled_train_indxes.append(train_indexes)
            self.no_shuffled_val_indxes.append(val_indexes)      


    def get_shuffled_indexes(self):
        return self.shuffled_train_indexes, self.shuffled_val_indexes
    
    # (TODO) clean up the interface
    def train_test_split(self, data, ecg_data, behavioral_labels, no_shuffle: bool=True):
        dataset = []
        val_indxes = self.no_shuffled_val_indxes if no_shuffle else self.shuffled_val_indexes
        train_indxes = self.no_shuffled_train_indxes if no_shuffle else self.shuffled_train_indexes
        
        for idx, val_indexes in enumerate(val_indxes):
            train_indexes = train_indxes[idx]
            train_labels = np.array(behavioral_labels)[train_indxes[idx]]
            val_label = np.array(behavioral_labels)[val_indexes]

            train_data = [data[train_indexes], ecg_data[train_indexes]] if len(ecg_data) > 0 else [data[train_indexes]]
            val_data = [data[val_indexes], ecg_data[val_indexes]] if len(ecg_data) > 0 else [data[val_indexes]]
            
            dataset.append((train_data, train_labels, val_data, val_label))
        
        return dataset   