import numpy as np

from sklearn.utils import shuffle

class EEGDataset():
    def __init__(self, val_indexes_group: list):
        self.val_indexes_group = val_indexes_group        
        self.shuffled_val_indexes = []
        self.shuffled_train_indexes = []

    def get_shuffled_indexes(self, num_labels, enabled_shuffle: bool=True):
        for val_indexes in self.val_indexes_group:
            train_indexes = list(set(range(num_labels)) - set(val_indexes))
            if enabled_shuffle:
                self.shuffled_train_indexes.append(shuffle(train_indexes, random_state=0))
                self.shuffled_val_indexes.append(shuffle(val_indexes, random_state=0))
            else:
                self.shuffled_train_indexes.append(train_indexes)
                self.shuffled_val_indexes.append(val_indexes)                
        
        return self.shuffled_train_indexes, self.shuffled_val_indexes
    
    def train_test_split(self, data, behavioral_labels):
        dataset = []
        for idx, val_indexes in enumerate(self.shuffled_val_indexes):
            train_indexes = self.shuffled_train_indexes[idx]
            train_labels = np.array(behavioral_labels)[train_indexes]
            train_data = data[train_indexes]

            val_data = data[val_indexes]
            val_label = np.array(behavioral_labels)[val_indexes]
            dataset.append((train_data, train_labels, val_data, val_label))
        
        return dataset   