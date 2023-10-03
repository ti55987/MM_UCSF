import numpy as np

def binary_label(all_labels, threshold=0.5):
    y_transformed = []
    c0, c1 = 0, 0
    for label in all_labels:
        if label < threshold:
            y_transformed.append(0)
            c0 += 1
        else:
            y_transformed.append(1)
            c1 += 1

    print(f"0 label: {c0}, 1 label: {c1}")
    return y_transformed


def get_tranformed_labels(condition_to_labels: dict):
    transformed = dict()
    for condition, label in condition_to_labels.items():
        transformed[condition] = binary_label(label)
    return transformed

# (TODO) combine attention + arousal to 2*2*2 = 8 classes
def get_categorical_labels(condition_to_labels: dict, threshold=0.5, valence_threshold=0.6):
    valence_labels = condition_to_labels['valence']
    arousal_labels = condition_to_labels['arousal']
    claz = []
    for i, v_label in enumerate(valence_labels):
       label = get_behavioral_labels(v_label, arousal_labels[i], valence_threshold, threshold)
       if label == 'nvla': # nvla
         claz.append(0)
       elif label == 'nvha': # nvha
         claz.append(1)
       elif label == 'hvla': # hvla
         claz.append(2)
       else: # hvha
         claz.append(3)

    return claz

def get_raw_labels(blocks, subject_data, num_repeat: int=1):
    v_labels = []
    a_labels = []
    attention_labels = []

    for b in blocks:
        block_data = subject_data[b]

        vl = np.repeat(block_data.get_labels().flatten(), num_repeat)
        arl = np.repeat(block_data.get_labels('arousal').flatten(), num_repeat)
        atl = np.repeat(block_data.get_labels('attention').flatten(), num_repeat)
        v_labels.extend(vl)
        a_labels.extend(arl)
        attention_labels.extend(atl)

    return v_labels, a_labels, attention_labels

def get_label_means(subject_data: dict):
    v_labels = []
    a_labels = []

    for _, data in subject_data.items():
        v_labels.extend(data.get_labels().flatten())
        a_labels.extend(data.get_labels("arousal").flatten())

    return np.mean(v_labels), np.mean(a_labels)

def get_behavioral_labels(valence, arousal, v_threshold=0.6, a_threshold=0.5):
    a_label = 'la' if arousal < a_threshold else 'ha'
    v_label = "nv" if valence < v_threshold else 'hv'

    return v_label + a_label

def print_label_count(label_list: list):
    num_to_count = {0:0, 1: 0, 2: 0, 3:0}
    for d in label_list:
        num_to_count[d] += 1

    print(num_to_count)
