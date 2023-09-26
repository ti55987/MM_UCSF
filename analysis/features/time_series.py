import numpy as np

from labels import get_behavioral_labels

def get_time_series_data_by_channel(block_data, marker, channel_type: str):
    channel_data = []
    time_series_data = block_data.get_all_data()[marker] # (channel, time, trial)
    
    # loop through all trials: time
    for t in range(time_series_data.shape[2]):
        all_channel_data = []
        for i, c in enumerate(block_data.get_chanlocs(marker)):
            if not c.startswith(channel_type):
                continue
            # data is in the shape of (12288, 13)
            data = np.array(time_series_data[i][:, t])
            all_channel_data = (
                np.vstack((all_channel_data, data))
                if len(all_channel_data) > 0
                else data
            )

        all_channel_data = np.swapaxes(all_channel_data, 0, -1)

        channel_data = (
            np.vstack((channel_data, all_channel_data))
            if len(channel_data) > 0
            else all_channel_data
        )

    return channel_data


def get_block_time_series_features(blocks, subject_data, marker, channel):
    raw_data = []
    behavioral_labels = []

    for b in blocks:
        block_data = subject_data[b]
        psd_data = get_time_series_data_by_channel(block_data, marker, channel)

        v_label = block_data.get_labels()
        a_label = block_data.get_labels("arousal")
        # (TODO) extend the labels to match the time series dimention
        labels = [
            get_behavioral_labels(v_label[i], a_label[i]) for i in range(len(v_label))
        ]
        behavioral_labels.extend(labels)

        raw_data = np.vstack((psd_data, raw_data)) if len(raw_data) > 0 else psd_data

    return raw_data, behavioral_labels