import numpy as np

from collections import namedtuple

BP = namedtuple(
    "BP", ["xmin", "xmax", "units", "times", "srate", "systolic", "raw", "diastolic"]
)
ECG = namedtuple(
    "ECG",
    [
        "xmin",
        "xmax",
        "units",
        "times",
        "srate",
        "data",
        "avgHR",
        "LFHFratio",
        "LF",
        "HF",
    ],
)
EEG = namedtuple(
    "EEG",
    [
        "xmin",
        "xmax",
        "units",
        "uncorrsegs",
        "times",
        "srate",
        "data",
        "chanlocs",
        "artsegs",
    ],
)
EGG = namedtuple(
    "EGG",
    [
        "xmin",
        "xmax",
        "units",
        "times",
        "srate",
        "phase_degrees",
        "phase",
        "filtered",
        "data",
        "artifact",
        "amplitude",
    ],
)
EMG = namedtuple("EMG", ["xmin", "xmax", "units", "times", "srate", "data", "chanlocs"])
EOG = namedtuple("EOG", ["xmin", "xmax", "units", "times", "srate", "data", "chanlocs"])
GSR = namedtuple("GSR", ["xmin", "xmax", "units", "times", "srate", "raw", "phasic"])
Resp = namedtuple(
    "Resp", ["xmin", "xmax", "units", "times", "srate", "rateUnits", "rate", "data"]
)
TREV = namedtuple(
    "TREV",
    ["xmin", "xmax", "times", "srate", "data1Units", "data1", "data2Units", "data2"],
)
Behavior = namedtuple(
    "behavior", ["arousal", "attention", "block", "trialnum", "valence"]
)

ALL_MARKERS = [
    BP.__name__,
    ECG.__name__,
    EEG.__name__,
    EGG.__name__,
    EMG.__name__,
    EOG.__name__,
    GSR.__name__,
    Resp.__name__,
    TREV.__name__,
    Behavior.__name__,
]


class BioMarkers:
    def __init__(self, marker_data: dict):
        self.marker_to_namedtuple = {}
        for marker, val in marker_data.items():
            self.marker_to_namedtuple[marker] = namedtuple(marker, val.keys())(
                *val.values()
            )

    def get_labels(self):
        return np.array(getattr(self.marker_to_namedtuple[Behavior.__name__], 'valence'))

    def get_all_data(self):
        marker_to_data = {}
        for marker, data in self.marker_to_namedtuple.items():
            if marker == Behavior.__name__:
                continue

            field_name = self.get_data_field(marker)
            marker_to_data[marker] = np.array(getattr(data, field_name))
        return marker_to_data

    def get_data_field(self, marker):
        if marker == TREV.__name__ :
            return 'data1'
        elif marker == BP.__name__ or marker == GSR.__name__:
            return 'raw'

        return 'data'

    def get_data(self, marker_name):
        print(f"Get {marker_name} data...")
        data  = self.marker_to_namedtuple[marker_name]

        print(f"All Fields: {data._fields}")
        field = input("Enter the field to inspect:")
        npdata = np.array(getattr(data, field))

        print(f"The shape of the data: {npdata.shape}")
        return npdata, field

    def print_marker(self, marker_name):
        print(f"Inspecting {marker_name} data...")
        data = self.marker_to_namedtuple[marker_name]
        exit = False
        while not exit:
            print(f"All Fields: {data._fields}")
            field = input("Enter the field to inspect:")
            print(getattr(data, field))
            exit = input("Exit inpecting this marker? [y/n]: ") == "y"
