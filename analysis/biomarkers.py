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
EEG_NUM_CHANNELS = 128
EEG_MONTAGES = "biosemi128"
EEG_CHANEL_NAMES = [
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10",
    "A11",
    "A12",
    "A13",
    "A14",
    "A15",
    "A16",
    "A17",
    "A18",
    "A19",
    "A20",
    "A21",
    "A22",
    "A23",
    "A24",
    "A25",
    "A26",
    "A27",
    "A28",
    "A29",
    "A30",
    "A31",
    "A32",
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B15",
    "B16",
    "B17",
    "B18",
    "B19",
    "B20",
    "B21",
    "B22",
    "B23",
    "B24",
    "B25",
    "B26",
    "B27",
    "B28",
    "B29",
    "B30",
    "B31",
    "B32",
    "C1",
    "C2",
    "C3",
    "C4",
    "C5",
    "C6",
    "C7",
    "C8",
    "C9",
    "C10",
    "C11",
    "C12",
    "C13",
    "C14",
    "C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C30",
    "C31",
    "C32",
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    "D6",
    "D7",
    "D8",
    "D9",
    "D10",
    "D11",
    "D12",
    "D13",
    "D14",
    "D15",
    "D16",
    "D17",
    "D18",
    "D19",
    "D20",
    "D21",
    "D22",
    "D23",
    "D24",
    "D25",
    "D26",
    "D27",
    "D28",
    "D29",
    "D30",
    "D31",
    "D32",
]
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

BEHAVIOR_LIST = ["valence", "arousal", "attention"]

MARKER_TO_CHANNEL_NAMES = {
    EEG.__name__: EEG_CHANEL_NAMES,
    EMG.__name__: ["Corrugator", "Zygomaticus"],
    EOG.__name__: ["LEOG", "VEOG"],
}


class BioMarkersInterface:
    def get_labels(self, name="valence"):
        """Get the behavior lables"""
        pass

    def get_block_name(self) -> str:
        """Get the block name e.g. audio-hvha"""
        pass

    def get_all_data(self) -> dict:
        """Get all the data with key as marker name and data as numpy.array"""
        pass

    def get_chanlocs(self, marker: str) -> list:
        pass

    def get_times(self, marker: str) -> list:
        pass

    def get_data_field(self, marker):
        if marker == TREV.__name__:
            return "data2"
        elif marker == BP.__name__ or marker == GSR.__name__:
            return "raw"
        # elif marker == Resp.__name__:
        #     return "rate"

        return "data"


class Mat73BioMarkers(BioMarkersInterface):
    def __init__(self, marker_data: dict):
        self.marker_to_namedtuple = {}
        for marker, val in marker_data.items():
            self.marker_to_namedtuple[marker] = namedtuple(marker, val.keys())(
                *val.values()
            )

    def get_labels(self, name="valence"):
        return np.array(getattr(self.marker_to_namedtuple[Behavior.__name__], name))

    def get_block_name(self) -> str:
        return getattr(self.marker_to_namedtuple[Behavior.__name__], "block")

    def get_times(self, marker: str) -> list:
        return getattr(self.marker_to_namedtuple[marker], "times")

    def get_chanlocs(self, marker: str):
        if marker == BP.__name__:
            return ["Systolic", "Diastolic"]
        elif marker == ECG.__name__:
            return ["HF", "LF", "LFHFratio", "avgHR"]
        elif marker == EGG.__name__:
            return ["filtered", "phase", "amplitude"]
        elif marker in [EEG.__name__, EMG.__name__, EOG.__name__]:
            locs = getattr(self.marker_to_namedtuple[marker], "chanlocs")["labels"]
            if marker == EEG.__name__:
                return [l["labels"] for l in locs]
            return locs["labels"]

        return [self.get_data_field(marker)]

    def get_all_data(self):
        marker_to_data = {}
        for marker, data in self.marker_to_namedtuple.items():
            if marker == Behavior.__name__:
                continue

            if marker == BP.__name__:
                marker_to_data[marker] = np.stack(
                    (
                        np.array(getattr(data, "systolic")),
                        np.array(getattr(data, "diastolic")),
                    ),
                )
            elif marker == ECG.__name__:
                marker_to_data[marker] = np.stack(
                    (
                        np.array(getattr(data, "HF")),
                        np.array(getattr(data, "LF")),
                        np.array(getattr(data, "LFHFratio")),
                        np.array(getattr(data, "avgHR")),
                    ),
                )
            else:
                field_name = self.get_data_field(marker)
                marker_to_data[marker] = np.array(getattr(data, field_name))

        return marker_to_data

    def get_data(self, marker_name):
        print(f"Get {marker_name} data...")
        data = self.marker_to_namedtuple[marker_name]

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


class SIOBioMarkers(BioMarkersInterface):
    def __init__(self, marker_data):
        self.marker_to_data = {}
        for marker in ALL_MARKERS:
            self.marker_to_data[marker] = marker_data[marker][0][0]

    def get_labels(self, name="valence"):
        return self.marker_to_data[Behavior.__name__][name].item()

    def get_block_name(self) -> str:
        return self.marker_to_data[Behavior.__name__]["block"].item()[0]

    def get_times(self, marker: str) -> list:
        return self.marker_to_data[marker]["times"].item()[0]

    def get_chanlocs(self, marker: str):
        if marker == BP.__name__:
            return ["Systolic", "Diastolic"]
        elif marker == ECG.__name__:
            return ["HF", "LF", "LFHFratio", "avgHR"]
        elif marker == EGG.__name__:
            return ["filtered", "phase", "amplitude"]
        elif marker in [EEG.__name__, EMG.__name__, EOG.__name__]:
            return [i[0][0] for i in self.marker_to_data[marker]["chanlocs"].item()[0]]

        return [self.get_data_field(marker)]

    def get_all_data(self):
        marker_to_raw_data = {}
        for marker, data in self.marker_to_data.items():
            if marker == Behavior.__name__:
                continue

            if marker == BP.__name__:
                marker_to_raw_data[marker] = np.concatenate(
                    (data["systolic"].item(), data["diastolic"].item()), axis=0
                )
            elif marker == ECG.__name__:
                marker_to_raw_data[marker] = np.concatenate(
                    (
                        data["HF"].item(),
                        data["LF"].item(),
                        data["LFHFratio"].item(),
                        data["avgHR"].item(),
                    ),
                    axis=0,
                )
            elif marker == EGG.__name__:
                marker_to_raw_data[marker] = np.concatenate(
                    (
                        data["filtered"].item(),
                        data["phase"].item(),
                        data["amplitude"].item(),
                    ),
                    axis=0,
                )
            else:
                field_name = self.get_data_field(marker)
                marker_to_raw_data[marker] = data[field_name].item()
        return marker_to_raw_data
