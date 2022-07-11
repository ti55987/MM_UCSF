import mat73
import argparse

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

    def get_bp(self) -> BP:
        return self.marker_to_namedtuple["BP"]

    def print_marker(self, marker_name):
        print(f"Inspecting {marker_name} data...")
        data = self.marker_to_namedtuple[marker_name]
        exit = False
        while not exit:
            print(f"All Fields: {data._fields}")
            field = input("Enter the field to inspect:")
            print(getattr(data, field))
            exit = input("Exit inpecting this marker? [y/n]: ") == "y"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="the path to the .mat file")
    parser.add_argument("--interactive", help="whether to enable interactive mode")
    args = parser.parse_args()

    raw_data = mat73.loadmat(args.file_name)
    signal = raw_data["Signal"]

    print(f"Complete loading {len(signal)} markers")
    markers = BioMarkers(signal)

    while args.interactive:
        print(
            "[1] Blood Pressure [2] ECG [3] EEG [4] EGG [5] EMG [6] EOG [7] GSR [8] Respitory [9] TREV [10] Behavior"
        )
        num = int(input("Enter the number of the marker to inspect:"))
        markers.print_marker(ALL_MARKERS[num - 1])


if __name__ == "__main__":
    main()
