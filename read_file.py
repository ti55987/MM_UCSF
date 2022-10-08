import mat73
import argparse

from analysis.biomarkers import BioMarkers, ALL_MARKERS


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
            "[1] Blood Pressure [2] ECG [3] EEG [4] EGG [5] EMG [6] EOG [7] GSR [8] Respitory [9] TREV [10] Behavior [11] Exit"
        )
        num = int(input("Enter the number of the marker to inspect:"))
        if num == 11:
            exit()

        markers.print_marker(ALL_MARKERS[num - 1])


if __name__ == "__main__":
    main()
