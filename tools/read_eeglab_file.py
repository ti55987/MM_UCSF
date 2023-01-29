import mat73
import argparse
import scipy.io as sio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="the path to the .mat file")
    args = parser.parse_args()

    raw_data = sio.loadmat(args.file_name)
    print(len(raw_data["chanlocs"]))
    locations = []
    for data in raw_data["chanlocs"]:
        locations.append(data[0][0].strip())

    print(locations[:128])


if __name__ == "__main__":
    main()
