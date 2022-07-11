This is a simple python command line tool to read .mat file in version 7.3

## Prerequisite
- Python3 Installation

## Installation
```
$ pip install -r requirements.txt
```

## Usage
```
usage: read_file.py [-h] [--interactive INTERACTIVE] file_name

positional arguments:
  file_name             the path to the .mat file

optional arguments:
  -h, --help            show this help message and exit
  --interactive INTERACTIVE
                        whether to enable interactive mode
```

Inspect data:
```
$ python3 read_file.py 1004_audio_hvha_allSignals_cleaned_final.mat --interactive=true
```

Output:
```
$ Complete loading 10 markers
$ [1] Blood Pressure [2] ECG [3] EEG [4] EGG [5] EMG [6] EOG [7] GSR [8] Respitory [9] TREV [10] Behavior
$ Enter the number of the marker to inspect:9
$ Inspecting TREV data...
$ All Fields: ('data1', 'data1Units', 'data2', 'data2Units', 'srate', 'times', 'xmax', 'xmin')
$ Enter the field to inspect:data1
```
