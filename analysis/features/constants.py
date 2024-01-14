from enum import Enum


class Feature(Enum):
    EEG_SPECTRAL = 28
    DELTA = 1
    THETA = 2
    ALPHA = 3
    BETA1 = 4
    BETA2 = 5
    GAMMA = 6
    ALL_SPECTRAL = 18
    ALL_CHANNEL_SPECTRAL = 19
    # statistical features
    MEAN = 7
    STD = 8
    PTP = 9
    VAR = 10
    MINIM = 11
    MAXIM = 12
    MEAN_SQUARE = 13
    RMS = 14
    ABS_DIFF = 15
    SKEWNESS = 16
    KURTOSIS = 17

    ECG_LF = 20
    ECG_HF = 21
    ECG_LFHF = 22

    EGG_FILTERED = 23
    EGG_PHASE = 24
    EGG_AMPLITUDE = 25
    TREV_VELOCITY = 26

    RAW_DATA = 27


AXIS = 0

EEG_BANDS = {
    Feature.DELTA: (1, 4),
    Feature.THETA: (4, 8),
    Feature.ALPHA: (8, 12),
    Feature.BETA1: (12, 20),
    Feature.BETA2: (20, 30),
    Feature.GAMMA: (30, 50),
}

MARKER_METADATA = [
    {"marker": "ECG", "chan": 1, "feature": "LF"},
    {"marker": "ECG", "chan": 2, "feature": "LF/HF"},
    {"marker": "TREV", "chan": 0, "feature": "Velocity"},
    {"marker": "EGG", "chan": 2, "feature": "Amplitude"},
]

MARKER_TO_FEATURE = {
    "ECG": [Feature.ECG_HF, Feature.ECG_LF, Feature.ECG_LFHF],
    "EGG": [Feature.EGG_FILTERED, Feature.EGG_PHASE, Feature.EGG_AMPLITUDE],
}
