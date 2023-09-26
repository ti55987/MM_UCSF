
from enum import Enum

class Feature(Enum):
    DELTA = 1
    THETA = 2
    ALPHA = 3
    BETA1 = 4
    BETA2 = 5
    GAMMA = 6
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


AXIS = 0

EEG_BANDS = {
    Feature.DELTA: (1, 4),
    Feature.THETA: (4, 8),
    Feature.ALPHA: (8, 12),
    Feature.BETA1: (12, 20),
    Feature.BETA2: (20, 30),
    Feature.GAMMA: (30, 50),
}