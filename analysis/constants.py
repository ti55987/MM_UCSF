SUEJECT_BATCHES = [
    [2000, 2001, 2002, 2003, 2004, 2005],
    [2017, 2018, 2020, 2024, 2025, 2026],
    [2028, 2029, 2031, 2032, 2033, 2035], 
    [2036, 2039, 2040, 2041, 2042, 2043],
]

SORTED_BLOCK_NAMES = [
    "audio_hvha",
    "audio_hvla",
    "audio_nvha",
    "audio_nvla",
    "breath_hvha",
    "breath_hvla",
    "breath_nvha",
    "breath_nvla",
    "meditation",
    "mind wandering",
]

AUDIO_BLOCKS = ["audio_hvha", "audio_hvla", "audio_nvha", "audio_nvla"]

ALL_EEG_BANDS = ["DELTA", "THETA", "ALPHA", "BETA1", "BETA2", "GAMMA"]

COLOR_MAP = {
    "audio_hvha": "red",
    "audio_hvla": "magenta",
    "audio_nvha": "green",
    "audio_nvla": "olive",
}
V_COLOR_MAP = {
    "hvha": "red",
    "hvla": "magenta",
    "nvha": "green",
    "nvla": "olive",
    "lvha": "blue",
    "lvla": "steelblue",
}
