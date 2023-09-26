import numpy as np
from features.constants import EEG_BANDS, Feature


def calc_bands_power(x, dt, bands):
    from scipy.signal import welch

    f, psd = welch(x, fs=1.0 / dt)
    power = {
        band: np.mean(psd[np.where((f >= lf) & (f <= hf))])
        for band, (lf, hf) in bands.items()
    }
    return power


def avg_welch_bandpower(freqs, psd, band, relative=False):
    from scipy.integrate import simps

    band = np.asarray(band)
    low, high = band

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


def welch_bandpower(data, sf, band, window_sec=None):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    f: ndarray
        Array of sample frequencies.
    Pxx: ndarray
        Power spectral density or power spectrum of x.        
    """

    from scipy.signal import welch

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        band = np.asarray(band)
        low, _ = band
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    return welch(data, sf, nperseg=nperseg)

def get_psd(trial_data, srate, band):
    low, high = band
    freqs, psd = welch_bandpower(trial_data, srate, None, 2)

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    return psd[idx_band]


def get_psd_by_channel(block_data, marker, channel_type: str, feature: Feature):
    psd_data = []
    time_series_data = block_data.get_all_data()[marker]

    # loop through all trials: time -> frequency
    for t in range(time_series_data.shape[2]):
        all_channel_psd = []
        for i, c in enumerate(block_data.get_chanlocs(marker)):
            if not c.startswith(channel_type):
                continue

            data = time_series_data[i]
            psd = get_psd(data[:, t], block_data.get_srate(marker), EEG_BANDS[feature])
            all_channel_psd = (
                np.hstack((all_channel_psd, psd)) if len(all_channel_psd) > 0 else psd
            )

        psd_data = (
            np.vstack((psd_data, all_channel_psd))
            if len(psd_data) > 0
            else all_channel_psd
        )

    return psd_data