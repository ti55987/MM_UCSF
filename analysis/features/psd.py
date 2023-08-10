import numpy as np


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
