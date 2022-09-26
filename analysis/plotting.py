import pandas as pd


def plot_eeg_band(eeg_band_fft):
    print(f"eeg_band_fft {eeg_band_fft}")
    df = pd.DataFrame(columns=["band", "val"])
    df["band"] = eeg_band_fft.keys()
    df["val"] = [eeg_band_fft[band] for band in eeg_band_fft.keys()]
    ax = df.plot.bar(x="band", y="val", legend=False)
    ax.set_xlabel("EEG band")
    ax.set_ylabel("Mean band Amplitude")
    ax.yaxis.grid(True)
