import os
from glob import glob

import numpy as np
import obspy
import pandas as pd
import scipy.signal as sp

# paths
sacdirnames = sorted(glob("data/irpinia/sac/*/"))

# parameters
cutoff = 25.0
corners = 4  # two for freqmin and two fro cutoff
taper = 0.1  # twice than obspy taper
paz = {
    "sensitivity": 2800,
    "zeros": [0j, 0j],
    "gain": 1,
    "poles": [-6.2832 - 4.7124j, -6.2832 + 4.7124j],
}


def align(st):
    starttime = max(tr.stats.starttime for tr in st)
    endtime = min(tr.stats.endtime for tr in st)
    st.trim(starttime, endtime)


def preprocess(st, integrate_twice=False):
    for tr in st:
        # unpack
        fs = tr.stats.sampling_rate
        data = tr.data
        # detrend
        data = sp.detrend(data, type="linear")
        # taper
        data *= sp.get_window(("tukey", 0.1), len(data), fftbins=False)
        # remove instrumental response
        data *= tr.stats.sac["user2"] / tr.stats.sac["user3"]
        # filter
        sos = sp.iirfilter(corners, cutoff, btype="low", output="sos", fs=fs)
        data = sp.sosfilt(sos, data)
        # integrate
        data = np.cumsum(data) / fs
        if integrate_twice:
            data = np.cumsum(data) / fs
        # simulate Wood-Anderson response
        b, a = sp.zpk2tf(paz["zeros"], paz["poles"], paz["gain"])
        b, a = sp.bilinear(b, a, fs=tr.stats["sampling_rate"])
        data = paz["sensitivity"] * sp.lfilter(b, a, data)
        # modify in place
        tr.data = data


def get_rms_peak(st):
    data = np.sqrt(np.mean([np.square(tr.data) for tr in st], axis=0))
    return np.max(data)


def get_hypocentral_distance(st):
    metadata = st[0].stats.sac
    epi = metadata["dist"]
    depth = metadata["stel"] / 1000 + metadata["evdp"]
    return np.hypot(epi, depth)


def estimate_magnitude(st):
    peak = get_rms_peak(st)
    A = 1e3 * peak
    R = get_hypocentral_distance(st)
    return np.log10(A) + 1.79 * np.log10(R) - 0.58


def smad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


catalog = []
for sacdirname in sacdirnames:
    event = os.path.split(os.path.split(sacdirname)[0])[-1]
    st_vel = obspy.read(os.path.join(sacdirname, "*.C0[4-5].*.sac"))
    st = st_vel
    align(st)
    preprocess(st_vel)
    stations = {tr.stats.station for tr in st}
    magnitudes = []
    for station in stations:
        st_sta = st.select(station=station)
        magnitudes.append(estimate_magnitude(st_sta))
    catalog.append(
        {"event": event, "magnitude": np.median(magnitudes), "smad": smad(magnitudes)}
    )
catalog = pd.DataFrame.from_records(catalog, index="event")
catalog.to_csv("results/info_catalog.csv")
