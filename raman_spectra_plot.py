import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle
import os


with open("spectra.pickle", "rb") as f:
    data = pickle.load(f)

freq = np.asarray(data["delta_freq1"])
spectra = np.asarray(data["spectra1"])

f = interpolate.interp1d(freq, spectra, kind='cubic')
freq_new = np.linspace(515., 775., 500)
spectra_new = f(freq_new)
plt.plot(freq_new, spectra_new, 'r')
plt.show()