import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import os


data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
lamb = np.array(data[:, 0])
Pr_abs = np.array(data[:, 1])
Pr_ems = np.array(data[:, 3])
Pfr_abs = np.array(data[:, 2])
Pfr_ems = np.array(data[:, 4])

Pr_abs /= Pr_abs.max()
Pfr_abs /= Pfr_abs.max()
Pr_ems /= Pr_ems.max()
Pfr_ems /= Pfr_ems.max()

mixed_abs = Pr_abs + Pfr_abs

mixed_ems = Pr_ems + Pfr_ems

freq = 2. * np.pi * 3e2 / lamb
Pr_abs_freq = Pr_abs * 2. * np.pi * 3e2 / (freq * freq)
Pfr_abs_freq = Pfr_abs * 2. * np.pi * 3e2 / (freq * freq)

freq = freq[::-1]
Pr_abs_freq = Pr_abs_freq[::-1]
Pfr_abs_freq = Pfr_abs_freq[::-1]


os.system("gcc -O3 -Wall $(gsl-config --cflags) opto_10_levels_Cph8_dephasing_work_done.c $(gsl-config --libs)")
os.system("./a.out 0.1 100.0 ")

data = np.loadtxt("pop_dynamical.out")

freq1 = data[:, 0] + 2.0
PR = data[:, 1]
PFR = data[:, 2]

# PR /= PR.max()
# PFR /= PFR.max()
Pr_abs_freq /= Pfr_abs_freq.max()
Pfr_abs_freq /= Pfr_abs_freq.max()

plt.figure()
plt.subplot(121)
plt.title('Cph1 absorption spectra')
plt.plot(freq, Pr_abs_freq, 'r', label='PR')
plt.plot(freq, Pfr_abs_freq, 'k', label='PFR')
plt.xlim(2.0, 4.0)
plt.ylim(0.0, 1.1)
plt.legend()
plt.subplot(122)
plt.title('Absorption spectra from model')
plt.plot(freq1, PR, 'r', label='PR')
plt.plot(freq1, PFR, 'k', label='PFR')
plt.legend()
plt.show()