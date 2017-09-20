import numpy as np
import matplotlib.pyplot as plt
from subprocess import call
import os

omega = 0.0775
displacement = .5
os.system("python A_matrix.py " + str(omega) + " " + str(displacement))
os.system("gcc -O3 -Wall $(gsl-config --cflags) Cph8_absorption_spectra.c $(gsl-config --libs)")
os.system("./a.out " + str(omega))

data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
data_field = np.asarray(np.loadtxt("field_ini.txt"))

# plt.figure()
# plt.plot(data_field, 'r')
# plt.show()
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

# freq = 2. * np.pi * 3e2 / lamb
# Pr_abs_freq = Pr_abs * 2. * np.pi * 3e2 / (freq * freq)
# Pfr_abs_freq = Pfr_abs * 2. * np.pi * 3e2 / (freq * freq)

# freq = freq[::-1]
# Pr_abs_freq = Pr_abs_freq[::-1]
# Pfr_abs_freq = Pfr_abs_freq[::-1]

data = np.loadtxt("pop_dynamical.out")
# print data.shape
freq1 = data[:, 0]
PR = data[:, 1]
# PFR = data[:, 2]

PR /= PR.max()

# PFR /= PFR.max()
# Pr_abs_freq /= Pr_abs_freq.max()
# Pfr_abs_freq /= Pfr_abs_freq.max()

plt.figure()
plt.title('Cph1 absorption spectra')
plt.plot(lamb, Pr_abs, 'r', label='PR_expt')

plt.title('Absorption spectra from model')
plt.plot(660./freq1, PR, 'k-.', label='PR_model')
# plt.plot(freq, Pr_abs_freq, 'r', label='PR')
plt.legend()
plt.show()
