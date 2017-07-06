import numpy as np
import matplotlib.pyplot as plt
import pickle

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
# mixed_abs /= mixed_abs.max()

mixed_ems = Pr_ems + Pfr_ems
# mixed_ems /= mixed_ems.max()

freq = 2. * np.pi * 3e2 / lamb
Pr_abs_freq = Pr_abs * 2. * np.pi * 3e2 / (freq * freq)
Pfr_abs_freq = Pfr_abs * 2. * np.pi * 3e2 / (freq * freq)

freq = freq[::-1]
Pr_abs_freq = Pr_abs_freq[::-1]
Pfr_abs_freq = Pfr_abs_freq[::-1]
plt.figure()
plt.plot(freq, Pr_abs_freq, 'r')
plt.plot(freq, Pfr_abs_freq, 'k')
# plt.plot(lamb, mixed_abs, 'b')
plt.xlim(2.0, 4.0)

# plt.figure()
# plt.plot(lamb, Pr_ems, 'r')
# plt.plot(lamb, Pfr_ems, 'k')
# plt.plot(lamb, mixed_ems, 'b')
# plt.xlim(650., 950.)
plt.show()