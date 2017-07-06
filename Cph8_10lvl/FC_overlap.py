import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.misc import factorial
from scipy.integrate import simps

x = np.linspace(-20., 20., 1024)
omega = np.linspace(0.3, 2.5, 512)


def factor(k, d, w):
    return (1./np.sqrt(2**k * factorial(k)))*((w/np.pi)**0.25)*np.exp(-0.5*w*(x-d)**2)*eval_hermite(k, np.sqrt(w)*x)


def overlap(m, n, d, w):
    data = factor(m, 0., w)*factor(n, d, w)
    return simps(data, x)

m_num, n_num = 4, 4
displacement = 6.45
gamma = 1/60.
gamma01 = 1/5.
omega_eg = 0.454
omega0 = 0.135
spectra = np.empty(omega.size)
for m in range(m_num):
    for n in range(m, n_num):
        print m, n

        spectra += ((overlap(m, n, displacement, omega0)**2) / (
            gamma - 1j*(omega - omega_eg + omega0*(m-n)))).real

plt. figure()
plt.plot((660.*.454)/omega, spectra/spectra.max())

# plt.figure()
# plt.plot(x, np.exp(-0.5*omega0*(x-displacement)**2)*eval_hermite(0, np.sqrt(omega0)*x))
# plt.plot(x, np.exp(-0.5*omega0*(x-displacement)**2)*eval_hermite(1, np.sqrt(omega0)*x))
# plt.plot(x, np.exp(-0.5*omega0*(x-displacement)**2)*eval_hermite(2, np.sqrt(omega0)*x))
# plt.plot(x, np.exp(-0.5*omega0*(x-displacement)**2)*eval_hermite(3, np.sqrt(omega0)*x))

plt.figure()
data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
plt.plot(data[:, 0], data[:, 1])
plt.plot(data[:, 0], data[:, 2])
plt.show()