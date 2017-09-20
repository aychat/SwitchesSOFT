import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

omega10 = 2.354             # fs-1
omega20 = 4.708+2e-6        # fs-1
gamma10 = 1e-6              # fs-1
gamma20 = 1.5e-6            # fs-1
omegaM1 = 2.354 + 7e-8      # fs-1
omegaM2 = 2.354 + 1e-8      # fs-1
gamma = 1e-9                # fs-1
omega = np.linspace(2.354-2e-6, 2.354+2e-6, 8192)

omega_del1 = 9e-8
omega_del2 = 9e-8

N = 20

field1 = np.zeros([omega.size])
field2 = np.zeros([omega.size])

for i in range(-N, N):
    field1 += gamma / ((omega - omegaM1 - i*omega_del1)**2 + gamma**2)
    field2 += gamma / ((omega - omegaM2 - i*omega_del2)**2 + gamma**2)

pol2 = np.zeros([omega.size], dtype=np.complex)

B = omega20 - omega - 1j*gamma20

for i in range(-N, N):
    for j in range(-N, N):
        A1 = omega - omegaM2 - j*omega_del2 - omega10 + 1j*(gamma + gamma10)
        A2 = omegaM1 + i*omega_del2 - omega10 + 1j*(gamma + gamma10)
        K1 = omega - omegaM1 - omegaM2 - i*omega_del2 - j*omega_del2 + 2j*gamma10
        K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - i*omega_del2 - j*omega_del2)**2 + 4.*gamma**2)
        pol2 += K1*K2/(A1*A2*B)

print pol2
mat_resp = (1./(omega - omega10 + 1j*gamma10)).real
print mat_resp

plt.figure()
plt.plot(omega, field1, 'r')
plt.plot(omega, field2, 'b')
plt.plot(omega, 100*mat_resp, 'k')
plt.plot(omega, pol2.real, 'k')
plt.show()