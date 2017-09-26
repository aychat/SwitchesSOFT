import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

omega10 = 2.354+1.e-6       # fs-1
omega20 = 4.708+2.e-6       # fs-1
gamma10 = 0.33e-6           # fs-1
gamma20 = 1.5e-6            # fs-1

omega21 = 2.354+1.e-6      # fs-1
gamma21 = 1.e-6            # fs-1

omegaM1 = 2.354 + 7e-8      # fs-1
omegaM2 = 2.354 + 1e-8      # fs-1
omegaM3 = 2.354 + 4e-8      # fs-1
gamma = 1e-9                # fs-1

freq = np.linspace(2.*2.354-9e-6, 2.*2.354+9e-6, 8192)
freq_n = np.linspace(-2.*2.354-4e-5, -2.*2.354+4e-6, 8192)

omega_del1 = 9e-8
omega_del2 = 9e-8

N=100
del_omega1 = omega_del1 * np.asarray(range(-N, N))
del_omega2 = omega_del2 * np.asarray(range(-N, N))

omega = freq[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = gamma / ((omega - 2*omegaM1 - 2*comb_omega1)**2 + 4*gamma**2)
field2 = gamma / ((omega - 2*omegaM2 - 2*comb_omega2)**2 + 4*gamma**2)
field_het = gamma / ((omega - 2*omegaM3 - 2*comb_omega2)**2 + 4*gamma**2)


def calculate_pol_term1(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def calculate_pol_term2(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM1 - comb_omega2 - w_in - 1j*(gamma + g_in)
    A2 = -omegaM2 - comb_omega2 + w_in + 1j*(gamma + g_in)
    K1 = (-omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 + 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))

plt.figure()
plt.suptitle("$2^{nd}$ Order rNL Polarization for the case (l=0, m=1, n=2)")
plt.subplot(211)
plt.title("$P^{(2)}_{(a_1)}$")
plt.plot(omega.reshape(-1)*1e6, calculate_pol_term1(omega10, gamma10, omega20, gamma20).real, 'r', label='$P^{(2)}_{(a_1)}$')
plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.plot(omega.reshape(-1)*1e6, 1e14*field_het.sum(axis=(1, 2)), 'r', label='E_field 2')
plt.xlabel("freq (in GHz)")
plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")

plt.subplot(212)
plt.title("$P^{(2)}_{(b_1)}$")
plt.plot(omega.reshape(-1)*1e6, calculate_pol_term2(omega10, gamma10, omega20, gamma20).real, 'r', label='$P^{(2)}_{(b_1)}$')
plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
plt.xlabel("freq (in GHz)")
plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")

plt.show()

