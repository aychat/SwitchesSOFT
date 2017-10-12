import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

omega10_M1 = 2.354+1.e-6       # fs-1
omega20_M1 = 4.708+2.e-6       # fs-1
gamma10_M1 = 0.33e-6           # fs-1
gamma20_M1 = 1.5e-6            # fs-1

omega10_M2 = 2.354+1.5e-6      # fs-1
omega20_M2 = 4.708+2.5e-6      # fs-1
gamma10_M2 = 0.66e-6           # fs-1
gamma20_M2 = 1.e-6             # fs-1

omega10_M3 = 2.354+2e-6        # fs-1
omega20_M3 = 4.708+2.25e-6     # fs-1
gamma10_M3 = 0.75e-6           # fs-1
gamma20_M3 = 2.e-6             # fs-1

omega21 = 2.354+1.e-6          # fs-1
gamma21 = 1.e-6                # fs-1

omegaM1 = 2.354 + 7e-8         # fs-1
omegaM2 = 2.354 + 1e-8         # fs-1
omegaM3 = 2.354 + 4e-8         # fs-1
gamma = 1e-9                   # fs-1

freq = np.linspace(2.*2.354-4e-6, 2.*2.354+4e-6, 10000)
freq_n = np.linspace(-2.*2.354-4e-5, -2.*2.354+4e-6, 4096)

omega_del1 = 9e-8
omega_del2 = 9e-8

N=20
del_omega1 = omega_del1 * np.arange(-N, N)
del_omega2 = omega_del2 * np.arange(-N, N)

omega = freq[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = gamma / ((omega - 2*omegaM1 - 2*comb_omega1)**2 + 4*gamma**2)
field2 = gamma / ((omega - 2*omegaM2 - 2*comb_omega2)**2 + 4*gamma**2)
field_het = gamma / ((omega - 2*omegaM3 - 2*comb_omega2)**2 + 4*gamma**2)

# -------------------------------------------------------------------------------------------------------------------- #
#                                               4 P(2)_(a1) terms:
# -------------------------------------------------------------------------------------------------------------------- #


def calculate_pol_a1_11(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM1 - comb_omega1 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM1 - comb_omega1 - comb_omega1)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def calculate_pol_a1_12(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def calculate_pol_a1_21(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM1 - comb_omega1 - w_in + 1j*(gamma + g_in)
    A2 = omegaM2 + comb_omega2 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM2 - omegaM1 + comb_omega2 - comb_omega1 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM2 - omegaM1 - comb_omega2 - comb_omega1)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def calculate_pol_a1_22(w_in, g_in, w_out, g_out):
    """
    
    :param w_in: 
    :param g_in: 
    :param w_out: 
    :param g_out: 
    :return: 
    """
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM2 + comb_omega2 - w_in + 1j*(gamma + g_in)
    K1 = (omega - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM2 - omegaM2 - comb_omega2 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def calculate_pol_b1_12(w_in, g_in, w_out, g_out):
    """
    
    :param w_in: 
    :param g_in: 
    :param w_out: 
    :param g_out: 
    :return: 
    """
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM1 - comb_omega2 - w_in - 1j*(gamma + g_in)
    A2 = -omegaM2 - comb_omega2 + w_in + 1j*(gamma + g_in)
    K1 = (-omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 + 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))


def normalize_real(data):
    """
    
    :param data: 
    :return: 
    """
    return data.real / np.linalg.norm(data.real, np.inf)

# plt.figure()
# plt.suptitle("$2^{nd}$ Order rNL Polarization for the case (l=0, m=1, n=2)")
# plt.subplot(221)
# plt.title("$P^{(2)}_{(a_1)} term (1, 1)$")
# plt.plot(
#     omega.reshape(-1)*1e6,
#     normalize_real(calculate_pol_a1_11(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1)),
#     'r', label='$P^{(2)}_{(a_1)}$'
# )
# plt.plot(
#     omega.reshape(-1)*1e6,
#     5e14*field1.sum(axis=(1, 2)),
#     'b', label='E_field 1'
# )
# plt.plot(
#     omega.reshape(-1)*1e6,
#     5e14*field2.sum(axis=(1, 2)),
#     'g', label='E_field 2'
# )
# # plt.plot(omega.reshape(-1)*1e6, 1e14*field_het.sum(axis=(1, 2)), 'r', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(222)
# plt.title("$P^{(2)}_{(a_1)} term (1, 2)$")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_12(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(223)
# plt.title("$P^{(2)}_{(a_1)}$ term (2, 1)")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_21(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(224)
# plt.title("$P^{(2)}_{(a_1)} term (2, 2)$")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_22(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
#
# plt.figure()
# plt.suptitle("$2^{nd}$ Order rNL Polarization for the case (l=0, m=1, n=2)")
# plt.subplot(221)
# plt.title("$P^{(2)}_{(a_1)} term (1, 1)$")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_11(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# # plt.plot(omega.reshape(-1)*1e6, 1e14*field_het.sum(axis=(1, 2)), 'r', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(222)
# plt.title("$P^{(2)}_{(a_1)} term (1, 2)$")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_12(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(223)
# plt.title("$P^{(2)}_{(a_1)}$ term (2, 1)")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_21(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")
#
# plt.subplot(224)
# plt.title("$P^{(2)}_{(a_1)} term (2, 2)$")
# plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_22(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real, 'r', label='$P^{(2)}_{(a_1)}$')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field1.sum(axis=(1, 2)), 'b', label='E_field 1')
# plt.plot(omega.reshape(-1)*1e6, 5e14*field2.sum(axis=(1, 2)), 'g', label='E_field 2')
# plt.xlabel("freq (in GHz)")
# plt.ylabel("Electric fields and polarization (in $fs^{-1}$)")

plt.figure()
plt.suptitle("$P^{(2)}(\\omega)$ for Molecule 1 and Molecule 2")
plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_12(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1).real, label='Molecule1')
plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_12(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real, label='Molecule2')
plt.plot(omega.reshape(-1)*1e6, calculate_pol_a1_12(omega10_M3, gamma10_M3, omega20_M3, gamma20_M3).real, label='Molecule3')
plt.legend()
plt.grid()
plt.xlabel("freq (in GHz)")
plt.ylabel("Polarization (arbitrary units)")
plt.show()

import pickle

with open('data_pol2.pickle', 'wb') as file_out:
    pickle.dump(
        {
            'omega': omega.reshape(-1)*1e6,
            'pol2_mol1': calculate_pol_a1_12(omega10_M1, gamma10_M1, omega20_M1, gamma20_M1).real,
            'pol2_mol2': calculate_pol_a1_12(omega10_M2, gamma10_M2, omega20_M2, gamma20_M2).real,
            'pol2_mol3': calculate_pol_a1_12(omega10_M3, gamma10_M3, omega20_M3, gamma20_M3).real
        },
        file_out
    )


