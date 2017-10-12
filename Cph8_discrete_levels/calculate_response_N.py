import numpy as np
import matplotlib.pyplot as plt

N = 4
data = np.loadtxt("molecules.dat", delimiter=' ')

assert N <= data.shape[0], "Not enough data for given N"

for i in range(N):
    globals()["omega10_M" + str(i + 1)] = 2.354 + data[i, 0]*1e-6
    globals()["omega20_M" + str(i + 1)] = 4.708 + data[i, 1]*1e-6
    globals()["gamma10_M" + str(i + 1)] = data[i, 2]*1e-6
    globals()["gamma20_M" + str(i + 1)] = data[i, 3]*1e-6


omegaM1 = 2.354 + 7e-8
omegaM2 = 2.354 + 1e-8
omegaM3 = 2.354 + 4e-8
gamma = 1e-9

Nfreq = 10
freq = np.linspace(2.*2.354-4e-6, 2.*2.354+4e-6, Nfreq)

omega_del1 = 9e-8
omega_del2 = 9e-8

N_comb = 20
del_omega1 = omega_del1 * np.arange(-N_comb, N_comb)
del_omega2 = omega_del2 * np.arange(-N_comb, N_comb)

omega = freq[:, np.newaxis, np.newaxis]
comb_omega1 = del_omega1[np.newaxis, :, np.newaxis]
comb_omega2 = del_omega2[np.newaxis, np.newaxis, :]

field1 = gamma / ((omega - 2*omegaM1 - 2*comb_omega1)**2 + 4*gamma**2)
field2 = gamma / ((omega - 2*omegaM2 - 2*comb_omega2)**2 + 4*gamma**2)

########################################################################################################################
#                                                                                                                      #
#         ----------------- SECOND ORDER POLARIZATION FUNCTIONS FOR THE OFF-DIAGONAL TERM --------------------         #
#                                                                                                                      #
########################################################################################################################


def calculate_pol_a1_12(w_in, g_in, w_out, g_out):
    gamma_net = 2 * gamma + g_in
    A1 = omega - omegaM2 - comb_omega2 - w_in + 1j*(gamma + g_in)
    A2 = omegaM1 + comb_omega1 - w_in + 1j*(gamma + g_in)
    K1 = (omega + omegaM1 - omegaM2 + comb_omega1 - comb_omega2 - 2*w_in) + 2j*gamma_net
    K2 = 2*np.pi*gamma/((omega - omegaM1 - omegaM2 - comb_omega1 - comb_omega2)**2 + 4.*gamma**2)
    B = omega - w_out + 1j*g_out

    return (K1*K2/(A1*A2*B)).sum(axis=(1, 2))

########################################################################################################################
#                                                                                                                      #
#                ----------------- PLOT SECOND ORDER POLARIZATION FOR N MOLECULES --------------------                 #
#                                                                                                                      #
########################################################################################################################

plt.figure()
plt.suptitle("$P^{(2)}(\\omega)$ for N molecules")
for i in range(N):
    plt.subplot(2, 4, i+1)
    globals()["Pol_M" + str(i + 1)] = calculate_pol_a1_12(
            globals()["omega10_M" + str(i + 1)],
            globals()["gamma10_M" + str(i + 1)],
            globals()["omega20_M" + str(i + 1)],
            globals()["gamma20_M" + str(i + 1)]
    ).real
    plt.plot(omega.reshape(-1)*1e6, globals()["Pol_M" + str(i + 1)].real, label='Molecule %d' % (i+1))

    plt.legend()
    plt.grid()
    plt.xlabel("freq (in GHz)")
    plt.ylabel("Polarization (arbitrary units)")

pol_matrix = globals()["Pol_M" + str(1)]

for i in range(1, N):
    pol_matrix = np.column_stack((pol_matrix, globals()["Pol_M" + str(i + 1)]))

pol_QR_matrix = np.zeros((N, Nfreq, N))
Qmat = np.zeros_like(pol_QR_matrix)
Rmat = np.zeros((N, N, N))

for i in range(N):
    pol_QR_matrix[i] = pol_matrix
    pol_QR_matrix[i][:, [0, i]] = pol_QR_matrix[i][:, [i, 0]]
    Qmat[i] = np.linalg.qr(pol_QR_matrix[i])[0]
    Rmat[i] = np.linalg.qr(pol_QR_matrix[i])[1]

    for j in range(N):
        print globals()["Pol_M" + str(i + 1)].dot(Qmat[i][:, j])
    print

for i in range(N):
    plt.subplot(2, 4, i+N+1)

    plt.plot(omega.reshape(-1)*1e6, globals()["Pol_M" + str(i + 1)].real, label='Molecule %d' % (i+1))

    plt.legend()
    plt.grid()
    plt.xlabel("freq (in GHz)")
    plt.ylabel("Polarization (arbitrary units)")
plt.show()