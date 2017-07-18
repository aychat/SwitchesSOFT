import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType
from eigenvectors import MUBQHamiltonian
from FCfactors import FCfactors
from scipy.integrate import simps


class Spectra(FCfactors):
    """
    Calculate Franck Condon factors for 1D system in the coordinate representation.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
        """
        MUBQHamiltonian.__init__(self, **kwargs)
        FCfactors.__init__(self, **kwargs)
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)


if __name__ == '__main__':

    import time
    start = time.time()
    import matplotlib.pyplot as plt # Plotting facility

    print(FCfactors.__doc__)
    np.set_printoptions(precision=6)

    omega = .075
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    potential_func = FCfactors(
                        num_levels=4,
                        X_gridDIM=512,
                        X_amplitude=30.,
                        omega=omega,
                        displacement=3.75,
                        Vground="0.5 * (omega * X) ** 2",
                        Vexcited="0.5 * (omega * (X-displacement)) ** 2",
                        # Vexcited="5 * omega * (1 - exp(-sqrt(omega/(2*5))*(X - displacement)))**2",
                        K="0.5 * P ** 2",
                    )

    # plot eigenfunctions
    for n in range(potential_func.num_levels):
        plt.plot(potential_func.X, 0.075*potential_func.get_eigenstate_ground(n).real
                 + potential_func.get_energy_ground(n).real)

        plt.plot(potential_func.X, 0.075*potential_func.get_eigenstate_excited(n).real
                 + potential_func.get_energy_excited(n).real + 15. * potential_func.omega)

    steps = 96
    plt.plot(potential_func.X[256 - steps:256 + steps], potential_func.potential_ground[256 - steps:256 + steps])
    plt.plot(potential_func.X, potential_func.potential_excited + 15. * potential_func.omega)

    print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
    print(potential_func.energies_ground[:potential_func.num_levels])
    print("\n\nFirst energies for morse oscillator with omega = %f" % omega)
    print(potential_func.energies_excited[:potential_func.num_levels])
    E0 = potential_func.energies_excited[0]
    E1 = potential_func.energies_excited[1]
    E2 = potential_func.energies_excited[2]
    E3 = potential_func.energies_excited[3]
    print E0, E1, E2, E3

    with open("/home/ayanc/PycharmProjects/Switches/Cph8_discrete_levels/parameters.txt", "w") as f:
        potential_func.energies_excited[:potential_func.num_levels].real.tofile(f, sep=" ", format="%2.6lf")

    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 25 * potential_func.omega)

    with open("/home/ayanc/PycharmProjects/Switches/Cph8_discrete_levels/A_matrix.txt", "w") as f:
        for k in range(1, 2 * potential_func.num_levels)[::-1]:
            print "Matrix A" + str(k) + str(k - 1)
            print potential_func.A_matrix(k)
            print "\n"

            potential_func.A_matrix(k).real.tofile(f, sep=" ", format="%2.6lf")
            f.write(" ")
            potential_func.A_matrix(k).imag.tofile(f, sep=" ", format="%2.6lf")

    with open("/home/ayanc/PycharmProjects/Switches/Cph8_discrete_levels/mu_matrix.txt", "w") as f_mu:
        mu = potential_func.Dipole_matrix()
        mu.real.tofile(f_mu, sep=" ", format="%2.6lf")
        f_mu.write(" ")
        mu.imag.tofile(f_mu, sep=" ", format="%2.6lf")
        print "Matrix mu"
        print mu
        print "\n"

    import os
    #
    # os.system("gcc -O3 $(gsl-config --cflags) Cph8_absorption_spectra.c $(gsl-config --libs)")
    # os.system("./a.out " + str(omega) + " " + str(E0) + " " + str(E1) + " " + str(E2) + " " + str(E3))

    data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
    data_field = np.asarray(np.loadtxt("field_ini.txt"))

    plt.figure()
    plt.plot(data_field, 'r')
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

    data = np.loadtxt("pop_dynamical.out")
    freq1 = data[:, 0]
    print freq1.size
    PR = data[:, 1]

    PR /= PR.max()

    plt.figure()
    plt.title('Cph1 absorption spectra')
    plt.plot(lamb, Pr_abs, 'r', label='PR_expt')

    plt.title('Absorption spectra from model')
    plt.plot(660./freq1, PR, 'k-.', label='PR_model')
    plt.legend()
    plt.xlim(500., 750.)
    plt.grid()
    plt.show()

    print time.time() - start
