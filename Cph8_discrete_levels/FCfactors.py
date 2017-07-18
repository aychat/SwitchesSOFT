import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType
from eigenvectors import MUBQHamiltonian
from scipy.integrate import simps


class FCfactors(MUBQHamiltonian):
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
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        # Check that all attributes were specified
        try:
            # make sure self.X_amplitude has a value of power of 2
            assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, \
                "A value of the grid size (X_gridDIM) must be a power of 2"
        except AttributeError:
            raise AttributeError("Grid size (X_gridDIM) was not specified")

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate range (X_amplitude) was not specified")

        try:
            self.Vground
            self.potential_ground = ne.evaluate(self.Vground, local_dict=self.__dict__)
        except AttributeError:
            raise AttributeError("Ground otential energy (Vground) was not specified")

        try:
            self.Vexcited
            self.potential_excited = ne.evaluate(self.Vexcited, local_dict=self.__dict__)
        except AttributeError:
            raise AttributeError("Ground otential energy (Vground) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.num_levels
        except AttributeError:
            raise AttributeError("Number of vibrational levels (num_levels) was not specified")

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        k = np.arange(self.X_gridDIM)
        self.X = (k - self.X_gridDIM / 2) * self.dX
        # The same as
        # self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P = (k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        self.get_eigenstate_excited(self.num_levels)
        self.get_eigenstate_ground(self.num_levels)

    def FC_overlap(self, m, n):

        if m <= n:
            data = np.conj(self.eigenstates_ground[m]) * self.eigenstates_excited[n]
        else:
            data = np.zeros_like(self.X)
        result = simps(data.real, self.X) + simps(data.imag, self.X)*1j
        return result

    def A_matrix(self, k):

        F_mk = np.zeros(2*self.num_levels, dtype=np.complex)
        F_jn = np.zeros(2*self.num_levels, dtype=np.complex)

        for m in range(2 * self.num_levels):
            if m in range(self.num_levels) and k in range(self.num_levels, 2 * self.num_levels):
                F_mk[m] = (self.FC_overlap(m, k - self.num_levels))
            elif m == k:
                F_mk[m] = 1. + 0j
            else:
                F_mk[m] = 0. + 0j

        for n in range(2 * self.num_levels):
            if k - 1 in range(self.num_levels) and n in range(self.num_levels, 2 * self.num_levels):
                F_jn[n] = (self.FC_overlap(k - 1, n - self.num_levels))
            elif n == k - 1:
                F_jn[n] = 1. + 0j
            else:
                F_jn[n] = 0. + 0j
        F = np.outer(np.asarray(F_mk), np.asarray(F_jn))

        return F

    def Dipole_matrix(self):
        mu = np.zeros([2 * self.num_levels, 2 * self.num_levels], dtype=np.complex)

        for i in range(self.num_levels):
            for j in range(self.num_levels, 2 * self.num_levels):
                array = -np.conj(self.eigenstates_ground[i]) * self.X * self.eigenstates_excited[j - self.num_levels]
                mu[i][j] = simps(array.real, self.X) + simps(array.imag, self.X)*1j
                mu[j][i] = np.conj(mu[i][j])
        return mu

if __name__ == '__main__':

    import matplotlib.pyplot as plt # Plotting facility

    print(FCfactors.__doc__)
    np.set_printoptions(precision=6)

    omega = .1
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    potential_func = FCfactors(
                        num_levels=4,
                        X_gridDIM=512,
                        X_amplitude=30.,
                        omega=omega,
                        displacement=2.85,
                        Vground="0.5 * (omega * X) ** 2",
                        Vexcited="num_levels * omega * (1 - exp(-sqrt(omega/(2*num_levels))*(X - displacement)))**2",
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
    print(potential_func.energies_excited[:potential_func.num_levels] + 15. * potential_func.omega)

    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 25 * potential_func.omega)
    plt.show()

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
