import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType
from eigenvectors import MUBQHamiltonian
from scipy.integrate import trapz


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

    def Dipole_matrix(self):
        mu = np.zeros([2 * self.num_levels, 2 * self.num_levels], dtype=np.complex)
        overlap_grd_exc = np.zeros([self.num_levels, self.num_levels], dtype=np.complex)
        overlap_exc_grd = np.zeros_like(overlap_grd_exc, dtype=np.complex)

        for i in range(self.num_levels):
            for j in range(self.num_levels):
                overlap_grd_exc[i][j] = sum(np.conj(self.eigenstates_ground[i]) * self.eigenstates_excited[j])*self.dX
                overlap_exc_grd[i][j] = sum(np.conj(self.eigenstates_excited[i]) * self.eigenstates_ground[j])*self.dX

        zero_mat = np.zeros_like(overlap_grd_exc, dtype=np.complex)

        return np.bmat('zero_mat, overlap_grd_exc; overlap_exc_grd, zero_mat')

if __name__ == '__main__':

    import matplotlib.pyplot as plt # Plotting facility

    print(FCfactors.__doc__)
    np.set_printoptions(precision=3, suppress=True)

    omega = .1
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    potential_func = FCfactors(
                        num_levels=1,
                        X_gridDIM=512,
                        X_amplitude=30.,
                        omega=omega,
                        displacement=2.85,
                        # Vground="0.5 * (omega * X) ** 2",
                        Vground="8 * omega * (1 - exp(-sqrt(omega/(2*8))*X))**2",
                        Vexcited="8 * omega * (1 - exp(-sqrt(omega/(2*8))*(X - displacement)))**2",
                        K="0.5 * P ** 2",
                    )

    font_r_12 = {'family': 'serif',
                 'color': 'darkred',
                 'weight': 'normal',
                 'size': 12,
                }

    font_k_12 = {'family': 'serif',
                 'color': 'black',
                 'weight': 'normal',
                 'size': 12,
                 }

    font_k_8 = {'family': 'serif',
                 'color': 'black',
                 'weight': 'normal',
                 'size': 8,
                 }
    # plot eigenfunctions
    for n in range(potential_func.num_levels):
        plt.plot(potential_func.X, 0.1*potential_func.get_eigenstate_ground(n).real
                 + potential_func.get_energy_ground(n).real)
        plt.text(-34., potential_func.get_energy_ground(n).real, r'$\psi_{g%d}$' % n, fontdict=font_k_8)

        plt.plot(potential_func.X, 0.1*potential_func.get_eigenstate_excited(n).real
                 + potential_func.get_energy_excited(n).real + 15. * potential_func.omega)
        plt.text(-34., potential_func.get_energy_ground(n).real + 15. * potential_func.omega,
                 r'$\psi_{e%d}$' % n, fontdict=font_k_8)

    steps = 256
    plt.plot(potential_func.X[256 - steps:256 + steps], potential_func.potential_ground[256 - steps:256 + steps], 'r')
    plt.plot(potential_func.X, potential_func.potential_excited + 15. * potential_func.omega, 'k')

    print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
    print(potential_func.energies_ground[:potential_func.num_levels])
    print("\n\nFirst energies for morse oscillator with omega = %f" % omega)
    print(potential_func.energies_excited[:potential_func.num_levels] + 15. * potential_func.omega)
    plt.text(31.2, 0.62, r'$S_0$', fontdict=font_r_12)
    plt.text(31.2, 2.10, r'$S_1$', fontdict=font_k_12)

    # plt.title("Eigenfunctions for ground and excited states with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.xlim(-35., 35.)
    plt.ylim(0.0, 25 * potential_func.omega)
    plt.show()

    with open("/home/ayanc/PycharmProjects/Switches/Cph8_discrete_levels/mu_matrix.txt", "w") as f_mu:
        mu = potential_func.Dipole_matrix()
        mu.real.tofile(f_mu, sep=" ", format="%2.6lf")
        f_mu.write(" ")
        mu.imag.tofile(f_mu, sep=" ", format="%2.6lf")
        print "Matrix mu"
        print mu
        print "\n"
