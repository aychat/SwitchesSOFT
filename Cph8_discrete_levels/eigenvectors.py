import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType


class MUBQHamiltonian:
    """
    Generate quantum Hamiltonian, H(x,p) = K(p) + V(x),
    for 1D system in the coordinate representation using mutually unbiased bases (MUB).
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
        """

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
        except AttributeError:
            raise AttributeError("Ground potential energy (V) was not specified")

        try:
            self.Vexcited
        except AttributeError:
            raise AttributeError("Excited potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        # get coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # generate coordinate range
        k = np.arange(self.X_gridDIM)
        self.X = (k - self.X_gridDIM / 2) * self.dX
        # The same as
        # self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)

        # generate momentum range as it corresponds to FFT frequencies
        self.P = (k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)

        self.potential_ground = ne.evaluate(self.Vground, local_dict=self.__dict__)

        self.potential_excited = ne.evaluate(self.Vexcited, local_dict=self.__dict__)

        # 2D array of alternating signs
        minus = (-1) ** (k[:, np.newaxis] + k[np.newaxis, :])
        # see http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # for explanation of np.newaxis and other array indexing operations
        # also https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
        # for understanding the broadcasting in array operations

        # Construct the momentum dependent part
        self.Hamiltonian_ground = np.diag(
            ne.evaluate(self.K, local_dict=self.__dict__)
        )
        self.Hamiltonian_ground *= minus
        self.Hamiltonian_ground = fftpack.fft(self.Hamiltonian_ground, axis=1, overwrite_x=True)
        self.Hamiltonian_ground = fftpack.ifft(self.Hamiltonian_ground, axis=0, overwrite_x=True)
        self.Hamiltonian_ground *= minus

        self.Hamiltonian_excited = np.diag(
            ne.evaluate(self.K, local_dict=self.__dict__)
        )
        self.Hamiltonian_excited *= minus
        self.Hamiltonian_excited = fftpack.fft(self.Hamiltonian_excited, axis=1, overwrite_x=True)
        self.Hamiltonian_excited = fftpack.ifft(self.Hamiltonian_excited, axis=0, overwrite_x=True)
        self.Hamiltonian_excited *= minus

        # Add diagonal potential energy
        self.Hamiltonian_ground += np.diag(
            ne.evaluate(self.Vground, local_dict=self.__dict__)
        )

        self.Hamiltonian_excited += np.diag(
            ne.evaluate(self.Vexcited, local_dict=self.__dict__)
        )

    def get_eigenstate_ground(self, n):
        """
        Return n-th eigenfunction
        :param n: order
        :return: a copy of numpy array containing eigenfunction
        """
        self.diagonalize()
        return self.eigenstates_ground[n].copy()

    def get_eigenstate_excited(self, n):
        """
        Return n-th eigenfunction
        :param n: order
        :return: a copy of numpy array containing eigenfunction
        """
        self.diagonalize()
        return self.eigenstates_excited[n].copy()

    def get_energy_ground(self, n):
        """
        Return the energy of the n-th eigenfunction
        :param n: order
        :return: real value
        """
        self.diagonalize()
        return self.energies_ground[n]

    def get_energy_excited(self, n):
        """
        Return the energy of the n-th eigenfunction
        :param n: order
        :return: real value
        """
        self.diagonalize()
        return self.energies_excited[n]

    def diagonalize(self):
        """
        Diagonalize the Hamiltonian if necessary
        :return: self
        """
        # check whether the hamiltonian has been diagonalized
        try:
            self.eigenstates_ground
            self.energies_ground
            self.eigenstates_excited
            self.energies_excited
        except AttributeError:
            # eigenstates have not been calculated so
            # get real sorted energies and underlying wavefunctions
            # using specialized function for Hermitian matrices
            self.energies_ground, self.eigenstates_ground = linalg.eigh(self.Hamiltonian_ground)
            self.energies_excited, self.eigenstates_excited = linalg.eigh(self.Hamiltonian_excited)

            # extract real part of the energies
            self.energies_ground = np.real(self.energies_ground)
            self.energies_excited = np.real(self.energies_excited)

            # covert to the formal convenient for storage
            self.eigenstates_ground = self.eigenstates_ground.T
            self.eigenstates_excited = self.eigenstates_excited.T

            # normalize each eigenvector
            for psi in self.eigenstates_ground:
                psi /= linalg.norm(psi) * np.sqrt(self.dX)

            for phi in self.eigenstates_excited:
                phi /= linalg.norm(phi) * np.sqrt(self.dX)

            # check that the ground state is not negative
            if self.eigenstates_ground[0].real.sum() < 0:
                self.eigenstates_ground *= -1
            if self.eigenstates_excited[0].real.sum() < 0:
                self.eigenstates_excited *= -1

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':
    import matplotlib.pyplot as plt # Plotting facility

    print(MUBQHamiltonian.__doc__)

    omega = .15
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    potential_func = MUBQHamiltonian(
                        num_levels=4,
                        X_gridDIM=512,
                        X_amplitude=30.,
                        omega=omega,
                        displacement=1.85,
                        Vground="0.5 * (omega * X) ** 2",
                        Vexcited="16 * omega * (1 - exp(-sqrt(omega/(2*16))*(X - displacement)))**2",
                        # Vexcited="0.5 * (omega * (X-displacement)) ** 2 + 0.75 * (omega*(X ))**4",
                        K="0.5 * P ** 2",
                    )

    # plot eigenfunctions
    for n in range(4):
        plt.plot(potential_func.X, 0.25*potential_func.get_eigenstate_ground(n).real
                 + potential_func.get_energy_ground(n).real)

        plt.plot(potential_func.X, 0.25*potential_func.get_eigenstate_excited(n).real
                 + potential_func.get_energy_excited(n).real + 7.5*potential_func.omega)

    steps = 60
    plt.plot(potential_func.X[256-steps:256+steps], potential_func.potential_ground[256-steps:256+steps])
    plt.plot(potential_func.X, potential_func.potential_excited + 7.5*potential_func.omega)

    print("\n\nFirst energies for harmonic oscillator with omega = %f" % omega)
    print(potential_func.energies_ground[:20])
    print("\n\nFirst energies for morse oscillator with omega = %f" % omega)
    print(potential_func.energies_excited[:20])

    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 16*potential_func.omega)
    plt.legend()
    plt.show()