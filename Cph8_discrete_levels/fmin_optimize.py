import numpy as np
import numexpr as ne
from types import MethodType, FunctionType
from scipy.optimize import fmin
from FCfactors import FCfactors


class Propagator(FCfactors):
    """
    Calculate Franck Condon factors for 1D molecule1 in the coordinate representation.
    """

    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - specifying the grid size
            X_amplitude - maximum value of the coordinates
            V - potential energy (as a string to be evaluated by numexpr)
            K - momentum dependent part of the hamiltonian (as a string to be evaluated by numexpr)
        """
        FCfactors.__init__(self, **kwargs)
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.mu = self.Dipole_matrix()
        self.level = np.empty(2)
        self.level = (0.0, 1.87855 / 27.211385)

        self.discrete_Ham0 = np.diag(self.level)
        self.rho_init = np.zeros_like(self.discrete_Ham0, dtype=np.complex)
        self.rho_init[0][0] = 1.0 + 0.0j
        self.rho = self.rho_init.copy()
        self.t = np.linspace(0.0, self.dt * self.tsteps, self.tsteps)
        self.t0 = ne.evaluate(self.t0_code, local_dict=self.__dict__)

        # self.lindblad_matrix = np.zeros([2*self.num_levels, 2*self.num_levels], dtype=np.complex)
        self.Adagger_a = np.zeros([2 * self.num_levels - 1, 2 * self.num_levels, 2 * self.num_levels], dtype=np.complex)
        self.Amatrix = np.zeros([2 * self.num_levels - 1, 2 * self.num_levels, 2 * self.num_levels], dtype=np.complex)

        for i in range(2 * self.num_levels - 1):
            self.Adagger_a[i][i + 1][i + 1] = 1. + 0.j
            self.Amatrix[i][i][i + 1] = 1. + 0.j

        self.evolution = np.empty([self.tsteps, self.num_levels * 2], dtype=np.complex)

    def spectra_field(self, w):
        return 2. * self.field_amp * np.sin(w * self.t)  # *np.exp(-(self.t-self.t0)**2 / (self.sigma2))

    def Lindblad_dt(self, field_t, mat):
        self.discrete_Ham_tot = self.discrete_Ham0 - self.mu * field_t
        return -1j * (self.discrete_Ham_tot.dot(mat) - mat.dot(self.discrete_Ham_tot)) \
               + self.relaxation(mat) - self.gamma_c * (mat - np.diag(np.diag(mat))) \


    def relaxation(self, mat):
        return self.gamma * np.asarray([[mat[1, 1], -0.5 * mat[0, 1]], [-0.5 * mat[1, 0], -mat[1, 1]]])

    def propagate_dt(self, field_t):
        rho_copy = self.rho.copy()
        k = 1
        while np.linalg.norm(rho_copy) > 1e-10:
            rho_copy = self.Lindblad_dt(field_t, rho_copy)
            rho_copy *= self.dt / k
            k += 1
            self.rho += rho_copy

    def propagate_rho(self, w):
        field_w = self.spectra_field(w)
        self.rho = self.rho_init.copy()
        for i in range(self.tsteps):
            self.propagate_dt(field_w[i])
            self.evolution[i] = np.diag(self.rho)


if __name__ == '__main__':

    import time

    start = time.time()
    import matplotlib.pyplot as plt  # Plotting facility

    np.set_printoptions(precision=8, suppress=True)

    omega = .2 / 27.211385
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2

    np.random.seed(1783)

    molecule1 = Propagator(
        num_levels=1,
        field_amp=0.0005,  # np.random.uniform(0.001, 0.01),
        tsteps=5000,
        dt=1.,
        t0_code="dt*tsteps / 2.",
        omega=omega,
        gamma=1. / (50. * 41.3413789),  # 50 fs
        gamma_dephasing=1. / (50. * 41.3413789),  # 50 fs
        gamma_c=1. / (100. * 41.3413789),  # 100 fs
        X_gridDIM=256,
        X_amplitude=120.,
        displacement=2.85,
        Vground="0.5 * (omega * X) ** 2",
        Vexcited="8 * omega * (1 - exp(-sqrt(omega/(2*8))*(X - displacement)))**2",
        K="0.5 * P ** 2",
    )

    Nf = 60
    freq = np.empty(Nf)
    spectra1 = np.empty(Nf)
    for i in range(Nf):
        freq[i] = (.725 + i * .01333) * 1.87855 / 27.211385
    lamb_axis = (660. * 1.87855 / 27.211) / freq

    def calculate_spectra(gamma, gamma_c):
        molecule1.gamma = 1. / (gamma * 41.3413789)
        molecule1.gamma_c = 1. / (gamma_c * 41.3413789)

        print molecule1.gamma
        print molecule1.gamma_c

        for i in range(Nf):
            molecule1.propagate_rho(freq[i])
            pop1 = np.diag(molecule1.rho)
            spectra1[i] = pop1[molecule1.num_levels:].real.sum()
            print i, freq[i], spectra1[i]

        return spectra1

    base_spectra = calculate_spectra(50, 10)
    print base_spectra
    new_spectra = calculate_spectra(25, 100)
    print new_spectra

    plt.figure()
    plt.suptitle('$\\tau_c = %s \\quad \\quad \\tau_{vib} = %s$' % (
    1. / (molecule1.gamma_c * 41.3413789), 1. / (molecule1.gamma * 41.3413789)))
    plt.plot(lamb_axis, base_spectra, 'r', label='50_10')
    plt.plot(lamb_axis, new_spectra, 'k', label='25_100')
    plt.grid()

    plt.show()

    end = time.time()
    print end - start