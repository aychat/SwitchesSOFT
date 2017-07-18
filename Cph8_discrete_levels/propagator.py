import numpy as np
import numexpr as ne
from scipy import fftpack # Tools for fourier transform
from scipy import linalg # Linear algebra for dense matrix
from types import MethodType, FunctionType
from eigenvectors import MUBQHamiltonian
from FCfactors import FCfactors
from scipy.integrate import simps


class Propagator(FCfactors, MUBQHamiltonian):
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
        FCfactors.__init__(self, **kwargs)
        MUBQHamiltonian.__init__(self, **kwargs)
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.A = np.empty([7, 2 * self.num_levels, 2 * self.num_levels], dtype=np.complex)

        for k in range(1, 2 * self.num_levels)[::-1]:
            self.A[k-1] = self.A_matrix(k)

        self.mu = self.Dipole_matrix()
        self.level = np.empty([2*self.num_levels])
        self.level[:self.num_levels] = self.energies_ground[:self.num_levels]
        self.level[self.num_levels:2*self.num_levels] = self.energies_excited[:self.num_levels] + self.omega0

        self.discrete_Ham0 = np.diag(self.level)
        self.rho_init = np.zeros_like(self.discrete_Ham0, dtype=np.complex)
        self.rho_init[0][0] = 1.0+0.0j
        self.rho = self.rho_init.copy()
        self.t = np.linspace(0.0, self.dt*self.tsteps, self.tsteps)
        self.t0 = ne.evaluate(self.t0_code, local_dict=self.__dict__)

    def spectra_field(self, w):
        return 0.001*np.cos(w*self.t)*np.exp(-(self.t-self.t0)**2 / self.sigma2)

    def Lindblad_dt(self, field_t, matrix):
        self.discrete_Ham_tot = self.discrete_Ham0 - self.mu * field_t
        return -1j*(self.discrete_Ham_tot.dot(matrix) - matrix.dot(self.discrete_Ham_tot))

    def propagate_dt(self, field_t):
        rho_copy = self.rho.copy()
        k = 1
        while rho_copy.max() > 1e-6:
            rho_copy = self.Lindblad_dt(field_t, rho_copy)
            rho_copy *= self.dt/k
            k += 1
            self.rho += rho_copy

    def propagate_rho(self, w):
        field_w = self.spectra_field(w)
        self.rho = self.rho_init.copy()
        for i in range(self.tsteps):
            self.propagate_dt(field_w[i])

if __name__ == '__main__':

    import time
    start = time.time()
    import matplotlib.pyplot as plt # Plotting facility

    print(FCfactors.__doc__)
    np.set_printoptions(precision=8, suppress=True)

    omega = .05
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    system = Propagator(
                        num_levels=4,
                        X_gridDIM=256,
                        X_amplitude=30.,
                        tsteps=50000,
                        dt=0.1,
                        t0_code="dt*tsteps / 2.",
                        sigma2=2*420.**2,
                        omega=omega,
                        omega0=1.,
                        displacement=4.5,
                        # Vground="8 * omega * (1 - exp(-sqrt(omega/(2*8))*X))**2",
                        Vground="0.5 * (omega * X) ** 2",
                        # Vexcited="0.5 * (omega * (X-displacement)) ** 2",
                        Vexcited="8 * omega * (1 - exp(-sqrt(omega/(2*8))*(X - displacement)))**2",
                        K="0.5 * P ** 2"
    )

    # plot eigenfunctions
    for n in range(system.num_levels):
        plt.plot(system.X, 0.05*system.get_eigenstate_ground(n).real
                 + system.get_energy_ground(n).real)

        plt.plot(system.X, 0.05*system.get_eigenstate_excited(n).real
                 + system.get_energy_excited(n).real + 15. * system.omega)

    steps = 256
    plt.plot(system.X[256 - steps:256 + steps], system.potential_ground[256 - steps:256 + steps])
    plt.plot(system.X, system.potential_excited + 15. * system.omega)

    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 25 * system.omega)

    plt.figure()
    plt.plot(system.t, system.spectra_field(w=.85))
    plt.show()

    Nf = 200
    freq = np.empty(Nf)
    spectra = np.empty(Nf)
    for i in range(Nf):
        freq[i] = .85 + i*.0025
        system.propagate_rho(freq[i])
        pop = np.diag(system.rho)
        spectra[i] = pop[system.num_levels:].real.sum()
        print i, 660./freq[i], spectra[i]

    data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')

    lamb = np.array(data[:, 0])
    Pr_abs = np.array(data[:, 1])

    Pr_abs /= Pr_abs.max()
    spectra /= spectra.max()
    plt.figure()
    plt.title('Cph1 absorption spectra')
    plt.plot(lamb, Pr_abs, 'r', label='PR_expt')
    plt.plot(660./freq, spectra, 'k', label='PR_model')
    plt.grid()
    plt.show()