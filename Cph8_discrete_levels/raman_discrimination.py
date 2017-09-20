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
        MUBQHamiltonian.__init__(self, **kwargs)
        # save all attributes
        for name, value in kwargs.items():
            # if the value supplied is a function, then dynamically assign it as a method;
            # otherwise bind it a property
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.mu = self.Dipole_matrix()
        self.level = np.empty([2*self.num_levels])
        self.level[:self.num_levels] = self.energies_ground[:self.num_levels]/1.87855
        self.level[self.num_levels:2*self.num_levels] = self.energies_excited[:self.num_levels]/1.87855 + self.omega0/1.87855

        self.discrete_Ham0 = np.diag(self.level)
        self.rho_init = np.zeros_like(self.discrete_Ham0, dtype=np.complex)
        self.rho_init[0][0] = 1.0+0.0j
        self.rho = self.rho_init.copy()
        self.t = np.linspace(0.0, self.dt*self.tsteps, self.tsteps)
        self.t0 = ne.evaluate(self.t0_code, local_dict=self.__dict__)

        self.omega0_invfs = .454*2.*np.pi
        self.gamma = 1./(self.omega0_invfs*np.asarray(self.gamma_list))
        self.gamma_dephasing = 1./(self.omega0_invfs*self.gamma_dephasing)

        # self.lindblad_matrix = np.zeros([2*self.num_levels, 2*self.num_levels], dtype=np.complex)
        self.Adagger_a = np.zeros([2*self.num_levels-1, 2*self.num_levels, 2*self.num_levels], dtype=np.complex)
        self.Amatrix= np.zeros([2*self.num_levels-1, 2*self.num_levels, 2*self.num_levels], dtype=np.complex)

        for i in range(2*self.num_levels-1):
            self.Adagger_a[i][i+1][i+1] = 1. + 0.j
            self.Amatrix[i][i][i+1] = 1. + 0.j

    def spectra_field(self, w):
        return .1*(np.cos(.5*self.t) + np.cos((.5 + w)*self.t))*np.exp(-(self.t-self.t0)**2 / self.sigma2) + \
               .1*(np.cos(w*self.t)*np.exp(-(self.t-self.t0-1000.)**2 / self.sigma2))

    def Lindblad_dt(self, field_t, mat):
        self.discrete_Ham_tot = self.discrete_Ham0 - self.mu * field_t
        return -1j*(self.discrete_Ham_tot.dot(mat) - mat.dot(self.discrete_Ham_tot)) \
            #  - self.gamma_dephasing*(mat - np.diag(np.diag(mat))) \
            # + sum([self.relaxation(k, mat) for k in range(2*self.num_levels - 1)])

    def relaxation(self, k, mat):
        return self.gamma[k]*((self.Amatrix[k].dot(mat)).dot(np.conj(self.Amatrix[k].T))
                              - 0.5*(self.Adagger_a[k].dot(mat) + mat.dot(self.Adagger_a[k])))

    def propagate_dt(self, field_t):
        rho_copy = self.rho.copy()
        k = 1
        while np.linalg.norm(rho_copy) > 1e-10:
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

    omega = .2066
    # Find energies of a harmonic oscillator V = 0.5*(omega*x)**2
    molecule1 = Propagator(
                        num_levels=4,
                        X_gridDIM=256,
                        X_amplitude=30.,
                        tsteps=10000,
                        dt=.1*2.85,
                        t0_code="dt*tsteps / 4.",
                        sigma2=2*150.**2,
                        omega=omega,
                        omega0=1.87855,
                        displacement=2.5,
                        # Vground="8 * omega * (1 - exp(-sqrt(omega/(2*8))*X))**2",
                        Vground="0.5 * (omega * X) ** 2",
                        # Vexcited="0.5 * (omega * (X-displacement)) ** 2",
                        Vexcited="10 * omega * (1 - exp(-sqrt(omega/(2*10))*(X - displacement)))**2",
                        K="0.5 * P ** 2",
                        gamma_list=[50., 50., 50., 26000., 50., 50., 50.],
                        gamma_dephasing=30.
    )

    omega_molecule2 = .2566
    molecule2 = Propagator(
        num_levels=4,
        X_gridDIM=256,
        X_amplitude=30.,
        tsteps=10000,
        dt=.1 * 2.85,
        t0_code="dt*tsteps / 4.",
        sigma2=2 * 150. ** 2,
        omega=omega_molecule2,
        omega0=1.82330,
        displacement=2.75,
        # Vground="8 * omega * (1 - exp(-sqrt(omega/(2*8))*X))**2",
        Vground="0.5 * (omega * X) ** 2",
        # Vexcited="0.5 * (omega * (X-displacement)) ** 2",
        Vexcited="10 * omega * (1 - exp(-sqrt(omega/(2*10))*(X - displacement)))**2",
        K="0.5 * P ** 2",
        gamma_list=[50., 50., 50., 26000., 50., 50., 50.],
        gamma_dephasing=32.5
    )

    # plot eigenfunctions
    plt.figure()
    plt.subplot(121)
    for n in range(molecule1.num_levels):
        plt.plot(molecule1.X, 0.05*molecule1.get_eigenstate_ground(n).real
                 + molecule1.get_energy_ground(n).real)

        plt.plot(molecule1.X, 0.05*molecule1.get_eigenstate_excited(n).real
                 + molecule1.get_energy_excited(n).real + 15. * molecule1.omega)

    steps = 256
    plt.plot(molecule1.X[256 - steps:256 + steps], molecule1.potential_ground[256 - steps:256 + steps])
    plt.plot(molecule1.X, molecule1.potential_excited + 15. * molecule1.omega)
    print molecule1.level
    print molecule1.gamma
    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 25 * molecule1.omega)

    plt.subplot(122)
    for n in range(molecule2.num_levels):
        plt.plot(molecule2.X, 0.05 * molecule2.get_eigenstate_ground(n).real
                 + molecule2.get_energy_ground(n).real)

        plt.plot(molecule2.X, 0.05 * molecule2.get_eigenstate_excited(n).real
                 + molecule2.get_energy_excited(n).real + 15. * molecule2.omega)

    steps = 256
    plt.plot(molecule2.X[256 - steps:256 + steps], molecule2.potential_ground[256 - steps:256 + steps])
    plt.plot(molecule2.X, molecule2.potential_excited + 15. * molecule2.omega)
    print molecule2.level
    print molecule2.gamma
    plt.title("Eigenfunctions for harmonic oscillator with omega = %.2f (a.u.)" % omega)
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('wave functions ($\\psi_n(x)$)')
    plt.ylim(0.0, 25 * molecule1.omega)

    plt.figure()
    plt.plot(molecule1.t, molecule1.spectra_field(w=.105))
    plt.xlabel("time (in fs)")
    plt.ylabel("Electric field (in $10^9 W/cm^2$ )")
    plt.xlim(0, 2500)
    plt.show()

    Nf = 500
    freq = np.empty(Nf)
    spectra1 = np.empty(Nf)
    spectra2 = np.empty(Nf)
    raman1 = np.empty(Nf)
    raman2 = np.empty(Nf)

    for i in range(Nf):
        freq[i] = .0734 + i*.0025
        molecule1.propagate_rho(freq[i])
        molecule2.propagate_rho(freq[i])
        pop1 = np.diag(molecule1.rho)
        pop2 = np.diag(molecule2.rho)
        spectra1[i] = pop1[molecule1.num_levels:].real.sum()
        spectra2[i] = pop2[molecule2.num_levels:].real.sum()
        raman1[i] = pop1[1].real
        raman2[i] = pop2[1].real
        print i, 15151.528*freq[i], spectra1[i], raman1[i], spectra2[i], raman2[i]

    end = time.time()
    print end - start

    plt.figure()
    plt.subplot(211)
    plt.plot(15151.528*freq, spectra1, 'r', label='$P_R$')
    plt.plot(15151.528*freq, spectra2, 'k', label='$P_{FR}$')
    plt.legend()
    plt.grid()
    plt.xlabel("Wavenumber (in $cm^{-1}$)")
    plt.ylabel("Normalized electronic absorption profile")
    plt.subplot(212)
    plt.plot(15151.528 * freq, raman1, 'r', label='$P_R$')
    plt.plot(15151.528 * freq, raman2, 'k', label='$P_{FR}$')
    plt.xlabel("Wavenumber (in $cm^{-1}$)")
    plt.ylabel("Raman absorption profile")

    plt.legend()
    plt.grid()
    plt.show()