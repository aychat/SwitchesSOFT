import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType  # this is used to dynamically add method to class
import numexpr as ne
from scipy import interpolate
import pickle
import time


class RhoPropagate:
    """
    Calculates rho(t) given rho(0) using the split-operator method for 
    the Hamiltonian H = p^2/2 + V(x) with V(x) = (1/2)*(omega*x)^2 
    =================================================================
    """

    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            Vg(x) - the ground electronic state adiabatic potential curve
            Ve(x) - the first excited electronic state adiabatic potential curve
            Vge(x, t) - coupling between ground and excited states via laser-molecule interaction
            K(p) - kinetic energy
            dt - time step
            t0 (optional) - initial value of time
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

            # Check that all attributes were specified

        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert 2**int(np.log2(self.X_gridDIM)) == self.X_gridDIM, "Coordinate grid not a power of 2"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            print("Warning: Absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = 1.

        # coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = np.pi / self.X_amplitude
        self.P_amplitude = self.dP * self.X_gridDIM / 2.

        # grid for bra and kets
        self.Xrange = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)
        self.Prange = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.X_gridDIM)

        # generate coordinate grid
        self.X1 = self.Xrange[:, np.newaxis]
        self.X2 = self.Xrange[np.newaxis, :]

        # generate momentum grid
        self.P1 = fft.fftshift(self.Prange[:, np.newaxis])
        self.P2 = fft.fftshift(self.Prange[np.newaxis, :])

        # Pre-calculate the potential energy phase
        self.expV = self.Vg(self.X2) - self.Vg(self.X1) + self.Ve(self.X2) - self.Ve(self.X1)
        self.expV = 0.5 * 1j * self.dt * self.expV
        np.exp(self.expV, out=self.expV)

        # Pre-calculate the kinetic energy phase
        self.expK = self.K(self.P2) - self.K(self.P1)
        self.expK = 1j * self.dt * self.expK
        np.exp(self.expK, out=self.expK)

        self.w_coeff = np.exp(-self.gamma * self.dt)
        self.w0_coeff = 1 - self.w_coeff

        print self.w_coeff
        print self.w0_coeff

        self.gibbs_state = SplitOpRho(**kwargs).get_gibbs_state()

    def get_CML_matrices(self, q, t):
        # assert q is self.X1 or q is self.X2, "Either X1 or X2 expected as coordinate"

        Vg_minus_Ve = self.Vg(q) - self.Ve(q)
        Vge = self.Vge(q, t)

        D = np.sqrt(Vge**2 + 0.25*Vg_minus_Ve**2)

        S = np.sin(D * self.dt)
        S /= D

        C = np.cos(D * self.dt)

        M = 0.5 * S * Vg_minus_Ve

        L = S * Vge

        return C, M, L

    def get_T_left(self, t):

        C, M, L = self.get_CML_matrices(self.X1, t)
        return C-1j*M, -1j*L, C+1j*M

    def get_T_right(self, t):

        C, M, L = self.get_CML_matrices(self.X2, t)
        return C+1j*M, 1j*L, C-1j*M

    def get_T_left_A(self, t):

        C, M, L = self.get_CML_matrices(self.X1, t)
        return C + 1j * M, 1j * L, C - 1j * M

    def get_T_right_A(self, t):

        C, M, L = self.get_CML_matrices(self.X2, t)
        return C - 1j * M, -1j * L, C + 1j * M

    def Vg(self, q):
        return eval(self.codeVg)

    def Ve(self, q):
        return eval(self.codeVe)

    def Vge(self, q, t):
        return eval(self.codeVge)

    def field(self, t):
        return eval(self.codefield)

    def dipole(self, q):
        return eval(self.codedipole)

    def K(self, p):
        return eval(self.codeK)

    def slice(self, *args):
        """
        Slice array A listed in args as if they were returned by fft.rfft(A, axis=0)
        """
        return (A[:(1 + self.X_gridDIM//2), :] for A in args)

    def single_step_propagation(self, time):
        """
        Perform single step propagation
        """

        # Construct T matrices
        TgL, TgeL, TeL = self.get_T_left(time)
        TgR, TgeR, TeR = self.get_T_right(time)

        # Save previous version of the density matrix
        rhoG, rhoGE, rhoGE_c, rhoE = self.rho_g, self.rho_ge, self.rho_ge_c, self.rho_e

        # First update the complex valued off diagonal density matrix
        self.rho_ge = (TgL*rhoG + TgeL*rhoGE_c)*TgeR + (TgL*rhoGE + TgeL*rhoE)*TeR
        self.rho_ge_c = (TgeL*rhoG + TeL*rhoGE_c)*TgR + (TgeL*rhoGE + TeL*rhoE)*TgeR

        # Slice arrays to employ the symmetry (savings in speed)
        # TgL, TgeL, TeL = self.slice(TgL, TgeL, TeL)
        # TgR, TgeR, TeR = self.slice(TgR, TgeR, TeR)
        # rhoG, rhoGE, rhoE = self.slice(rhoG, rhoGE, rhoE)

        # Calculate the remaining real valued density matrix
        self.rho_g = (TgL*rhoG + TgeL*rhoGE_c)*TgR + (TgL*rhoGE + TgeL*rhoE)*TgeR
        self.rho_e = (TgeL*rhoG + TeL*rhoGE_c)*TgeR + (TgeL*rhoGE + TeL*rhoE)*TeR

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expV
        self.rho_ge_c *= self.expV
        self.rho_g *= self.expV  # [:(1 + self.X_gridDIM//2), :]
        self.rho_e *= self.expV  # [:(1 + self.X_gridDIM//2), :]

        # --------------- x1 x2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_ge_c = fftpack.ifft(self.rho_ge_c, axis=0, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=0, overwrite_x=True)

        # --------------- p1 x2  ->  p1 p2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_ge_c = fftpack.fft(self.rho_ge_c, axis=1, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=1, overwrite_x=True)

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expK
        self.rho_ge_c *= self.expK
        self.rho_g *= self.expK  # [:, :(1 + self.X_gridDIM//2)]
        self.rho_e *= self.expK  # [:, :(1 + self.X_gridDIM//2)]

        # --------------- p1 p2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_ge_c = fftpack.ifft(self.rho_ge_c, axis=1, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=1, overwrite_x=True)

        # --------------- p1 x2  ->  x1 x2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_ge_c = fftpack.fft(self.rho_ge_c, axis=0, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=0, overwrite_x=True)

        self.rho_g *= self.w_coeff
        self.rho_g += self.w0_coeff * self.gibbs_state
        self.rho_ge *= self.w_coeff
        # self.rho_ge += self.w0_coeff * self.gibbs_state
        self.rho_ge_c *= self.w_coeff
        # self.rho_ge_c += self.w0_coeff * self.gibbs_state
        self.rho_e *= self.w_coeff
        # self.rho_e += self.w0_coeff * self.gibbs_state

        self.normalize_rho()

    def normalize_rho(self):
        norm = np.trace(self.rho_g) + np.trace(self.rho_e)

        self.rho_e /= norm
        self.rho_g /= norm
        self.rho_ge /= norm
        self.rho_ge_c /= norm

    def set_initial_rho(self, rhoG=0, rhoGE=0, rhoGE_c=0, rhoE=0):
        shape = (self.X_gridDIM, self.X_gridDIM)
        self.rho_g = (rhoG if isinstance(rhoG, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_e = (rhoE if isinstance(rhoE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_ge = (rhoGE if isinstance(rhoGE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_ge_c = (rhoGE_c if isinstance(rhoGE_c, np.ndarray) else np.zeros(shape, dtype=np.complex))
        return self

    def calculate_vib_1(self, freq, q):
        vib_1 = (4.0*freq**3/np.pi)**0.25 * q * np.exp(-0.5*freq*q**2)
        return vib_1

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(RhoPropagate.__doc__)

    from initial_gibbs import SplitOpRho

    qsys_params = dict(
        t=0.,
        dt=0.01,

        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=500,
        field_sigma2=2 * .6 ** 2,
        gamma=0.5,

        # kinetic energy part of the hamiltonian
        codeK="0.5*p**2",
        freq_Vg=1.075,
        freq_Ve=1.075,
        disp=1.,
        Ediff=9.,
        delt=0.75,
        # potential energy part of the hamiltonian
        codeVg="0.5*(self.freq_Vg*q)**2",
        codeVe="0.5*(self.freq_Ve*(q-self.disp))**2 + self.Ediff",
        codeVge="-.05*q*self.field(t)",
        codedipole=".05*q",
        codefield="10.*np.exp(-(1./self.field_sigma2)*(t - 0.5*self.dt*self.Tsteps)**2)*np.cos(self.delt*self.Ediff*t)"
    )
    start = time.time()
    molecule = RhoPropagate(**qsys_params)

    # molecule.set_initial_rho(molecule.gibbs_state)
    t = np.linspace(0.0, molecule.dt*molecule.Tsteps, molecule.Tsteps)
    #
    plt.figure()
    plt.plot(3.5*t, (5.7e-4**2)*3.55e16*molecule.field(t))
    plt.xlabel("time  (in fs)")
    plt.ylabel("Field strength (in $W/{cm}^2$)")
    plt.show()

    N = 50
    data = np.zeros(N)
    freq = np.zeros(N)
    spectra = np.zeros(molecule.Tsteps)
    for k in range(N):

        molecule.delt += .02
        print molecule.delt

        molecule.set_initial_rho(molecule.gibbs_state)
        for j in range(molecule.Tsteps):
            molecule.single_step_propagation(j * molecule.dt)
            spectra[j] = np.trace(molecule.rho_ge.dot(np.diag(molecule.dipole(molecule.Xrange)))) \
                         + np.trace(molecule.rho_ge_c.dot(np.diag(molecule.dipole(molecule.Xrange))))

            field_grad = np.gradient(molecule.field(t))
            spectra[j] *= -field_grad[j]
            spectra[j] /= (molecule.field(t)**2).max()
        data[k] = np.sum(spectra)
        freq[k] = 1240./(molecule.delt*molecule.Ediff*.187855)
        print data[k], freq[k]

    plt.figure()
    plt.plot(freq, data)
    plt.show()

    end = time.time()
    print end - start

