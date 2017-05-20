import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType # this is used to dynamically add method to class
import numexpr as ne


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

        # grid for bra and kets
        self.k = np.arange(self.X_gridDIM)
        self.X = (self.k - self.X_gridDIM / 2)*self.dX
        self.k1 = self.k[:, np.newaxis]
        self.k2 = self.k[np.newaxis, :]

        # generate coordinate grid
        self.X1 = (self.k1 - self.X_gridDIM / 2)*self.dX
        self.X2 = (self.k2 - self.X_gridDIM / 2)*self.dX

        # generate momentum grid
        self.P1 = fft.fftshift((self.k1 - self.X_gridDIM / 2) * (np.pi / self.X_amplitude))
        self.P2 = fft.fftshift((self.k2 - self.X_gridDIM / 2) * (np.pi / self.X_amplitude))

        # Pre-calculate the potential energy phase
        self.expV = self.Vg(self.X2) - self.Vg(self.X1) + self.Ve(self.X2) - self.Ve(self.X1)
        self.expV = 0.5 * 1j * self.dt * self.expV
        np.exp(self.expV, out=self.expV)

        # Pre-calculate the kinetic energy phase
        self.expK = self.K(self.P2) - self.K(self.P1)
        self.expK = 1j * self.dt * self.expK
        np.exp(self.expK, out=self.expK)

    def get_CML_matrices(self, q, t):
        assert q is self.X1 or q is self.X2, "Either X1 or X2 expected as coordinate"

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

    def Vg(self, q):
        return 0.5*2*q**2 + 0.00001*q**4

    def Ve(self, q):
        return 0.5*3*(q-1)**2

    def Vge(self, q, t):
        return 0.0*(0.5*q + 0.4)*0.5*np.exp(-0.3*t**2)

    def K(self, p):
        return 0.5 * p ** 2

    def slice(self, *args):
        """
        Slice array A listed in args as if they were returned by fft.rfft(A, axis=0)
        """
        return (A[:(1 + self.X_gridDIM//2), :] for A in args)

    def single_step_propagation(self):
        """
        Perform single step propagation
        """

        # Construct T matrices
        TgL, TgeL, TeL = self.get_T_left(self.t)
        TgR, TgeR, TeR = self.get_T_right(self.t)

        # Save previous version of the density matrix
        rhoG, rhoGE, rhoE = self.rho_g, self.rho_ge, self.rho_e

        # First update the complex valued off diagonal density matrix
        self.rho_ge = (TgL*rhoG + TgeL*rhoGE.conj())*TgeR + (TgL*rhoGE + TgeL*rhoE)*TeR

        # Slice arrays to employ the symmetry (savings in speed)
        # TgL, TgeL, TeL = self.slice(TgL, TgeL, TeL)
        # TgR, TgeR, TeR = self.slice(TgR, TgeR, TeR)
        # rhoG, rhoGE, rhoE = self.slice(rhoG, rhoGE, rhoE)

        # Calculate the remaining real valued density matrix
        self.rho_g = (TgL*rhoG + TgeL*rhoGE.conj())*TgR + (TgL*rhoGE + TgeL*rhoE)*TgeR
        self.rho_e = (TgeL*rhoG + TeL*rhoGE.conj())*TgeR + (TgeL*rhoGE + TeL*rhoE)*TeR

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expV
        self.rho_g *= self.expV  # [:(1 + self.X_gridDIM//2), :]
        self.rho_e *= self.expV  # [:(1 + self.X_gridDIM//2), :]

        # --------------- x1 x2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=0, overwrite_x=True)

        # --------------- p1 x2  ->  p1 p2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=1, overwrite_x=True)

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expK
        self.rho_g *= self.expK  # [:, :(1 + self.X_gridDIM//2)]
        self.rho_e *= self.expK  # [:, :(1 + self.X_gridDIM//2)]

        # --------------- p1 p2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=1, overwrite_x=True)

        # --------------- p1 p2  ->  x1 x2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=0, overwrite_x=True)

        # self.normalize_rho()
    def normalize_rho(self):
        trace = np.trace(self.rho_e) + np.trace(self.rho_g)
        self.rho_e /= trace
        self.rho_g /= trace
        self.rho_ge /= trace

    def set_initial_rho(self, rhoG=0, rhoGE=0, rhoE=0):
        shape = (self.X_gridDIM, self.X_gridDIM)
        self.rho_g = (rhoG if isinstance(rhoG, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_e = (rhoE if isinstance(rhoE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_ge = (rhoGE if isinstance(rhoGE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        return self

if __name__ == '__main__':

    # load tools for creating animation
    import sys

    if sys.platform == 'darwin':
        # only for MacOS
        import matplotlib

        matplotlib.use('TKAgg')

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(RhoPropagate.__doc__)

    from initial_gibbs import SplitOpRho

    qsys_params = dict(
        t=0.,
        dt=0.005,

        X_gridDIM=256,
        X_amplitude=10.,

        kT=0.7,

        # kinetic energy part of the hamiltonian
        K1=lambda p: 0.5*p**2,

        # potential energy part of the hamiltonian
        V1=lambda x: 0.5*x**4,
    )

    molecule = RhoPropagate(**qsys_params)
    gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()

    molecule.set_initial_rho(gibbs_state)

    fig = plt.figure()

    ax1 = plt.subplot(221)
    plt.ylim(0.0, 1.0)
    ax2 = plt.subplot(222)
    plt.ylim(-1.0, 1.0)
    ax3 = plt.subplot(223)
    plt.ylim(-1.0, 1.0)
    ax4 = plt.subplot(224)
    plt.ylim(0.0, 1.0)

    line1, = ax1.plot(molecule.X, np.diag(molecule.rho_g.real))
    line2, = ax2.plot(molecule.X, np.diag(molecule.rho_ge.real))
    line3, = ax3.plot(molecule.X, np.diag(molecule.rho_ge.imag))
    line4, = ax4.plot(molecule.X, np.diag(molecule.rho_e.real))


    def animate(i):
        for _ in range(20):
            molecule.single_step_propagation()
            molecule.t += molecule.dt
        line1.set_ydata(np.diag(molecule.rho_g.real))  # update the data
        line2.set_ydata(np.diag(molecule.rho_ge.real))  # update the data
        line3.set_ydata(np.diag(molecule.rho_ge.imag))  # update the data
        line4.set_ydata(np.diag(molecule.rho_e.real))  # update the data
        return line1, line2, line3, line4


    # Init only required for blitting to give a clean slate.
    def init():
        line1.set_ydata(np.ma.array(molecule.X, mask=True))
        line2.set_ydata(np.ma.array(molecule.X, mask=True))
        line3.set_ydata(np.ma.array(molecule.X, mask=True))
        line4.set_ydata(np.ma.array(molecule.X, mask=True))
        return line1, line2, line3, line4

    ani = FuncAnimation(fig, animate, np.arange(1, 20), init_func=init, interval=5, blit=True)
    plt.show()