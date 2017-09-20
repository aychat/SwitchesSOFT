import numpy as np

# numpy.fft has better implementation of real fourier transform
# necessary for real split operator propagator
from numpy import fft
from scipy import fftpack


class PropagateGibbs:
    """
    The second-order split-operator propagator for the Moyal equation for density matrix
    with the time-dependent Hamiltonian H = K(p, t) + V(x, t).
    (K and V may not depend on time.)

    This implementation stores rho as a 2D real array.
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            V(x) - potential energy (as a function) may depend on time
            K(p) - momentum dependent part of the hamiltonian (as a function) may depend on time
            dt - time step
            t (optional) - initial value of time
        """

        # save all attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert self.X_gridDIM % 2 == 0, "Coordinate grid size (X_gridDIM) must be even"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.V
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        # get coordinate and momentum step sizes
        self.dX = 2.*self.X_amplitude / self.X_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX , self.X_gridDIM)
        self.X1 = self.X[np.newaxis, :]
        self.X2 = self.X[:, np.newaxis]

        self.P = fft.fftfreq(self.X_gridDIM, self.dX/(2*np.pi))
        self.P1 = self.P[np.newaxis, :]
        self.P2 = self.P[:, np.newaxis]

        try:
            # Pre-calculate the exponent, if the potential is time independent
            self._expV = np.exp(
                -self.dt*0.5j*(self.V(self.X1) - self.V(self.X2))
            )
        except TypeError:
            # If exception is generated, then the potential is time-dependent
            # and caching is not possible
            pass

        try:
            # Pre-calculate the exponent, if the kinetic energy is time independent
            self._expK = np.exp(
                -self.dt*1j*(self.K(self.P1) - self.K(self.P2))
            )
        except TypeError:
            # If exception is generated, then the kinetic energy is time-dependent
            # and caching is not possible
            pass

    def single_step_propagation(self):
        """
        Perform single step propagation. The final rho function is not normalized.
        :return: self.rho
        """
        expV = self.get_expV(self.t)

        self.rho *= expV

        # x1 x2  ->  p1 x2
        self.rho = fftpack.ifft(self.rho, axis=0)

        # p1 x2  ->  p1 p2
        self.rho = fftpack.fft(self.rho, axis=1)
        self.rho *= self.get_expK(self.t)

        # p1 p2  ->  p1 x2
        self.rho = fftpack.ifft(self.rho, axis=1)

        # p1 x2  ->  x1 x2
        self.rho = fftpack.fft(self.rho, axis=0)
        self.rho *= expV

        # increment current time
        self.t += self.dt

        return self.rho

    def propagate(self, time_steps=1):
        """
        Time propagate the rho function saved in self.rho
        :param time_steps: number of self.dt time increments to make
        :return: self.rho
        """

        for _ in xrange(time_steps):

            # advance by one time step
            self.single_step_propagation()

            # normalization
            self.rho /= np.trace(self.rho)

        return self.rho

    def get_expV(self, t):
        """
        Return the exponent of the potential energy difference at time (t)
        """
        try:
            # aces the pre-calculated value
            return self._expV
        except AttributeError:
            # Calculate in efficient way
            result = -self.dt*0.5j*(self.V(self.X1, t) - self.V(self.X2, t))
            return np.exp(result, out=result)

    def get_expK(self, t):
        """
        Return the exponent of the kinetic energy difference at time  (t)
        """
        try:
            # aces the pre-calculated value
            return self._expK
        except AttributeError:
            # Calculate result = np.exp(*self.K(self.P1, self.P2, t))
            result = -self.dt*1j*(self.K(self.P1, t) - self.K(self.P2, t))
            return np.exp(result, out=result)

    def get_K(self, t):
        """
        Return the kinetic energy at time (t)
        """
        try:
            return self._K
        except AttributeError:
            return self.K(self.P, t)

    def get_V(self, t):
        """
        Return the potential energy at time (t)
        """
        try:
            return self._V
        except AttributeError:
            return self.V(self.X, t)

    def set_rho(self, new_rho_func):
        """
        Set the initial rho function
        :param new_rho_func: 2D numoy array contaning the rho function
        :return: self
        """
        # perform the consistency checks
        assert new_rho_func.shape == (self.X_gridDIM, self.X_gridDIM), \
            "The grid sizes does not match with the rho function"

        assert new_rho_func.dtype == np.float, "Supplied rho function must be real"

        # make sure the rho function is stored as a complex array
        self.rho = new_rho_func.copy()

        # normalize
        self.rho /= self.rho.sum() * self.dX*self.dP

        return self

##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':
    # load tools for creating animation
    import sys

    if sys.platform == 'darwin':
        # only for MacOS
        import matplotlib

        matplotlib.use('TKAgg')

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from initial_gibbs import SplitOpRho
    # Use the documentation string for the developed class
    print(PropagateGibbs.__doc__)

    qsys_params = dict(
        t=0.,
        dt=0.005,

        X_gridDIM=256,
        X_amplitude=10.,

        kT=0.7,

        # kinetic energy part of the hamiltonian
        K1=lambda p: 0.5 * p ** 2,

        # potential energy part of the hamiltonian
        V1=lambda x: 0.5 * x ** 4,
    )

    class VisualizeDynamics:

        def __init__(self, fig):
            """
            Initialize all propagators and frame
            :param fig: matplotlib figure object
            """
            #  Initialize systems
            self.set_quantum_sys()

            #################################################################
            #
            # Initialize plotting facility
            #
            #################################################################

            self.fig = fig

            ax = fig.add_subplot(111)

            ax.set_title('rho, $\\rho(x,p,t)$')
            extent = [self.quant_sys.X1.min(), self.quant_sys.X1.max(), self.quant_sys.X2.min(),
                      self.quant_sys.X2.max()]

            # import utility to visualize the rho function
            from rho_normalize import RhoNormalize

            # generate empty plot
            self.img = ax.imshow([[0]],
                                 extent=extent,
                                 origin='lower',
                                 cmap='seismic',
                                 norm=RhoNormalize(vmin=-0.01, vmax=0.1)
                                 )

            self.fig.colorbar(self.img)

            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$x^{\'}$ (a.u.)')

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.syparams = qsys_params
            molecule = SplitOpRho(**qsys_params)
            self.quant_sys.set_rho(molecule.get_gibbs_state())

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()
            self.img.set_array([[0]])
            return self.img,

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            # propagate the rho function
            self.img.set_array(self.quant_sys.propagate(20).real)
            return self.img,


    fig = plt.gcf()
    visualizer = VisualizeDynamics(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(100),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)
    plt.show()



