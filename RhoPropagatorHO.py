import numpy as np
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
        Initializes required parameters
        :param kwargs:
        X_gridDIM - the coordinate grid size
        X_amplitude - maximum value of the coordinates
        t (optional) - initial value of time (default t = 0)
        V - a string of the C code specifying potential energy. Coordinate (X) and time (t) variables are declared.
        K - a string of the C code specifying kinetic energy. Momentum (P) and time (t) variables are declared.
        dt - time step
        abs_boundary_p (optional) - a string of the C code specifying function of P,
                                which will be applied to the density matrix at each propagation step
        abs_boundary_x (optional) - a string of the C code specifying function of X,
                                which will be applied to the density matrix at each propagation step
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

        assert 2**int(np.log2(self.X_gridDIM))==self.X_gridDIM, "Coordinate grid not a power of 2"

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

        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.k = np.arange(self.X_gridDIM)
        self.X = (self.k - self.X_gridDIM / 2)*self.dX

        self.P = (self.k - self.X_gridDIM / 2) * (np.pi / self.X_amplitude)
        self.dP = self.P[1] - self.P[0]

        self.rho = np.empty([self.X_gridDIM, self.X_gridDIM], dtype=np.complex)

        for X1 in range(self.X_gridDIM):
            for X2 in range(self.X_gridDIM):
                x1 = (X1 - self.X_gridDIM / 2)*self.dX
                x2 = (X2 - self.X_gridDIM / 2)*self.dX
                self.rho[X1, X2] = np.exp(-self.sigma*((x1-self.X0)**2 + (x2-self.X0)**2))
        self.rho /= np.trace(self.rho)

    def minus(self):
        k = np.arange(self.X_gridDIM)
        k1 = k[:, np.newaxis]
        k2 = k[np.newaxis, :]

        return (-1)**(k1+k2)

    def expV(self):
        V = -0.5j*self.dt*0.5 * self.omega * self.omega * self.X * self.X
        return linalg.expm(self.minus()*np.diag(V))

    def expK(self):
        K = -0.5j*self.dt*0.5 * self.P * self.P
        return linalg.expm(np.diag(K))

    def single_step_propagation(self):
        self.rho = self.expK()*np.dot(self.expV(), fftpack.ifft2(self.expK()
                        * fftpack.fft2(np.dot(self.expV(), self.rho), axes=(-2, -1)), axes=(-2, -1)))
        self.rho /= np.linalg.norm(self.rho)
        self.t += self.dt
        return self.rho

    def propagate(self):
        for i in range(self.Tsteps):
            self.single_step_propagation()
        return 0

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib import cm

    print RhoPropagate.__doc__

    for x in range(1, 5):
        globals()['HarmonicOscillator%s' % x] = RhoPropagate(
            t=0.,
            dt=0.1,
            Tsteps=2,
            X_gridDIM=256,
            X_amplitude=1.,
            sigma=np.random.uniform(2., 4.),
            X0=np.random.uniform(-0.5, 0.5),
            omega=np.random.uniform(1.5, 2.5),
            abs_boundary=1.
    )

    fig = plt.figure()
    for x in range(1, 5):
        Xr = globals()['HarmonicOscillator%s' % x].X
        plt.subplot(220 + x)
        cim = plt.imshow(
            globals()['HarmonicOscillator%s' % x].rho.real, extent=(Xr.min(), Xr.max(), Xr.min(),
                                                                    Xr.max()), origin='lower', cmap=cm.hot
        )
        fig.colorbar(cim)

    for x in range(1, 5):
        globals()['HarmonicOscillator%s' % x].propagate()

    fig = plt.figure()
    for x in range(1, 5):
        Xr = globals()['HarmonicOscillator%s' % x].X
        plt.subplot(220 + x)
        cim = plt.imshow(
            globals()['HarmonicOscillator%s' % x].rho.real, extent=(Xr.min(), Xr.max(), Xr.min(),
                                                                    Xr.max()), origin='lower', cmap=cm.hot
        )
        fig.colorbar(cim)

    fig = plt.figure()
    for x in range(1, 5):
        Xr = globals()['HarmonicOscillator%s' % x].X
        plt.plot(Xr, 0.25*x+np.diag(globals()['HarmonicOscillator%s' % x].rho.T.real))
    plt.show()