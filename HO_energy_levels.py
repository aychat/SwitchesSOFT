import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType

class Eigenstates:
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

        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = np.pi / self.X_amplitude
        self.P_amplitude = self.dP * self.X_gridDIM / 2.

        # grid for bra and kets
        self.Xrange = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # generate coordinate grid
        self.X1 = self.Xrange[:, np.newaxis]
        self.X2 = self.Xrange[np.newaxis, :]

    def Vg(self, q):
        return eval(self.codeVg)

    def Ve(self, q):
        return eval(self.codeVe)

    def calculate_eigenstates(self, freq, q):
        vib_0 = (freq/np.pi)**0.25 * np.exp(-0.5*freq*q**2)
        vib_1 = (4.0*freq**3/np.pi)**0.25 * q * np.exp(-0.5*freq*q**2)
        vib_2 = (freq/(4.*np.pi))**0.25 * (2.*freq*q**2 - 1.) * np.exp(-0.5*freq*q**2)
        vib_3 = (freq**3/(9.*np.pi))**0.25 * (2.*freq*q**3 - 3.*q) * np.exp(-0.5*freq*q**2)
        return vib_0, vib_1, vib_2, vib_3


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Use the documentation string for the developed class
    print(Eigenstates.__doc__)

    qsys_params = dict(
        t=0.,
        dt=0.1,

        X_gridDIM=512,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=100,

        # kinetic energy part of the hamiltonian
        codeK="0.5*p**2",

        # potential energy part of the hamiltonian
        freqVg=1.,
        freqVe=1.25,
        ge_displacement = 3.5,
        codeVg="0.5 * (self.freqVg * q) ** 2",
        codeVe="0.5 * (self.freqVe * (q-self.ge_displacement))**2",
        codeVge="(2.0*q + 0.4)*self.field(t)",
        codefield="2.5*np.exp(-0.1*(t - 0.5*self.dt*self.Tsteps)**2)*np.cos(4.5*t)"
    )

    molecule = Eigenstates(**qsys_params)
    vib0_g, vib1_g, vib2_g, vib3_g = molecule.calculate_eigenstates(molecule.freqVg, molecule.Xrange)
    vib0_e, vib1_e, vib2_e, vib3_e = molecule.calculate_eigenstates(molecule.freqVe,
                                                                    molecule.Xrange - molecule.ge_displacement)

    plt.figure()
    plt.subplot(221)
    plt.plot(molecule.Xrange, vib0_e + 0.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib1_e + 1.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib2_e + 2.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib3_e + 3.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, molecule.Ve(molecule.Xrange))
    plt.ylim(0., 4.5*molecule.freqVe)

    plt.subplot(222)
    plt.plot(molecule.Xrange, vib0_e**2 + 0.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib1_e**2 + 1.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib2_e**2 + 2.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, vib3_e**2 + 3.5 * molecule.freqVe)
    plt.plot(molecule.Xrange, molecule.Ve(molecule.Xrange))
    plt.ylim(0., 4.5*molecule.freqVe)

    plt.subplot(223)
    plt.plot(molecule.Xrange, vib0_g + 0.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib1_g + 1.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib2_g + 2.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib3_g + 3.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, molecule.Vg(molecule.Xrange))
    plt.ylim(0., 4.5*molecule.freqVg)

    plt.subplot(224)
    plt.plot(molecule.Xrange, vib0_g**2 + 0.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib1_g**2 + 1.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib2_g**2 + 2.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, vib3_g**2 + 3.5 * molecule.freqVg)
    plt.plot(molecule.Xrange, molecule.Vg(molecule.Xrange))
    plt.ylim(0., 4.5*molecule.freqVg)
    plt.show()
