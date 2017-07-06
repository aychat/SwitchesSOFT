import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType
from rho_propagator import Propagator
from initial_gibbs import SplitOpRho
from gibbs_state_A import SplitOpRho_A
import pickle
import os


class HillClimb(Propagator):
    """
    Calculates field parameters to maximize rho_e(T) given rho_(0) using stochastic hill climbing for 
    the Hamiltonian H = p^2/2 + V(x) for a 2d system interacting with an electric field 
    ==================================================================================================
    """
    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            dt - time step
            t (optional) - initial value of time
        """
        Propagator.__init__(self, **kwargs)
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

        assert 2 ** int(np.log2(self.X_gridDIM)) == self.X_gridDIM, "Coordinate grid not a power of 2"

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
            self.field_freq_num
        except AttributeError:
            raise AttributeError("# of frequency terms not specified")

        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)

        # generate coordinate grid
        self.X1 = self.Xrange[:, np.newaxis]
        self.X2 = self.Xrange[np.newaxis, :]

        if os.path.getsize("/home/ayanc/PycharmProjects/Switches/new_params.txt") > 0:
            with open("new_params.txt", "rb") as f:
                data = pickle.load(f)
            self.A_params = data["A_params"]
            self.phi_params = data["phi_params"]
            self.fitness = data["fitness"]
            self.field_list = data["field_list"]
            print "New parameters from file"
        else:
            self.A_params = np.random.uniform(5., 15., self.field_freq_num)
            self.phi_params = np.random.uniform(0.01, 0.1, self.field_freq_num)
            self.fitness = [0.]
            self.field_list = []
            print "New random parameters"

        self.freq = np.linspace(self.field_freq_min, self.field_freq_max, self.field_freq_num)

        self.gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()
        self.gibbs_state_A = SplitOpRho_A(**qsys_params).get_gibbs_state_A()

        self.Tmax = int(self.dt * self.Tsteps)
        self.code_envelope = "np.exp(-(1./ self.field_sigma2)*(t - (self.dt*self.Tsteps)/2.)**2)"
        self.dJ_dE = np.empty(self.Tsteps)
        self.dJ_dA = np.empty(self.field_freq_num)
        self.dJ_dphi = np.empty(self.field_freq_num)
        self.time_array = np.linspace(0.0, self.dt*self.Tsteps, self.Tsteps)

    def field(self, t):
        sum_field = 0.0
        for i in range(self.field_freq_num):
            sum_field += self.A_params[i]*np.cos(t*self.freq[i] + self.phi_params[i])
        return eval(self.code_envelope)*sum_field

    def climb_singe_step(self):
        #  rho_g_old, rho_ge_old, rho_ge_c_old, rho_e_old = self.rho_g, self.rho_ge, self.rho_ge_c, self.rho_e
        self.t = 0.
        for _ in range(self.Tsteps):
            self.single_step_propagation(self.t)
            self.t += self.dt
        self.fitness.append(np.trace(self.rho_e).real)

    def optimize(self):
        for _ in range(self.hill_steps):
            A_params_old = np.copy(self.A_params)
            phi_params_old = np.copy(self.phi_params)
            self.A_params += np.random.uniform(self.field_A_delta_min, self.field_A_delta_max, self.field_freq_num)
            self.phi_params += \
                np.random.uniform(self.field_phi_delta_min, self.field_phi_delta_max, self.field_freq_num)
            self.climb_singe_step()
            if self.fitness[-1] <= self.fitness[-2]:
                # print self.A_params
                # print self.fitness[-1]
                self.A_params = np.copy(A_params_old)
                self.phi_params = np.copy(phi_params_old)
                # del(self.fitness[-1])
                self.fitness[-1] = self.fitness[-2]
            print _, self.fitness[-1]
            self.field_list.append(self.field(self.time_array))


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(HillClimb.__doc__)

    from initial_gibbs import SplitOpRho

    qsys_params = dict(
        t=0.,
        dt=0.005,
        hill_steps=1,
        field_A_delta_min=-1.,
        field_A_delta_max=1.,
        field_phi_delta_min=-.05,
        field_phi_delta_max=.05,

        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=3000,
        field_sigma2=2*1.8**2,
        field_freq_num=20,
        field_freq_min=8.,
        field_freq_max=12.,
        gamma=0.0001,

        # kinetic energy part of the hamiltonian
        codeK="0.5*p**2",

        # potential energy part of the hamiltonian
        codeVg="0.5*(.3*q)**2",
        codeVe="0.5*(.3*(q-self.X_amplitude*0.1))**2 + 10.",
        codeVge="-.05*q*self.field(t)",
        code_envelope="np.exp(-(1./ self.field_sigma2)*(t - (self.dt*self.Tsteps)/2.)**2)"
    )

    molecule = HillClimb(**qsys_params)
    gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()

    molecule.set_initial_rho(gibbs_state)

    # plt.figure()
    # plt.plot(molecule.time_array*3.5, molecule.field(t)*5.7e-4)
    # plt.ylabel("$E(t)$ in (a.u.)")
    # plt.xlabel("time (in fs)")
    # plt.grid()
    # plt.show()

    # molecule.optimize()
    # plt.plot(molecule.fitness)
    # plt.grid()
    # plt.show()

    # with open("new_params.txt", "wb") as f:
    #     pickle.dump(
    #         {
    #             'A_params': molecule.A_params,
    #             'phi_params': molecule.phi_params,
    #             'fitness': molecule.fitness,
    #             'field_list': molecule.field_list
    #         },
    #         f
    #     )

    # print molecule.A_params, molecule.phi_params
    # print molecule.fitness

    field_initial = np.array(molecule.field_list[0])
    field_final = np.array(molecule.field_list[-1])
    plt.figure()
    plt.plot(3.5*molecule.time_array, 5.7e-4*field_initial, 'r', label='Initial Field')
    plt.plot(3.5*molecule.time_array, 5.7e-4*field_final, 'k', label='Final Field')
    plt.legend()
    plt.ylabel("Electric field (in a.u.)")
    plt.xlabel("time (in fs)")
    plt.grid()
    plt.show()
