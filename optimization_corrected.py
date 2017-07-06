import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType
from propagator import RhoPropagate
from initial_gibbs import SplitOpRho
from gibbs_state_A import SplitOpRho_A
from scipy import integrate


class OptimalFieldRho(RhoPropagate):
    """
    Calculates field parameters to maximize rho_e(T) given rho_(0) using the split-operator method for 
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
        RhoPropagate.__init__(self, **kwargs)
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

        self.A_params = np.random.uniform(0.01, 0.1, self.field_freq_num)
        # self.A_params = np.zeros(self.field_freq_num)
        # self.A_params.fill(0.01)
        self.phi_params = np.random.uniform(0.01, 0.1, self.field_freq_num)
        # self.phi_params = np.zeros(self.field_freq_num)
        # self.phi_params.fill(0.0)

        self.freq = np.linspace(self.field_freq_min, self.field_freq_max, self.field_freq_num)

        self.gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()
        self.gibbs_state_A = SplitOpRho_A(**qsys_params).get_gibbs_state_A()

        self.Tmax = int(self.dt * self.Tsteps)
        self.code_envelope = "np.exp(-(t - (self.dt*self.Tsteps)/2.)**2 / self.field_sigma2)"
        self.diff_field_A_params = np.empty(self.field_freq_num)
        self.diff_field_phi_params = np.empty(self.field_freq_num)
        self.dJ_dE = np.empty(self.Tsteps)
        self.dJ_dA = np.empty(self.field_freq_num)
        self.dJ_dphi = np.empty(self.field_freq_num)

    def field(self, t):
        sum_field = 0.0
        for i in range(self.field_freq_num):
            sum_field += self.A_params[i]*np.cos(t*self.freq[i] + self.phi_params[i])
        return eval(self.code_envelope)*sum_field

    def dipole(self, q):
        return eval(self.codeDipole)

    def calculate_dJ_dE(self):

        Tsteps = self.Tsteps
        Nfreq = self.field_freq_num
        dt = self.dt

        # ------------------------------------ rho(0) IS CREATED -------------------------------------- #
        self.set_initial_rho(self.gibbs_state.real)
        rhomu_g = np.zeros_like(self.gibbs_state)
        rhomu_ge = np.zeros_like(rhomu_g)
        rhomu_ge_c = np.zeros_like(rhomu_g)
        rhomu_e = np.zeros_like(rhomu_g)

        for tau_index in range(Tsteps):
            rhomu_g = np.diag(self.dipole(self.X)).dot((self.rho_ge_c - self.rho_ge))
            rhomu_ge = np.diag(self.dipole(self.X)).dot((self.rho_e - self.rho_g))
            rhomu_ge_c = - rhomu_ge
            rhomu_e = -rhomu_g

            self.single_step_propagation(tau_index*self.dt)
            rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e = \
                self.single_step_propagation_mu_rho(rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e, (Tsteps-tau_index)*dt)

            self.dJ_dE[tau_index] = (2.j * (self.gibbs_state_A * rhomu_e.T).sum()).real

        for i in range(Nfreq):
            dA = 0.0
            dphi = 0.0
            for tau_i in range(Tsteps):
                t = tau_i * dt
                envelope_func = eval(self.code_envelope)
                dEt_dA = np.cos(self.freq[i]*t + self.phi_params[i])
                dEt_dphi = -self.A_params[i]*np.sin(self.freq[i]*t + self.phi_params[i])

                if tau_i == 0:
                    k = 1
                elif tau_i == (Tsteps-1):
                    k = 1
                elif tau_i % 2 == 0:
                    k = 2
                elif tau_i % 2 == 1:
                    k = 4

                dA += self.dJ_dE[tau_i]*envelope_func * dEt_dA * k
                dphi += self.dJ_dE[tau_i]*envelope_func * dEt_dphi * k

            self.dJ_dA[i] = dA*dt/3.0
            self.dJ_dphi[i] = dphi*dt/3.0

        return np.concatenate((self.dJ_dA, self.dJ_dphi))

    def propagate_rho(self):
        self.set_initial_rho(self.gibbs_state.real)
        for t_indx in range(self.Tsteps):
            self.single_step_propagation(self.dt*t_indx)
        return np.trace(self.gibbs_state_A.real.dot(self.rho_e.real))

if __name__ == "__main__":

    from plot_functions import animate_1d_subplots, animate_2d_imshow, plot_2d_subplots
    import matplotlib.pyplot as plt
    import pickle


    qsys_params = dict(
        t=0.,
        dt=0.1,
        ds=1.0,
        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,

        Tsteps=200,
        field_sigma2=4.5,
        field_freq_num=10,
        field_freq_min=1,
        field_freq_max=1.5,

        codeK="0.5*p**2",
        codeVg="1. * q ** 2",
        codeVe="0.5*3*(q-2.)**2 + 0.35",
        codeDipole="-q",
        codeVge="-q*self.field(t)"
    )

    molecule = OptimalFieldRho(**qsys_params)
    # molecule.set_initial_rho(molecule.gibbs_state)
    # animate_1d_subplots(molecule, molecule.rho_g, molecule.rho_ge, molecule.rho_e)

    # plt.figure()
    # plt.plot(molecule.X, np.diag(molecule.gibbs_state), 'r', label="GS_ground")
    # plt.plot(molecule.X, np.diag(molecule.gibbs_state_A), 'k', label="GS_excited")
    # plt.legend()
    # plt.grid()
    # plt.show()
    #
    # molecule.calculate_dJ_dE()
    # plt.figure()
    # plt.plot(molecule.dJ_dE)
    # plt.show()

    from scipy.integrate import ode

    y0, t0 = np.concatenate((molecule.A_params, molecule.phi_params)), 0.

    def f(t, y):
        # print t
        result = molecule.calculate_dJ_dE()
        return result

    time = np.linspace(0.0, molecule.dt*molecule.Tsteps, molecule.Tsteps)

    plt.figure()
    plt.plot(time, molecule.field(time))
    plt.show()

    plt.figure()
    plt.plot(time, molecule.field(time))

    r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0, t0)
    s_max = 50.
    s_run = [0.0]
    J = [0.0]
    while r.successful() and r.t < s_max:
        r.integrate(r.t + molecule.ds)
        old_A_params = molecule.A_params
        old_phi_params = molecule.phi_params
        molecule.A_params = r.y[:molecule.field_freq_num]
        molecule.phi_params = r.y[molecule.field_freq_num:]
        J.append(molecule.propagate_rho())
        # while dJ_ds[-1] < dJ_ds[-2]:
        #     r.t -= molecule.ds
        #     molecule.ds /= 2.0
        #     print "ds changed to ", molecule.ds, "  dJ_ds = ", dJ_ds[-1]
        #     r.y[:molecule.field_freq_num] = old_A_params
        #     r.y[molecule.field_freq_num:] = old_phi_params
        #     r.integrate(r.t + molecule.ds)
        #     molecule.A_params = r.y[:molecule.field_freq_num]
        #     molecule.phi_params = r.y[molecule.field_freq_num:]
        #     del dJ_ds[-1]
        #     dJ_ds.append(molecule.propagate_rho())
        #     if molecule.ds < 1.e-5:
        #         break
        s_run.append(s_run[-1]+molecule.ds)
        print J[-1], s_run[-1], r.t
        # if dJ_ds[-1] - dJ_ds[-2] < 1.e-8:
        #     break

    plt.plot(time, molecule.field(time))
    plt.show()

    plt.figure()
    plt.plot(molecule.X, np.diag(molecule.rho_g + molecule.rho_e))
    plt.show()

    with open("saved_data.pickle", "wb") as f:
        pickle.dump(
            {
                'J': J,
                's array': s_run
            },
            f
        )

    plt.plot(s_run[1:-1], J[1:-1], 'ro-')
    plt.show()
