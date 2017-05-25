import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType
from rho_propagate import RhoPropagate
from initial_gibbs import SplitOpRho
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

        self.A_params = np.random.uniform(0.01, 1., self.field_freq_num)
        self.phi_params = np.random.uniform(0.0, 1.0, self.field_freq_num)
        self.freq = np.linspace(self.field_freq_min, self.field_freq_max, self.field_freq_num)

        self.gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()
        self.Tmax = int(self.dt * self.Tsteps)
        self.code_envelope = "np.exp(-50.*(t-0.25)**2)"
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

    def diff_field_params(self, t):
        for index in range(self.field_freq_num):
            self.diff_field_A_params[index] = \
                eval(self.code_envelope)*np.cos(self.freq[index]*t + self.phi_params[index])
            self.diff_field_phi_params[index] = \
                -eval(self.code_envelope)*self.A_params[index]*np.cos(self.freq[index]*t + self.phi_params[index])

    def dipole(self, q):
        return eval(self.codeDipole)

    def g_propagate_t_T(self, tau_index):

        t_ini = tau_index*self.dt

        # ---------------------- COMPUTE ELEMENTS OF [mu(x), rho(t)] = rho_mu(t) -----------------------#
        rho_mu_diag_t = 2. * np.diag(self.dipole(self.X)) * self.rho_ge.imag
        # rho_mu_offdiag_t = 1.j * np.diag(self.dipole(self.X)) * (self.rho_g - self.rho_e)
        #
        # # ------------------- PROPAGATING mu_rho(t) ---> mu_rho(T) -------------------------------------#
        # for _ in range(self.Tsteps - tau_index):
        #     rho_mu_diag_t = \
        #         self.single_step_propagation_mu_rho(rho_mu_diag_t, rho_mu_offdiag_t, t_ini)
        #     t_ini += self.dt

        return rho_mu_diag_t

    def A_propagate_T_t(self, tau_index):
        time = self.dt*tau_index
        rho_A = self.gibbs_state.real
        for _ in range(self.Tsteps - tau_index):
            rho_A = self.single_step_propagation_A_inverse(rho_A, time)
            time += self.dt

        return rho_A

    def calculate_dJ_dE(self):

        Tsteps = self.Tsteps
        Nfreq = self.field_freq_num
        dt = self.dt

        # ------------------------------------ rho(0) IS CREATED -------------------------------------- #
        self.set_initial_rho(self.gibbs_state.real)

        for tau_index in range(Tsteps):
            self.dJ_dE[tau_index] = (self.A_propagate_T_t(tau_index)*self.g_propagate_t_T(tau_index).T).sum()
            self.single_step_propagation(tau_index*self.dt)

        for i in range(Nfreq):
            dA = 0.0
            dphi = 0.0
            for tau_i in range(Tsteps):
                t = tau_i * dt
                gaussian = eval(self.code_envelope)
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

                dA += self.dJ_dE[tau_i]*gaussian * dEt_dA * k
                dphi += self.dJ_dE[tau_i]*gaussian * dEt_dphi * k

            self.dJ_dA[i] = dA*dt/3.0
            self.dJ_dphi[i] = dphi*dt/3.0

        return np.concatenate((self.dJ_dA, self.dJ_dphi))

if __name__ == "__main__":

    from plot_functions import animate_1d_subplots, animate_2d_imshow, plot_2d_subplots
    import matplotlib.pyplot as plt
    
    qsys_params = dict(
        t=0.,
        dt=0.01,
        ds=0.1,
        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,

        Tsteps=50,
        field_sigma=2.5,
        field_freq_num=4,
        field_freq_min=0.1,
        field_freq_max=3.1,

        codeK="0.5*p**2",
        codeVg="-0.05*q**2 + 0.03*q**4",
        codeVe="0.5*3*(q-1)**2",
        codeDipole="100.*q",
        codeVge="(0.5*q + 0.4)*self.field(t)"
    )

    molecule = OptimalFieldRho(**qsys_params)

    from scipy.integrate import ode

    y0, t0 = np.concatenate((molecule.A_params, molecule.phi_params)), 0.

    def f(t, y):
        result = molecule.calculate_dJ_dE()
        return result

    def propagate_rho(system):
        system.set_initial_rho(system.gibbs_state.real)
        for t_indx in range(system.Tsteps):
            system.single_step_propagation(system.dt*t_indx)
        return (system.gibbs_state.real*system.rho_e.T).sum()

    r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0, t0)
    s_max = 25.0
    dJ_ds = []
    while r.successful() and r.t < s_max:
        r.integrate(r.t + molecule.ds)
        molecule.A_params = r.y[:molecule.field_freq_num]
        dJ_ds.append(propagate_rho(molecule))
        print dJ_ds[-1]
        molecule.phi_params = r.y[molecule.field_freq_num:]
        # print r.t, r.y

    plt.plot(dJ_ds, 'ro')
    plt.show()