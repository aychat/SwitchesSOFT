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

        # generate coordinate grid
        self.X1 = self.Xrange[:, np.newaxis]
        self.X2 = self.Xrange[np.newaxis, :]

        self.A_params = np.random.uniform(5., 15., self.field_freq_num)
        self.phi_params = np.random.uniform(0.01, 0.1, self.field_freq_num)

        # self.A_params = [-0.64898977, -1.69550489, -2.0395354, -2.63135685, -3.31247001, -3.48197219,
        #                      -4.23476381, -4.50149803, -4.56541647, -4.81933806]
        # self.phi_params = [3.1880468, 1.52498729, 0.88176458, -0.07275246, -0.86039341, -1.16403224,
        #                        -1.44293199, -1.4153083, -1.40767955, -1.07932149]

        self.freq = np.linspace(self.field_freq_min, self.field_freq_max, self.field_freq_num)

        self.gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()
        self.gibbs_state_A = SplitOpRho_A(**qsys_params).get_gibbs_state_A()

        self.Tmax = int(self.dt * self.Tsteps)
        self.code_envelope = "np.exp(-(1./ self.field_sigma2)*(t - (self.dt*self.Tsteps)/2.)**2)"
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

    def propagate_mu_rho(self, mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e, tau_index):
        for i in range(self.Tsteps - tau_index):
            mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e \
                = self.single_step_propagation_mu_rho(mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e, self.dt*(i + tau_index
                                                                                                           ))
            return mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e

    def calculate_dJ_dE(self):

        Tsteps = self.Tsteps
        Nfreq = self.field_freq_num
        dt = self.dt

        # ------------------------------------ rho(0) IS CREATED -------------------------------------- #
        self.set_initial_rho(self.gibbs_state)

        for tau_index in range(self.Tsteps):
            mu_rho_g = np.diag(self.dipole(self.Xrange)).dot(self.rho_ge_c) \
                       - self.rho_ge.dot(np.diag(self.dipole(self.Xrange)))
            mu_rho_ge = np.diag(self.dipole(self.Xrange)).dot(self.rho_e) \
                       - self.rho_g.dot(np.diag(self.dipole(self.Xrange)))
            mu_rho_ge_c = np.diag(self.dipole(self.Xrange)).dot(self.rho_g) \
                       - self.rho_e.dot(np.diag(self.dipole(self.Xrange)))
            mu_rho_e = np.diag(self.dipole(self.Xrange)).dot(self.rho_ge) \
                       - self.rho_ge_c.dot(np.diag(self.dipole(self.Xrange)))

            mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e \
                = self.propagate_mu_rho(mu_rho_g, mu_rho_ge, mu_rho_ge_c, mu_rho_e, tau_index)
            self.single_step_propagation(tau_index*self.dt)

            self.dJ_dE[tau_index] = 1.j*(self.gibbs_state_A*mu_rho_e.T).sum()

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

    def J_T(self):
        self.set_initial_rho(self.gibbs_state.real)
        for t_indx in range(self.Tsteps):
            self.single_step_propagation(self.dt*t_indx)

        return (self.gibbs_state_A*self.rho_e.real.T).sum()

if __name__ == "__main__":

    from plot_functions import animate_1d_subplots, animate_2d_imshow, plot_2d_subplots
    import matplotlib.pyplot as plt
    import pickle
    from scipy.integrate import ode

    qsys_params = dict(
        t=0.,
        dt=0.005,
        ds=0.05,
        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=3000,
        field_sigma2=2 * 2. ** 2,
        field_freq_num=5,
        field_freq_min=8.,
        field_freq_max=12.,
        gamma=0.0001,

        # kinetic energy part of the hamiltonian
        codeK="0.5*p**2",

        # potential energy part of the hamiltonian
        codeVg="0.5*(.3*q)**2",
        codeVe="0.5*(.3*(q-self.X_amplitude*0.1))**2 + 10.",
        codeDipole="0.05*q",
        codeVge="-.05*q*self.field(t)",
    )

    molecule = OptimalFieldRho(**qsys_params)

    t = np.linspace(0.0, molecule.dt*molecule.Tsteps, molecule.Tsteps)

    plt.figure()
    plt.plot(t, molecule.field(t))
    plt.show()
    # molecule.calculate_dJ_dE()
    y0, t0 = np.concatenate((molecule.A_params, molecule.phi_params)), 0.


    def f(t, y):
        # print t
        result = molecule.calculate_dJ_dE()
        return result

    r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
    # r = ode(f).set_integrator('dop853')
    r.set_initial_value(y0, t0)
    s_max = 100
    s_run = [0.0]
    J = [0.0]
    count = 0
    while r.successful() and count < s_max:
        r.integrate(r.t + molecule.ds)
        molecule.A_params = r.y[:molecule.field_freq_num]
        molecule.phi_params = r.y[molecule.field_freq_num:]
        J.append(molecule.J_T())
        s_run.append(s_run[-1] + molecule.ds)
        print "target state population", J[-1], s_run[-1], r.t
        print
        count += 1

    plt.figure()
    plt.plot(molecule.dJ_dE, 'k')
    plt.ylabel("$\\frac{\\delta J}{\\delta \\epsilon (t)}$")
    plt.xlabel("$t$")
    plt.grid()

    plt.figure()
    plt.plot(J[1:], 'k')
    plt.xlabel("Iterations")
    plt.ylabel("$J$ (cost function)")
    plt.show()
