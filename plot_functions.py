import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_1d_subplots(system):
    
    fig = plt.figure()

    ax1 = plt.subplot(221)
    plt.ylim(0.0, 0.1)
    ax2 = plt.subplot(222)
    plt.ylim(-1.0, 1.0)
    ax3 = plt.subplot(223)
    plt.ylim(-1.0, 1.0)
    ax4 = plt.subplot(224)
    plt.ylim(0.0, 0.1)

    line1, = ax1.plot(system.X, np.diag(system.rho_g.real))
    line2, = ax2.plot(system.X, np.diag(system.rho_ge.real))
    line3, = ax3.plot(system.X, np.diag(system.rho_ge.imag))
    line4, = ax4.plot(system.X, np.diag(system.rho_e.real))

    def animate(i):
        for _ in range(30):
            system.single_step_propagation()
            system.t += system.dt
        line1.set_ydata(np.diag(system.rho_g.real))  # update the data
        line2.set_ydata(np.diag(system.rho_ge.real))  # update the data
        line3.set_ydata(np.diag(system.rho_ge.imag))  # update the data
        line4.set_ydata(np.diag(system.rho_e.real))  # update the data
        return line1, line2, line3, line4

    # Init only required for blitting to give a clean slate.
    def init():
        line1.set_ydata(np.ma.array(system.X, mask=True))
        line2.set_ydata(np.ma.array(system.X, mask=True))
        line3.set_ydata(np.ma.array(system.X, mask=True))
        line4.set_ydata(np.ma.array(system.X, mask=True))
        return line1, line2, line3, line4

    ani = FuncAnimation(fig, animate, np.arange(1, 50), init_func=init, interval=5, blit=True)

    plt.show()


def animate_2d_imshow(system, F1, F2):
    class VisualizeDynamics():

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

            from rho_normalize import RhoNormalize

            # bundle plotting settings
            imshow_settings = dict(
                origin='lower',
                cmap='seismic',
                norm=RhoNormalize(vmin=-0.1, vmax=0.1),
                extent=[self.molecule.X1.min(), self.molecule.X1.max(), self.molecule.X2.min(), self.molecule.X2.max()]
            )

            # generate plots
            ax = fig.add_subplot(221)
            ax.set_title('$W_{g}(x,p)$')
            self.rho_g_img = ax.imshow([[0]], **imshow_settings)
            ax.set_ylabel('$p$ (a.u.)')

            ax = fig.add_subplot(222)
            ax.set_title('$\\Re W_{ge}(x,p)$')
            self.re_rho_ge_img = ax.imshow([[0]], **imshow_settings)
            ax.set_ylabel('$p$ (a.u.)')

            ax = fig.add_subplot(223)
            ax.set_title('$\\Im W_{eg}(x,p)$')
            self.im_rho_ge_img = ax.imshow([[0]], **imshow_settings)
            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            ax = fig.add_subplot(224)
            ax.set_title('$W_{e}(x,p)$')
            self.rho_e_img = ax.imshow([[0]], **imshow_settings)
            ax.set_xlabel('$x$ (a.u.)')
            ax.set_ylabel('$p$ (a.u.)')

            # self.fig.colorbar(self.img)

        def set_quantum_sys(self):
            """
            Initialize quantum propagator
            :param self:
            :return:
            """
            self.molecule = system

        def empty_frame(self):
            """
            Make empty frame and reinitialize quantum system
            :param self:
            :return: image object
            """
            self.set_quantum_sys()

            self.rho_g_img.set_array([[0]])
            self.re_rho_ge_img.set_array([[0]])
            self.im_rho_ge_img.set_array([[0]])
            self.rho_e_img.set_array([[0]])

            return self.rho_g_img, self.re_rho_ge_img, self.im_rho_ge_img, self.rho_e_img

        def __call__(self, frame_num):
            """
            Draw a new frame
            :param frame_num: current frame number
            :return: image objects
            """
            for _ in xrange(F1):
                self.molecule.single_step_propagation()
                self.molecule.t += self.molecule.dt

            self.rho_g_img.set_array(self.molecule.rho_g.real)
            self.re_rho_ge_img.set_array(self.molecule.rho_ge.real)
            self.im_rho_ge_img.set_array(self.molecule.rho_ge.imag)
            self.rho_e_img.set_array(self.molecule.rho_e.real)

            return self.rho_g_img, self.re_rho_ge_img, self.im_rho_ge_img, self.rho_e_img

    fig = plt.gcf()
    visualizer = VisualizeDynamics(fig)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(F2),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)
    plt.show()
