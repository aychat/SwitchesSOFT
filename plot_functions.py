import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from rho_normalize import RhoNormalize


def plot_2d_subplots(system, rhoG, rhoGE, rhoE):
    fig = plt.figure()

    imshow_settings = dict(
        origin='lower',
        cmap='seismic',
        norm=RhoNormalize(vmin=-0.1, vmax=0.1),
        extent=[system.X1.min(), system.X1.max(), system.X2.min(), system.X2.max()]
    )

    # generate plots
    ax = fig.add_subplot(221)
    ax.set_title('$\\rho_{g}(x,p)$')
    ax.imshow(rhoG.real, **imshow_settings)
    ax.set_xlabel('$x$ (a.u.)')
    ax.set_ylabel('$x^{\'}$ (a.u.)')

    ax = fig.add_subplot(222)
    ax.set_title('$\\Re \\rho_{ge}(x,p)$')
    ax.imshow(rhoGE.real, **imshow_settings)
    ax.set_xlabel('$x$ (a.u.)')
    ax.set_ylabel('$x^{\'}$ (a.u.)')

    ax = fig.add_subplot(223)
    ax.set_title('$\\Im \\rho_{eg}(x,p)$')
    ax.imshow(rhoGE.imag, **imshow_settings)
    ax.set_xlabel('$x$ (a.u.)')
    ax.set_ylabel('$x^{\'}$ (a.u.)')

    ax = fig.add_subplot(224)
    ax.set_title('$\\rho_{e}(x,p)$')
    ax.imshow(rhoE.real, **imshow_settings)
    ax.set_xlabel('$x$ (a.u.)')
    ax.set_ylabel('$x^{\'}$ (a.u.)')

    plt.grid()
    plt.show()
    
    
def animate_1d_subplots(system, rhoG, rhoGE, rhoE):
    
    fig = plt.figure()

    ax1 = plt.subplot(221)
    plt.ylim(-0.1, 0.1)
    ax2 = plt.subplot(222)
    plt.ylim(-1.0, 1.0)
    ax3 = plt.subplot(223)
    plt.ylim(-1.0, 1.0)
    ax4 = plt.subplot(224)
    plt.ylim(-0.1, 0.1)

    line1, = ax1.plot(system.Xrange, np.diag(rhoG.real))
    line2, = ax2.plot(system.Xrange, np.diag(rhoGE.real))
    line3, = ax3.plot(system.Xrange, np.diag(rhoGE.imag))
    line4, = ax4.plot(system.Xrange, np.diag(rhoE.real))

    def animate(i):
        for _ in range(30):
            system.single_step_propagation(system.t)
            system.t += system.dt
        line1.set_ydata(np.diag(rhoG.real))  # update the data
        line2.set_ydata(np.diag(rhoGE.real))  # update the data
        line3.set_ydata(np.diag(rhoGE.imag))  # update the data
        line4.set_ydata(np.diag(rhoE.real))  # update the data
        return line1, line2, line3, line4

    # Init only required for blitting to give a clean slate.
    def init():
        line1.set_ydata(np.ma.array(system.Xrange, mask=True))
        line2.set_ydata(np.ma.array(system.Xrange, mask=True))
        line3.set_ydata(np.ma.array(system.Xrange, mask=True))
        line4.set_ydata(np.ma.array(system.Xrange, mask=True))
        return line1, line2, line3, line4

    ani = FuncAnimation(fig, animate, np.arange(1, 50), init_func=init, interval=5, blit=True)

    plt.show()


def animate_2d_imshow(system, F1, F2, rhoG, rhoGE, rhoE):

    fig = plt.gcf()
    visualizer = VisualizeDynamics(fig, system, F1, F2, rhoG, rhoGE, rhoE)
    animation = FuncAnimation(fig, visualizer, frames=np.arange(F2),
                              init_func=visualizer.empty_frame, repeat=True, blit=True)
    plt.show()


class VisualizeDynamics:
    def __init__(self, fig, system, F1, F2, rhoG, rhoGE, rhoE):
        """
        Initialize all propagators and frame
        :param fig: matplotlib figure object
        """

        #################################################################
        #
        # Initialize plotting facility
        #
        #################################################################

        self.fig = fig
        self.F1 = F1
        self.F2 = F2
        self.rhoG = rhoG
        self.rhoGE = rhoGE
        self.rhoE = rhoE
        self.system = system

        from rho_normalize import RhoNormalize

        # bundle plotting settings
        imshow_settings = dict(
            origin='lower',
            cmap='seismic',
            norm=RhoNormalize(vmin=-0.1, vmax=0.1),
            extent=[self.system.X1.min(), self.system.X1.max(), self.system.X2.min(), self.system.X2.max()]
        )

        # generate plots
        ax = fig.add_subplot(221)
        ax.set_title('$\\rho_{g}(x,p)$')
        self.rho_g_img = ax.imshow([[0]], **imshow_settings)
        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$x^{\'}$ (a.u.)')
        self.fig.colorbar(self.rho_g_img)

        ax = fig.add_subplot(222)
        ax.set_title('$\\Re \\rho_{ge}(x,p)$')
        self.re_rho_ge_img = ax.imshow([[0]], **imshow_settings)
        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$x^{\'}$ (a.u.)')
        self.fig.colorbar(self.re_rho_ge_img)

        ax = fig.add_subplot(223)
        ax.set_title('$\\Im \\rho_{eg}(x,p)$')
        self.im_rho_ge_img = ax.imshow([[0]], **imshow_settings)
        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$x^{\'}$ (a.u.)')
        self.fig.colorbar(self.im_rho_ge_img)

        ax = fig.add_subplot(224)
        ax.set_title('$\\rho_{e}(x,p)$')
        self.rho_e_img = ax.imshow([[0]], **imshow_settings)
        ax.set_xlabel('$x$ (a.u.)')
        ax.set_ylabel('$x^{\'}$ (a.u.)')
        self.fig.colorbar(self.rho_e_img)

    def empty_frame(self):
        """
        Make empty frame and reinitialize quantum system
        :param self:
        :return: image object
        """

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

        for _ in xrange(self.F1):
            self.system.single_step_propagation(self.system.t)
            self.system.t += self.system.dt

        self.rho_g_img.set_array(self.rhoG.real)
        self.re_rho_ge_img.set_array(self.rhoGE.real)
        self.im_rho_ge_img.set_array(self.rhoGE.imag)
        self.rho_e_img.set_array(self.rhoE.real)

        return self.rho_g_img, self.re_rho_ge_img, self.im_rho_ge_img, self.rho_e_img
