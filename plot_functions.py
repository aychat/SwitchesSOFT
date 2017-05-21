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
