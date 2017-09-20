import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-25., 25., 512)

omega = .0775
plt.figure()
plt.plot(x, 0.5*(omega*x)**2)
plt.plot(x, 20.*omega*(1 - np.exp(-np.sqrt(omega/40.)*(x-0.5)))**2)
plt.show()