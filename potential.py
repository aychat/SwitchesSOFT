import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.misc import factorial
from scipy.integrate import simps

x = np.linspace(-10., 10., 256)


plt.figure()
plt.plot(x, 0.5*(1.+np.tanh(-0.7*x)))
plt.plot(x, 1. + 0.5*(1.+np.tanh(-0.7*x)))
plt.show()