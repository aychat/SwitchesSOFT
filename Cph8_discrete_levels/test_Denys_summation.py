import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

X = x[:, np.newaxis, np.newaxis]
Y = x[np.newaxis, :, np.newaxis]
Z = x[np.newaxis, np.newaxis, :]

Fx = np.exp(-X**2 -0.5 * Y**2 - 0.02*Z**2).sum(axis=(1,2))
Fx_ = ne.evaluate(
    "sum(exp(-X**2 -0.5 * Y**2 - 0.02*Z**2), axis=2)"
).sum(axis=1)

assert np.allclose(Fx, Fx_)


Fy = np.exp(-X**2 -0.5 * Y**2 - 0.02*Z**2).sum(axis=(0,2))
Fz = np.exp(-X**2 -0.5 * Y**2 - 0.02*Z**2).sum(axis=(0,1))

plt.plot(X.reshape(-1), Fx, label="X")
plt.plot(Y.reshape(-1), Fy, label="Y")
plt.plot(Z.reshape(-1), Fz, label="Z")
plt.legend()
plt.show()