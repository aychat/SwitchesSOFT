import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("FP.dat", delimiter=',')
lamb = np.array(data[:, 0])
CFP = np.array(data[:, 1])
YFP = np.array(data[:, 2])
GFP = np.array(data[:, 3])

plt.figure()
plt.plot(lamb, CFP, 'c', linewidth=2., label='CFP')
plt.plot(lamb, YFP, linewidth=3., color='#F3F315', label='YFP')
plt.plot(lamb, GFP, 'g', linewidth=2., label='GFP')
plt.legend()
plt.grid()
plt.xlabel("Wavelength (in nm)")
plt.ylabel("Normalized absorption")
plt.show()