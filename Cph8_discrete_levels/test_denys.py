import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)

#
# for num, (alpha, title) in enumerate(zip(np.linspace(1, 2, 3), ["ABC", "ABC1", "ABC3"])):
#     plt.subplot(1, 3, num+1)
#     plt.title(title)
#     plt.plot(x, np.exp(-alpha * x**2), label=("%.3f" % alpha))


titles = ["ABC", "ABC1", "ABC3"]

for num, alpha in enumerate(np.linspace(1, 2, 12)):
    plt.subplot(3, 4, num+1)
    # plt.title(titles[num])
    plt.plot(x, np.exp(-alpha * x**2), label=("%.3f" % alpha))

plt.show()