import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(10, 20, 'ro')
ax.text(8, 22, '$\omega_{M_2} + i\Gamma$', fontsize=12)
ax.plot(10, -20, 'k*')
ax.text(8, -24, '$\omega_{M_2} + i\Gamma$', fontsize=12)

ax.plot(25, 20, 'ro')
ax.text(25, 22, '$\omega_{M_3} + i\Gamma$', fontsize=12)
ax.plot(25, -20, 'k*')
ax.text(25, -24, '$\omega_{M_3} + i\Gamma$', fontsize=12)

ax.set_aspect('equal')

ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
plt.xlim(-20., 40.)
plt.ylim(-40., 40.)
plt.xlabel("Real axis")
plt.ylabel("Imaginary axis")
plt.show()