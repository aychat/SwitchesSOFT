import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("pop_dynamical_python.out")

data = data.T
print data.shape
t = np.linspace(0.0, 100.0, 100000)
t_ = np.linspace(0.0, 250., 250)

field_ini = np.loadtxt("field_ini_python.out")
field_fin = np.loadtxt("field_fin_python.out")

pop_1 = data[0, :]
pop_2 = data[1, :]
pop_3 = data[2, :]
pop_4 = data[3, :]
pop_5 = data[4, :]

pop_6 = data[5, :]
pop_7 = data[6, :]
pop_8 = data[7, :]
pop_9 = data[8, :]
pop_10 = data[9, :]

fig2 = plt.figure()
ax1 = fig2.add_subplot(111)
ax1.title.set_text("Optimal field versus initial field")
ax1.plot(t_, field_ini, 'r', linewidth=1.5, label='Initial Field')
ax1.plot(t_, field_fin, 'k', linewidth=1.5, label='Final Field')
ax1.set_xlabel("Time(in fs)")
ax1.set_ylabel("Field Strength in a.u. (5.14x10$^{11}$ Vm$^{-1}$)")
ax1.legend()

fig = plt.figure()
ax11 = fig.add_subplot(221)
ax11.title.set_text("Fast dynamics of PR state")
ax11.plot(t[:500], pop_1[:500], linewidth=1.5, label='R1')
ax11.plot(t[:500], pop_2[:500], linewidth=1.5, label='R2')
ax11.plot(t[:500], pop_3[:500], linewidth=1.5, label='R3')
ax11.plot(t[:500], pop_4[:500], linewidth=1.5, label='R4')
ax11.plot(t[:500], pop_5[:500], linewidth=1.5, label='R5')
ax11.set_xlim(t[0], t[499])
ax11.set_xlabel("time (in ps)")
ax11.set_ylabel("Population of PR levels")
ax11.legend()

ax22 = fig.add_subplot(223)
ax22.title.set_text("Fast dynamics of PFR state")
ax22.plot(t[:500], pop_6[:500], linewidth=1.5, label='FR1')
ax22.plot(t[:500], pop_7[:500], linewidth=1.5, label='FR2')
ax22.plot(t[:500], pop_8[:500], linewidth=1.5, label='FR3')
ax22.plot(t[:500], pop_9[:500], linewidth=1.5, label='FR4')
ax22.plot(t[:500], pop_10[:500], linewidth=1.5, label='FR5')
ax22.set_xlim(t[0], t[499])
ax22.set_xlabel("time (in ps)")
ax22.set_ylabel("Population of PFR levels")
ax22.legend()

ax33 = fig.add_subplot(222)
ax33.title.set_text("Ultrafast dynamics of PR state")
ax33.plot(t, pop_1, linewidth=1.5, label='R1')
ax33.plot(t, pop_2, linewidth=1.5, label='R2')
ax33.plot(t, pop_3, linewidth=1.5, label='R3')
ax33.plot(t, pop_4, linewidth=1.5, label='R4')
ax33.plot(t, pop_5, linewidth=1.5, label='R5')
ax33.set_xlim(t[0], t[-1])
ax33.set_xlabel("time (in ps)")
ax33.set_ylabel("Population of PR levels")
ax33.legend()

ax44 = fig.add_subplot(224)
ax44.title.set_text("Ultrafast dynamics of PFR state")
ax44.plot(t, pop_6, linewidth=1.5, label='FR1')
ax44.plot(t, pop_7, linewidth=1.5, label='FR2')
ax44.plot(t, pop_8, linewidth=1.5, label='FR3')
ax44.plot(t, pop_9, linewidth=1.5, label='FR4')
ax44.plot(t, pop_10, linewidth=1.5, label='FR5')
ax44.set_xlim(t[0], t[-1])
ax44.set_ylim(-0.01, 0.51)
ax44.set_xlabel("time (in ps)")
ax44.set_ylabel("Population of PFR levels")
ax44.legend()

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.title.set_text("Optimal field versus initial field")
ax1.plot(t_, field_ini, 'r', linewidth=1.5, label='Initial Field')
ax1.plot(t_, field_fin, 'k', linewidth=1.5, label='Final Field')
ax1.set_xlabel("Time(in fs)")
ax1.set_ylabel("Field Strength in a.u. (5.14x10$^{11}$ Vm$^{-1}$)")
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.title.set_text("Total PR/PFR state population dynamics - fast")
ax2.plot(t[:500], pop_5[:500] + pop_6[:500], linewidth=1.5, label='total PFR')
ax2.plot(t[:500], pop_1[:500] + pop_10[:500], linewidth=1.5, label='total PR')
ax2.set_xlim(t[0], 0.5)
ax2.set_xlabel("Time(in ps)")
ax2.set_ylabel("Total PR population")
ax2.legend(loc=7)

ax3 = fig.add_subplot(224)
ax3.title.set_text("Total PR/PFR state population dynamics - ultrafast")
ax3.plot(t, pop_5 + pop_6, linewidth=1.5, label='total PFR')
ax3.plot(t, pop_1 + pop_10, linewidth=1.5, label='total PR')
ax3.set_xlim(t[0], t[-1])
ax3.set_xlabel("Time(in ps)")
ax3.set_ylabel("Total PR population")
ax3.legend(loc=7)

print pop_1[-1]
print pop_2[-1]
print pop_3[-1]
print pop_4[-1]
print pop_5[-1]
print pop_6[-1]
print pop_7[-1]
print pop_8[-1]
print pop_9[-1]
print pop_10[-1]

plt.show()
