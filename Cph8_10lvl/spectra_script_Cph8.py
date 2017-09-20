import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from subprocess import call
import os


def A_matrix(k):
    F_mk = []
    F_jn = []
    for m in range(2*n_lvl):
        if m in range(n_lvl) and k in range(n_lvl, 2*n_lvl):
            F_mk.append(FC[m][k-n_lvl])
        elif m in range(n_lvl, 2*n_lvl) and k in range(n_lvl):
            F_mk.append(FC[m-n_lvl][k])
        elif m == k:
            F_mk.append(1.0)
        else:
            F_mk.append(0.0)

    for n in range(2*n_lvl):
        if k-1 in range(n_lvl) and n in range(n_lvl, 2*n_lvl):
            F_jn.append(FC[k-1][n-n_lvl])
        elif k-1 in range(n_lvl, 2*n_lvl) and n in range(n_lvl):
            F_jn.append(FC[k-1-n_lvl][n])
        elif n == k-1:
            F_jn.append(1.0)
        else:
            F_jn.append(0.0)

    F = np.outer(np.asarray(F_mk), np.asarray(F_jn))
    # F = np.abs(F)
    # F = F.T

    return F


n_lvl = 4

for t in range(10):
    FC = np.empty([n_lvl, n_lvl])
    for i in range(n_lvl):
        for j in range(n_lvl):
            if i<j:
                FC[i][j] = random.uniform(0, 1)

    np.set_printoptions(precision=4)

    with open("/home/ayanc/PycharmProjects/Switches/Cph8_10lvl/A_matrix.txt", "w") as f:
        for k in range(1, 2*n_lvl)[::-1]:
            print "Matrix A" + str(k) + str(k-1)
            print A_matrix(k)
            print "\n"

            A_matrix(k).tofile(f, sep=" ", format="%3.6lf")
            f.write(" ")

    os.system("gcc -O3 -Wall $(gsl-config --cflags) Cph8_absorption_spectra.c $(gsl-config --libs)")
    os.system("./a.out 0.0875")

    data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
    data_field = np.asarray(np.loadtxt("field_ini.txt"))
    print data_field
    plt.figure()
    plt.plot(data_field, 'r')
    # plt.show()
    lamb = np.array(data[:, 0])
    Pr_abs = np.array(data[:, 1])
    Pr_ems = np.array(data[:, 3])
    Pfr_abs = np.array(data[:, 2])
    Pfr_ems = np.array(data[:, 4])

    Pr_abs /= Pr_abs.max()
    Pfr_abs /= Pfr_abs.max()
    Pr_ems /= Pr_ems.max()
    Pfr_ems /= Pfr_ems.max()

    mixed_abs = Pr_abs + Pfr_abs

    mixed_ems = Pr_ems + Pfr_ems

    freq = 2. * np.pi * 3e2 / lamb
    Pr_abs_freq = Pr_abs * 2. * np.pi * 3e2 / (freq * freq)
    Pfr_abs_freq = Pfr_abs * 2. * np.pi * 3e2 / (freq * freq)

    freq = freq[::-1]
    Pr_abs_freq = Pr_abs_freq[::-1]
    Pfr_abs_freq = Pfr_abs_freq[::-1]

    data = np.loadtxt("pop_dynamical.out")

    freq1 = data[:, 0]
    PR = data[:, 1]
    # PFR = data[:, 2]

    # PR /= PR.max()

    # PFR /= PFR.max()
    Pr_abs_freq /= Pr_abs_freq.max()
    # Pfr_abs_freq /= Pfr_abs_freq.max()

    plt.figure()

    plt.title('Absorption spectra from model')
    plt.plot(1240./(freq1*1.87855), PR, 'r', label='PR')
    plt.legend()
    plt.show()