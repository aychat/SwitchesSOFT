import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import interpolate
import pickle
import os
import datetime


with open("spectra_data.pickle", "rb") as f:
    data = pickle.load(f)

result = np.asarray(data["result"])
result_rows, result_col = result.shape

wavelength = result[1, :][::-1]
spectra = result[0, :][::-1]

func = interpolate.interp1d(wavelength, spectra, kind='cubic')
wavelength_new = np.arange(500., 875., 0.1)
spectra_new = func(wavelength_new)

# plt.plot(wavelength, spectra)
plt.plot(wavelength_new, spectra_new)
plt.grid()
plt.show()


print result_rows, result_col
# my_path = os.path.abspath("/home/ayanc/PycharmProjects/Switches/Plots_3")
#
# grid = np.asarray(data["grid"])
# result = np.asarray(data["result"])
#
# grid_rows, grid_cols = grid.shape
# result_rows, result_col, result_width = result.shape
#
# print grid
# print grid.shape
# assert grid_rows == result_rows, "Dimensions of result and grid do not match"
#
#
# for i in range(grid_rows):
#     with PdfPages('Spectra_Raman' + str(i+1) + '.pdf') as pdf:
#         displacement = grid[i][1]
#         potential = grid[i][0]
#         spectra = result[i][0]
#         wavelength = result[i][1]
#         spectra = spectra[::-1]
#         wavelength = wavelength[::-1]
#         func = interpolate.pchip(wavelength, spectra)
#         wavelength_new = np.arange(500., 25000., 0.1)
#         spectra_new = func(wavelength_new)
#         filename = "/spectra_" + str(i)
#         fig = plt.figure()
#         fig.suptitle("Oscillator frequency = %3.2lf \n Displacement = %3.2lf" %(potential, displacement))
#         ax = fig.add_subplot(111)
#         ax.plot(wavelength, spectra)
#         # ax.plot(wavelength_new, spectra_new)
#         # fig.savefig(my_path + filename)
#         pdf.savefig(fig)
#         plt.show()