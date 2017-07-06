import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("kinetic_dyn.pickle", "rb") as f:
    data = pickle.load(f)

print data
t = np.linspace(0.0, 100.0, 1000)

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