import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('data_pol2.pickle', 'rb') as file_out:
    data = pickle.load(file_out)

omega = data['omega']
pol2_mol1 = data['pol2_mol1']*1e-23
pol2_mol2 = data['pol2_mol2']*1e-23
pol2_mol3 = data['pol2_mol3']*1e-23

pol_mat1 = np.column_stack((pol2_mol1, pol2_mol2, pol2_mol3))

plt.figure()
plt.suptitle("$P^{(2)}(\\omega)$ for 2 nearly identical molecules and the heterodyne field that selectively detects one w.r.t. the other. The inner products for the cross figures are zero."
             " (Arbitrary units for polarization and electric fields).")
plt.subplot(231)
plt.title("$P^{(2)}(\\omega)$ for Molecule 1")
plt.plot(omega, pol_mat1[:, 0], 'r')
plt.xlabel("freq (in GHz)")
plt.subplot(232)
plt.title("$P^{(2)}(\\omega)$ for Molecule 2")
plt.plot(omega, pol_mat1[:, 1], 'r')
plt.xlabel("freq (in GHz)")
plt.subplot(233)
plt.title("$P^{(2)}(\\omega)$ for Molecule 3")
plt.plot(omega, pol_mat1[:, 2], 'r')
plt.xlabel("freq (in GHz)")

Q1 = np.linalg.qr(pol_mat1)[0]
R1 = np.linalg.qr(pol_mat1)[1]

print pol2_mol1.dot(Q1[:, 0])
print pol2_mol1.dot(Q1[:, 1])
print pol2_mol1.dot(Q1[:, 2])

pol_mat2 = np.column_stack((pol2_mol2, pol2_mol1, pol2_mol3))


Q2 = np.linalg.qr(pol_mat2)[0]
R2 = np.linalg.qr(pol_mat2)[1]

print pol2_mol2.dot(Q2[:, 0])
print pol2_mol2.dot(Q2[:, 1])
print pol2_mol2.dot(Q2[:, 2])

pol_mat3 = np.column_stack((pol2_mol3, pol2_mol1, pol2_mol2))


Q3 = np.linalg.qr(pol_mat3)[0]
R3 = np.linalg.qr(pol_mat3)[1]

print pol2_mol3.dot(Q3[:, 0])
print pol2_mol3.dot(Q3[:, 1])
print pol2_mol3.dot(Q3[:, 2])

plt.subplot(234)
plt.title("$E_{Heterodyne}(\\omega)$ for Molecule 1")
plt.plot(omega, Q1[:, 1], 'k')
plt.xlabel("freq (in GHz)")
plt.subplot(235)
plt.title("$E_{Heterodyne}(\\omega)$ for Molecule 2")
plt.plot(omega, Q2[:, 1], 'k')
plt.xlabel("freq (in GHz)")
plt.subplot(236)
plt.title("$E_{Heterodyne}(\\omega)$ for Molecule 3")
plt.plot(omega, Q3[:, 1], 'k')
plt.xlabel("freq (in GHz)")
plt.show()

