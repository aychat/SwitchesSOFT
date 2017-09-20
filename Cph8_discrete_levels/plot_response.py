import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

omega10 = 2.354             # fs-1
omega20 = 4.708+2e-6        # fs-1
gamma10 = 1e-6              # fs-1
gamma20 = 1.5e-6            # fs-1
omegaM1 = 2.354 + 5e-8      # fs-1
omegaM2 = 2.354 + 3e-8      # fs-1
gamma = 1e-9                # fs-1
omega = np.linspace(2.354-2e-6, 2.354+2e-6, 8192)

# omegaM1 = np.linspace(2.354-2e-6, 2.354+2e-6, 128)
# omegaM2 = np.linspace(2.354-2e-6, 2.354+2e-6, 128)
chi1 = 1./(omega10 - omega - 1j*gamma10) + 1./(omega10 + omega + 1j*gamma10)
field = gamma / ((omega - omegaM1)**2 + gamma**2)

plt.figure()
plt.plot(omega, chi1.real, 'r')
plt.plot(omega, chi1.imag, 'k')
plt.plot(omega, field)

plt.figure()
plt.plot(omega, (chi1*field).real, 'g')
plt.plot(omega, (chi1*field).imag, 'b')

# pol2 = np.zeros((omegaM1.size, omegaM2.size), dtype=np.complex)
# for i, M1 in enumerate(omegaM1):
#     for j, M2 in enumerate(omegaM2):
#         A1 = omega - M2 - omega10 + 1j*(gamma + gamma10)
#         A2 = M1 - omega10 + 1j*(gamma + gamma10)
#         B = omega - M1 - M2 + 2j*gamma
#         pol2[i, j] = ((A1+A2)*(B-B.conj())/2. - (B-B.conj())**2)/(A1*A2*B*B.conj())
#
# print pol2.shape
#
# plt.figure()
# plt.plot(omega, pol2.real, 'r')
# plt.plot(omega, pol2.imag, 'k')
# plt.plot(omega, field, 'b')

# fig = plt.figure()
#
# imshow_settings = dict(
#     origin='lower',
#     cmap='seismic',
#     extent=[omegaM1.min(), omegaM1.max(), omegaM2.min(), omegaM2.max()]
# )
#
# # generate plots
# ax = fig.add_subplot(121)
# im1 = ax.imshow(pol2.real, **imshow_settings)
# fig.colorbar(im1)
# # labels = ax.get_xticklabels()
#
# ax = fig.add_subplot(122)
# im2 = ax.imshow(pol2.imag, **imshow_settings)
# fig.colorbar(im2)
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.subplot(121)
# plt.plot(np.diag(pol2.real), 'k')
# plt.subplot(122)
# plt.plot(np.diag(np.fliplr(pol2.real)), 'r')
# plt.show()


omega = 4.708
gamma = 1./10**(np.arange(3, 9))

pol2 = np.zeros(gamma.size, dtype=np.complex)
for i, gamma_i in enumerate(gamma):
    A1 = omega - omegaM2 - omega10 + 1j*(gamma_i + gamma10)
    A2 = omegaM1 - omega10 + 1j*(gamma_i + gamma10)
    B = omega - omegaM1 - omegaM2 + 2j*gamma_i
    B_ = omega - omegaM1 - omegaM2 - 2j*gamma_i
    pol2[i] = ((A1+A2)*(B-B_)/2. - (B-B_)**2)/(A1*A2*B*B_)

plt.figure()
plt.plot(-np.log10(gamma), np.log10(np.abs(pol2)))
plt.xlabel("-$log_{10}(\Gamma)$")
plt.ylabel("$log_{10}(|P^{(2)}(\omega|)$")
plt.show()