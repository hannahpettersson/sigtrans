import numpy as np
import matplotlib.pyplot as plt

#coefficient vector with odd numbers from -15 ms to 15ms
# n = np.arange(-10,10,1) #s
# n = n[n != 0]
# t0 = 0.005 #s
# f0 = 1 / t0
# w0 = 2*np.pi*f0 #fundamental anguar freq
# w0_max = 15000
# wn = n*w0
# c_n = np.zeros_like(n, dtype=complex)
# nonzero = n != 0
# c_n[nonzero] = 1j * ((-1) ** np.floor(np.abs(n[nonzero]))) / np.abs(n[nonzero])
# Xn = 2 * np.pi * c_n

# fig, ax = plt.subplots(2,1)
# ax[0].stem(wn,np.abs(Xn))
# ax[1].stem(wn, np.angle(Xn))
# ax[0].set_xlabel('$ \omega$ / rad / s ')
# ax[1].set_xlabel(' $ \omega$ / rad / s ')
# ax[0].set_ylabel(r'$|X(\omega)|$')
# ax[1].set_ylabel(r'$\angle X(\omega)$ [rad]')
# ax[0].set_title(r'Magnitude Spectrum $|X(\omega)|$')
# ax[1].set_title(r'Phase Spectrum $\angle X(\omega)$')
# ax[1].grid()
# ax[1].set_xlim((-15000,15000))

# plt.show()

#TASK 2
dw = 100 #s, men osäker om detta!!!!!!!!!
w = np.arange(-15e3, 15e3, dw)
alpha = 1000 * np.pi

frequency_response = (alpha ** 2) / ((alpha + (1j * w)) ** 2)

# Plot magnitude and phase
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Magnitude plot
ax1.plot(w, np.abs(frequency_response))
ax1.set_xlabel('Frequency ω (rad/s)')
ax1.set_ylabel('|H(ω)|')
ax1.set_title('Magnitude Response')
ax1.grid(True)

# Phase plot
ax2.plot(w, np.angle(frequency_response))
ax2.set_xlabel('Frequency ω (rad/s)')
ax2.set_ylabel('∠H(ω) (radians)')
ax2.set_title('Phase Response')
ax2.grid(True)