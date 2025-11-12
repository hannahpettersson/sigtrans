import numpy as np
import matplotlib.pyplot as plt

######## TASK 1b ########
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

######## TASK 2b ########
# dw = 100 #s, men osäker om detta!!!!!!!!!
# w = np.arange(-15e3, 15e3, dw)
# alpha = 1000 * np.pi

# frequency_response = (alpha ** 2) / ((alpha + (1j * w)) ** 2)

# # Plot magnitude and phase
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# # Magnitude plot
# ax1.plot(w, np.abs(frequency_response))
# ax1.set_xlabel('Frequency ω (rad/s)')
# ax1.set_ylabel('|H(ω)|')
# ax1.set_title('Magnitude Response')
# ax1.grid(True)

# # Phase plot
# ax2.plot(w, np.angle(frequency_response))
# ax2.set_xlabel('Frequency ω (rad/s)')
# ax2.set_ylabel('∠H(ω) (radians)')
# ax2.set_title('Phase Response')
# ax2.grid(True)
# plt.show()

######## TASK 3a ########
# n = np.arange(-10,10,1) #s
# n = n[n != 0]
# t0 = 0.005 #s
# f0 = 1 / t0
# w0 = 2*np.pi*f0 #fundamental anguar freq
# w0_max = 15000
# wn = n*w0
# dw = 100 #s
# w = np.arange(-15e3, 15e3, dw)
# c_n = np.zeros_like(n, dtype=complex)
# nonzero = n != 0
# c_n[nonzero] = 1j * ((-1) ** np.floor(np.abs(n[nonzero]))) / np.abs(n[nonzero])
# Xn = 2 * np.pi * c_n
# alpha = 1000 * np.pi
# frequency_response = (alpha ** 2) / ((alpha + (1j * wn)) ** 2)
# Yn = frequency_response * Xn

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# ax1.stem(wn, np.abs(Xn))
# ax1.stem(wn, np.abs(Yn))
# ax1.set_title('magnitude spectrum')
# ax1.set_xlabel(r'$\omega$ [rad/s]')
# ax1.set_ylabel(r'$X|(\omega)|, |Y(\omega)|$')
# ax1.legend()
# ax1.grid(True)

# ax2.stem(wn, np.angle(Xn))
# ax2.stem(wn, np.angle(Yn))
# ax2.set_title('phase spectrum')
# ax2.set_xlabel(r'$\omega$ [rad/s]')
# ax2.set_ylabel(r'Fas [rad]')
# ax2.legend()
# ax2.grid(True)

# plt.show()

T = 1
dt = 0.001
t_interval = np.arange(0, 0.015, dt)
x = 1.0*np.logical_and(t_interval >= -T/2, t_interval < T/2) - 1.0*np.logical_and(t_interval >= T/2, t_interval < 3*T/2)

# Add a second period for illustration
t = np.concatenate((t_interval, t_interval+T))
x = np.concatenate((x, x))

N_vals = np.array([1, 2, 10, 20])
xs = np.zeros((N_vals.shape[0], t_interval.shape[0]))

t0 = 0.005 #s
f0 = 1 / t_interval
w0 = 2*np.pi*f0 #fundamental anguar freq
w0_max = 15000

fig, ax2 = plt.subplots()

for i in range(N_vals.shape[0]):
    n = np.arange(-N_vals[i], N_vals[i] + 2, 2)
    wn = n*w0

    c_n = np.zeros_like(n, dtype=complex)
    nonzero = n != 0
    c_n[nonzero] = 1j * ((-1) ** np.floor(np.abs(n[nonzero]))) / np.abs(n[nonzero])

    w = np.exp(1j * wn * np.outer(t_interval, n))

    x_n = 2 * np.pi * c_n
    alpha = 1000 * np.pi
    frequency_response = (alpha ** 2) / ((alpha + (1j * wn)) ** 2)

    y_n = frequency_response * x_n

    xs[i, :] = np.real(w@y_n)
    
    ax2.plot(t, xs[i, :], label=('$x(t)$ (N = ' + str(N_vals[i]) + ')'))

ax2.legend(bbox_to_anchor=(1, 1), loc='upper left')
ax2.grid()
ax2.set_xlim((-T/2, 3/4*T+T))
plt.show()