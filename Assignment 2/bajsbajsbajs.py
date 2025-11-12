import numpy as np
import matplotlib.pyplot as plt

n = np.arange(1, 21, 1)
n = n[n != 0]
t0 = 0.005 #s
f0 = 1 / t0
w0 = 2*np.pi*f0 #fundamental anguar freq
w0_max = 15000
wn = n*w0
dw = 100 #s
w = np.arange(-15e3, 15e3, dw)
c_n = np.zeros_like(n, dtype=complex)
nonzero = n != 0
c_n[nonzero] = 1j * ((-1) ** np.floor(np.abs(n[nonzero]))) / np.abs(n[nonzero])
Xn = 2 * np.pi * c_n
alpha = 1000 * np.pi
frequency_response = (alpha ** 2) / ((alpha + (1j * wn)) ** 2)
Yn = frequency_response * Xn
dt = 1e-6
t = np.arange(0,0.015, dt)

exp = np.exp(1j * np.outer(t, wn))

y_t = (Yn@exp.T) / (2 * np.pi)

N_vals = [1,2,10,20] #(1,2,10,20)

fig, ax = plt.subplots()

for i, N in enumerate(N_vals):
    y_t_N = Yn[:N] @ exp[:, :N].T
    ax.plot(t, np.real(y_t_N), label=f"y(t), N = {N}")

ax.set_title('Output signal y(t) for different harmonic counts')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude')
ax.legend()
ax.grid(True)
plt.show()

