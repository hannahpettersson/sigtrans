import numpy as np
import matplotlib.pyplot as plt

N = [1,2,10,20]
t = np.array([0, 0.015])  # s
t0 = t[1] - t[0]
f0 = 1/t0

w0 = 2*np.pi*f0 #fundamental anguar freq
w0_max = 15000

start, stop = t
increment = 0.001 # tillfälligt
t_vec = np.linspace(start, stop, 1000)

n = np.arange(-20,20,1) #s
n = n[n != 0]

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

import numpy as np

def inverse_fourier_series(Yn, n, w0, t):
    t = np.atleast_1d(t)
    exp_term = np.exp(1j * np.outer(n, w0 * t))
    y_t = (1 / (2 * np.pi)) * np.sum(Yn[:, None] * exp_term, axis=0)
    return y_t

hej = np.linspace(0, 0.01, 1000)

# Reconstruct y(t)
y_t = inverse_fourier_series(Yn, n, w0, hej)

print(y_t)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
N_vals = [1, 2, 10, 20]
plt.tight_layout()
plt.show()