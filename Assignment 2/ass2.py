import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10,10,1) #s
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
N_vals = [1, 2, 10, 20]
t = np.arange(0, 0.015, 0.0001)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, N in enumerate(N_vals):
    y_t = np.zeros_like(t, dtype=complex)
    
    # Sum from -N to N
    for i in range(-N, N + 1):
        if i != 0:
            j = np.where(n == i)[0][0]
            y_t += Yn[j] * np.exp(1j * i * w0 * t)
    
    y_t = np.real(y_t) / (2 * np.pi)
    
    ax = axes[idx // 2, idx % 2]
    ax.plot(t * 1000, y_t, 'b-', linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('y(t)')
    ax.set_title(f'N = {N}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




