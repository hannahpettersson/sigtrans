import numpy as np
import matplotlib.pyplot as plt
from scipy import signal 

#eîx = cosx+isinx
#t0 = 5ms
#w0 = 2pi/t0 = 1256 rad/s
t0 = 0.005 #s
f0 = 1 / t0
omega0 = 2*np.pi*f0 #fundamental anguar freq
n = np.arange(-10,10)
n = n[n != 0]
t = np.linspace(-t0/2, t0/2, 1000) #period interval
#cn = ((-1) **(n+1)) / (1j * n * omega0) #when integrating 
h = 0.001
x = t.copy() #x(t)


def integral (x, t, n, t0):
    cn = []
    for ni in n:
        i = x * np.exp(-1j * ni * omega0 * t)
        i_t = (1/t0) * np.sum(i) * h
        cn.append(i_t)
    return np.array(cn)

cn = integral(x,t,n,t0)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.stem(n, np.abs(cn))
plt.title('Magnitude of cn')
plt.xlabel('n')
plt.ylabel('|cn|')

plt.subplot(1,2,2)
plt.stem(n, np.angle(cn))
plt.title('Phase of cn')
plt.xlabel('n')
plt.ylabel('∠cn [rad]')

plt.tight_layout()
plt.show()


