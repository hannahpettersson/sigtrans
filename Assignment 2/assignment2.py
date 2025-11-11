import numpy as np
import matplotlib.pyplot as plt

#coefficient vector with odd numbers from -15 ms to 15ms
n = np.arange(-10,10,1) #s
w0 = 15000
wn = n*w0
c_n = (1j * (-1)**np.floor(np.abs(n))/np.abs(n)*wn)

fig, ax = plt.subplots(2,1)
ax[0].stem(wn,np.abs(c_n))
ax[1].stem(wn, np.angle[c_n])
ax[1].set_xlabel('$n$')
ax[1].set_ylabel('$c_n$')
ax[1].grid()
ax[1].set_xlim((-15.5,15.5))