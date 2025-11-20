# %%
import numpy as np
import sys, os

# physion
sys.path += ['../../src'] # add src code directory for physion

import physion
import matplotlib.pylab as plt


#%%

T = 4        # total duration (seconds)
fs = 1000      # sampling frequency (Hz)
t = np.linspace(0, T, int(T * fs), endpoint=False)

f = 5          # signal frequency (Hz)


A = np.zeros_like(t)

cond = np.sin(2 * np.pi * f * t) > 0.5

A[cond] = 1


plt.plot(t, A)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()