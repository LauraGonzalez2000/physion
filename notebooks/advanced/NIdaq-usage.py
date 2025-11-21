# %%
import sys, time
sys.path += ['../../src']
import physion

import numpy as np
import matplotlib.pylab as plt

import os

def build_start_stop_signal(t):
    cond = ((t>0.1) & (t<0.15)) |\
          ((t>(t[-1]-0.1)) & (t<(t[-1]-0.05)))
    output = np.zeros_like(t)
    output[cond] = 5
    return output

fs = 50e3 

# %%

tstop = 60.
T = 30e-3          # period (seconds)
fs = 50e3       # sampling frequency
dt = 1/fs

t = np.arange(int(tstop/dt)+1)*dt

output = build_start_stop_signal(t)

acq = physion.hardware.NIdaq.main.Acquisition(\
                 sampling_rate=fs,
                 Nchannel_analog_in=2,
                 outputs=np.array([output], dtype=np.float64),
                 max_time=tstop)

# %%
acq.launch()
tic = time.time()
while (time.time()-tic)<tstop:
    pass
acq.close()
# %%
t0 =0.132
cond = (t>t0) & (t<(t0+0.003))
plt.plot(1e3*(t[cond]-t0), acq.analog_data[0][cond], label='start')
plt.plot(1e3*(t[cond]-t0), acq.analog_data[1][cond], label='stop')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend(loc=(1.,0.2))

# %%
# performing multiple recordings
for i in range(1,4):
        
    fs = 50e3
    tstop = 3 # 60*60
    t = np.arange(int(tstop*fs)+1)/fs
    output = build_start_stop_signal(t)
    acq = physion.hardware.NIdaq.main.Acquisition(\
                    sampling_rate=fs,
                    Nchannel_analog_in=2,
                    outputs=np.array([output], dtype=np.float64),
                    filename=os.path.expanduser('~/Desktop/Sample%i.npy' % i),
                    max_time=tstop)
    acq.launch()
    tic = time.time()
    while (time.time()-tic)<tstop:
        pass
    acq.close()
    time.sleep(4)
#######################################################
#######################################################









#%%
#######################################################
#apparition time stop or start signal
#######################################################
data = np.load(\
        os.path.expanduser('~/Desktop/Sample2.npy'),
        allow_pickle=True).item()

#%%  START INTERVALS
fig, ax = plt.subplots(1, figsize=(20,4))

start_signal = data['analog'][0]
t = np.arange(len(data['analog'][0]))*data['dt'] 
fs = 50e3       # sampling frequency   #limit accuracy

# Rising-edge detection
start_indices = np.where((start_signal[:-1] <= 4.99) & (start_signal[1:] > 4.99))[0] + 1
start_times = start_indices / fs

distances = 1000*np.diff(start_times)

print(start_times)
print(distances)

#plt.scatter(start_times[1:], distances, marker='o')
plt.hist(distances-np.mean(distances))#, marker='o')
plt.xlabel("Start instance")
plt.ylabel("Interval to next start (ms)")
plt.title("Start Interval Over Time")
plt.grid(True)
plt.show()

# %%
#  STOP INTERVALS
fig, ax = plt.subplots(1, figsize=(20,4))

stop_signal = data['analog'][1]
t = np.arange(len(data['analog'][1]))*data['dt'] 

# Rising-edge detection
stop_indices = np.flatnonzero((stop_signal[:-1] <= 4.99) & (stop_signal[1:] > 4.99)) + 1
stop_times = stop_indices / fs
distances = 1000*np.diff(stop_times)   # time between stops

print(stop_times)
print(distances)

zoom1 = 0 #1200
zoom2 = -1 #1300

plt.hist(distances)
print(np.mean(distances))

#plt.hist(distances-np.mean(distances))#, marker='o')
#plt.scatter(stop_times[zoom1+1:zoom2], distances[zoom1:zoom2], marker='o')
plt.xlabel("Stop instance")
plt.ylabel("Interval to next stop (ms)")
plt.title("Stop Interval Over Time")
plt.grid(True)
plt.show()


#%%
##############################################################
#interval stop-start
##############################################################
#%% STOP START INTERVALS
fig, ax = plt.subplots(1, figsize=(20,4))

start_signal = data['analog'][0]
stop_signal  = data['analog'][1]
t = np.arange(len(data['analog'][1]))*data['dt'] 

# Rising-edge detection
start_indices = np.where((start_signal[:-1] <= 4.99) & (start_signal[1:] > 4.99))[0] + 1
start_times = start_indices / fs

stop_indices = np.where((stop_signal[:-1] <= 4.99) & (stop_signal[1:] > 4.99))[0] + 1
stop_times = stop_indices / fs

#distances = 1000*(stop_times[:]-start_times[:-1])  # time between stops
distances = 1000*(start_times[1:-1]-stop_times[:-1])
#print(distances)

zoom1 = 0#1200 #0
zoom2 = -1 #1250 #-1
#plt.scatter(np.arange(len(distances))[zoom1:zoom2],distances[zoom1:zoom2], marker='o')
#plt.hist(distances-np.mean(distances))
plt.hist(distances)
plt.xlabel("Duration STOP START interval (s)")
plt.ylabel("instances")
print(np.mean(distances))
#plt.xlabel("STOP START item (s)")
#plt.ylabel("Duration STOP START interval (s)")
plt.title("Duration STOP START Interval Over Time")
plt.grid(True)
plt.show()



#%%
##################################################################
#HOW much can we predict of the future? What's our error 
##################################################################
# %%
fig, ax = plt.subplots(4, figsize=(7,10))
for i in range(1, 3):
        
    data = np.load(\
        os.path.expanduser('~/Desktop/Sample%i.npy' % i),
        allow_pickle=True).item()
    t = np.arange(len(data['analog'][0]))*data['dt']

    print("t", t)
    print("len(t)", len(t))

    t0 =0.500
    t0 = 100
    #t0 =901.1
    #t0 =1798.5
    cond = (t>t0+0.2) & (t<(t0+0.353))

    ax[0].plot(1e3*(t[cond]), data['analog'][0][cond], label='start')
    ax[1].plot(1e3*(t[cond]), data['analog'][1][cond], label='stop')

    start_signal = data['analog'][0]
    stop_signal  = data['analog'][1]

    # Detect rising edges for start
    start_indices = np.where((start_signal[:-1] <= 4.99) &
                            (start_signal[1:]  > 4.99))[0] + 1

    # Detect rising edges for stop
    stop_indices = np.where((stop_signal[:-1] <= 4.99) &
                            (stop_signal[1:]  > 4.99))[0] + 1

    # Create output signal (0 everywhere)
    on_signal = np.zeros_like(start_signal)

    # Set to 5 V between each start–stop pair
    for s, e in zip(stop_indices, start_indices[1:]):
        on_signal[s:e] = 5

    # Plot
    ax[2].plot(1e3*(t[cond]), on_signal[cond])

    print(len(stop_signal))
    start_nm = stop_indices[0]
    print(start_nm)

    duration_nm = 82
    inter_nm = 1700 #1697

    on_signal = np.zeros_like(start_signal)
    
    while start_nm < len(on_signal):
        on_signal[start_nm : start_nm + duration_nm] = 5
        start_nm += inter_nm

    ax[3].plot(1e3*(t[cond]), on_signal[cond])
    
ax[3].set_xlabel("Time (ms)")
ax[0].set_ylabel("Amplitude")
ax[0].grid(True)
ax[1].set_ylabel("Amplitude")
ax[1].grid(True)
ax[0].legend(loc=(1.,0.2))
ax[1].legend(loc=(1.,0.2))
ax[2].grid(True)
ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("Amplitude (V)")
ax[3].grid(True)
ax[3].set_ylabel("Amplitude (V)")

