# %% [markdown]
# # Running - Study Variation of dFoF

# %% [markdown]
### Load packages and define constants:

#%%

# general python modules for scientific analysis
import sys, os

import sys, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion

import numpy as np

from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils import plot_tools as pt
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt

running_speed_threshold = 0.5  #cm/s
pre_stim = 1

###################################################################################################################
#%% Load Data

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%%
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

protocol = "grey-10min"
index = 9
filename = SESSIONS['files'][index]
filename_ = os.path.basename(filename)
data = Data(filename,
            verbose=False)
data.build_dFoF(**dFoF_options, verbose=False)
data.build_running_speed()

#%%
episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

dFoF = episodes.dFoF[0]   #(#ROIs, timevalues)
loco = episodes.running_speed[0]  #(timevalues)

host = host_subplot(111)
par = host.twinx()
#average ROIS: 
p1, = host.plot(np.nanmean(dFoF, axis=0), label="dFoF", color='black', lw=0.5)
p2, = par.plot(loco, label="Running speed", color="blue", lw=0.5)

host.set_xlabel("Time (s)")
host.set_ylabel("dFoF", color="black")
host.set_ylim(bottom=0, top=4)
par.set_ylabel("running speed\n (cm/s)", color='blue')
host.yaxis.label.set_color(p1.get_color())
par.yaxis.label.set_color(p2.get_color())


#%%
dFoF = episodes.dFoF[0]   #(#ROIs, timevalues)
loco = episodes.running_speed[0]  #(timevalues)

# z-score normalization
dFoF_norm = (dFoF - dFoF.mean()) / dFoF.std()
loco_norm = (loco - loco.mean()) / loco.std()

host = host_subplot(111)
par = host.twinx()

host.set_xlabel("Time (s)")
host.set_ylabel("dFoF", color="black")
par.set_ylabel("running speed\n (cm/s)", color='blue')

p1, = host.plot(np.nanmean(dFoF_norm,axis=0), label="dFoF", color='black', lw=0.5)
p2, = par.plot(loco_norm, label="Running speed", color="blue", lw=0.5)

host.yaxis.label.set_color(p1.get_color())
par.yaxis.label.set_color(p2.get_color())


####### problem 601998 samples...due to prestim?
#%%
from scipy.signal import correlate, correlation_lags

# x and y must be 1-D, same length
x = np.nanmean(dFoF_norm,axis=0).copy()   # e.g. shape (T,)
y = loco_norm.copy()   # e.g. shape (T,)

T = len(x)
# center signals (important for correlation)
x0 = x - x.mean()
y0 = y - y.mean()

print(x0.shape)
print(y0.shape)

# cross-correlation normalized to Pearson-like cross-correlation in [-1, 1]
corr_norm = correlate(x0, y0, mode='full')/(T * x0.std(ddof=0) * y0.std(ddof=0))
lags = correlation_lags(T, T, mode='full')

# zero-lag value (correct way)
zero_lag_val = corr_norm[lags == 0][0]
print("Normalized cross-corr at lag 0: %.4f" % zero_lag_val)

# peak correlation and corresponding lag (in samples)
imax = np.argmax(np.abs(corr_norm))
print("Peak normalized correlation: %.4f at lag %d samples" % (corr_norm[imax], lags[imax]))

# Plot a zoomed window around zero (e.g. +/- 2000 samples)
maxlag = 600000
mask_plot = (lags >= -maxlag) & (lags <= maxlag)

plt.figure(figsize=(10, 4))
plt.plot(lags[mask_plot], corr_norm[mask_plot], marker='.', lw=0.5)
plt.axvline(0, color='k', ls='--', alpha=0.6)
plt.axhline(0, color='k', ls='--', alpha=0.6)
plt.scatter([lags[imax]], [corr_norm[imax]], color='red', zorder=5, label=f'peak {corr_norm[imax]:.3f} @ {lags[imax]}')
plt.title('Normalized cross-correlation (Pearson-like)')
plt.xlabel('Lag (samples)')
plt.ylabel('Correlation ([-1,1])')
plt.legend(loc='best')
plt.grid(ls=':', lw=0.5)
plt.show()

#%%[markdown]
#Mini Positive lag -> It means locomotion leads dFoF. movements in loco predict movement in dFoF (but it's almost simultaneous)
#%%
############################################ Linear regression model ##################################
#                                         dFoF​(t)=β0​+β1​⋅locomotion(t)+ϵ(t)
#######################################################################################################
from sklearn.linear_model import LinearRegression

protocol = "grey-10min"
episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

dFoF = episodes.dFoF[0]   #(#ROIs, timevalues)
loco = episodes.running_speed[0]  #(timevalues)

y = dFoF.T              #(timevalues, #ROIs)
X = loco.reshape(-1, 1) #(timevalues, 1)

nROIs, T = dFoF.shape
betas = np.zeros((nROIs, 2))   # store β0 and β1 for each ROI
dFoF_pred = np.zeros_like(dFoF) #(#ROIs, timevalues)

# TRAIN MODEL  - ADD crossvalidation
for i in range(nROIs):
    y = dFoF[i]  # ROI i trace
    model = LinearRegression()
    model.fit(X, y)
    # save model parameters
    betas[i,0] = model.intercept_
    betas[i,1] = model.coef_[0]
    dFoF_pred[i] = model.predict(X) # predicted dFoF from locomotion component for ROI i

dFoF_diff = dFoF - dFoF_pred # (#ROIs, timevalues) signal left, not predicted by locomotion

#%% # Plot example trace
##################################################
roi = 9
plt.plot(dFoF[roi], label='dFoF')
plt.plot(dFoF_diff[roi], label='dFoF diff')
plt.plot(dFoF_pred[roi], label="dFoF predicted")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout

#%% ###########################################################################
##################### TEST this model for visual stimulation ##################
###############################################################################

def predicted_trace(x, b, m) :
    y = b + x*m
    return y 

protocol = "static-patch"
episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

response_dFoF = episodes.dFoF  # (#trials, #ROIs, timevalues)
response_run = episodes.running_speed # (#trials, timevalues)
nROIs, T = episodes.dFoF[0].shape #(#ROIs, timevalues)

#USE MODEL
trial = 2

X = episodes.running_speed[trial].reshape(-1, 1)

y_pred_s = []
for roi in range(nROIs):
    b = betas[i,0]
    m = betas[i,1]
    y_pred = predicted_trace(X, b, m)
    y_pred_s.append(y_pred)

dFoF_diff = response_dFoF[trial] - np.squeeze(y_pred_s)

plt.figure(figsize=(10,4))

plt.plot(response_dFoF[trial][roi], label='raw dFoF_vis', alpha=0.6)
plt.plot(response_run[trial], label='raw loco_vis', alpha=0.6)
plt.plot(y_pred_s[roi], label="predicted", alpha=0.6)
plt.plot(dFoF_diff[roi], label="dFoF_diff", alpha=0.6)
plt.legend()

#%%
protocol = "static-patch"
episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

ntrials, nROIs, T = episodes.dFoF.shape #(#trials, #ROIs, timevalues)

dFoF = episodes.dFoF #(#trials, #ROIs, timevalues)

#USE MODEL
dFoF_pred = []

for trial in range(ntrials):
    response_dFoF = episodes.dFoF[trial] #(#ROIs, timevalues)
    response_run = episodes.running_speed[trial] #(timevalues)

    X = response_run.reshape(-1, 1)
    temp = []
    for roi in range(nROIs):
        b = betas[i,0]
        m = betas[i,1]
        y_pred = predicted_trace(X, b, m)
        temp.append(y_pred)
    dFoF_pred.append(temp)

dFoF_diff = response_dFoF[trial] - np.squeeze(dFoF_pred)

#%% #Plot example trace
#################################################################
trial = 2
roi = 9
plt.figure(figsize=(10,4))
plt.plot(dFoF[trial][roi], label='dFoF')
plt.plot(dFoF_diff[trial][roi], label='dFoF diff')
plt.plot(dFoF_pred[trial][roi], label="dFoF predicted")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout