# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys, pathlib
import numpy as np

sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'In_Vivo','physion', 'src'))

from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
#from physion.dataviz.raw import plot_test
from physion.dataviz import tools as dv_tools

import matplotlib.pyplot as plt

from scipy import stats
from physion.analysis.process_NWB import EpisodeData

base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'NWBs')

from physion.dataviz.episodes.trial_average import plot as plot_trial_average

import random

# %%
filename = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments', 'NDNF-WT-Dec-2022','NWBs',  '2022_12_14-13-27-41.nwb')
data = Data(filename, verbose=False)

data.build_dFoF()
data.build_running_speed()

# %%
data.nwbfile.acquisition

# %%
ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])

# %%
print(ep.dFoF.shape) # (episodes, rois, time samples)
n_episodes = ep.dFoF.shape[0]
#roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly
roi=11
print(roi)

fig, ax = pt.figure(figsize=(2, 2.5))
temp = ep.dFoF[:,roi,:]
print(temp.shape)
ax.plot(ep.t, temp.mean(axis=0)) #mean of all the episodes
ax.axvspan(0, 2, color='lightgrey')
ax.set_ylabel('dFoF')
ax.set_xlabel('time (s)')
ax.annotate('Visual stim', (0.25, 1), color='black', xycoords='axes fraction', va='top')

# %%
#ep.state is True if animal is running and False if resting
print(np.mean(ep.running_speed, axis=1))
ep.state = [True if speed > 0.07 else False for speed in np.mean(ep.running_speed, axis=1)]
print(ep.state)


# %%
fig, AX = pt.figure(axes=(2,1))
roi= random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly

for state, ax in zip(np.unique(ep.state), AX):
    pt.plot(ep.t, 
            ep.dFoF[ep.state==state, roi, :].mean(axis=0), 
            sy=ep.dFoF[ep.state==state, roi, :].std(axis=0), 
            ax=ax, title=state)

pt.set_common_ylims(AX)

print(len(ep.dFoF[ep.state==state, roi, :].mean(axis=0)))

# %%
i= 0
for run_speed in ep.running_speed :
    print(ep.state)
    if np.mean(run_speed, axis=0) > 0.5:
        ep.state[i] = True
    else: 
        ep.state[i] = False
    i+=1

# %%
roi=random.randint(0, ep.dFoF.shape[1]-1)

fig, ax1 = plt.subplots(figsize=(2, 2.5))
ax2 = ax1.twinx()
print(ep.running_speed.shape)
temp_dFoF = ep.dFoF[:,roi,:]
temp_running = ep.running_speed[roi,:]
print(temp_dFoF.shape)

ax1.plot(ep.t, temp_running)
ax2.plot(ep.t, temp_dFoF.mean(axis=0)) #mean of all the episodes
ax1.axvspan(0, 2, color='lightgrey')




# %%
ep.angl

# %%
