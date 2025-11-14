# %% [markdown]
# # Running overview 

# %% [markdown]
### Load packages and define constants:

#%%
import sys, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion

import numpy as np

from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from physion.analysis.behavior import population_analysis
import matplotlib.pyplot as plt

from physion.analysis.process_NWB import EpisodeData

from physion.dataviz.raw import plot as plot_raw

from General_overview_episodes import compute_high_arousal_cond

import random

running_speed_threshold = 0.5  #cm/s


def plot_locomotion(data,  
                    protocol,
                    running_speed_threshold=0.5,
                    pupil_threshold=2.9, 
                    pre_stim = 0):

    episodes = EpisodeData(data, 
                           quantities=['dFoF', 'running_speed', 'pupil'], 
                           protocol_name=protocol, 
                           prestim_duration=pre_stim)

    HMcond = compute_high_arousal_cond(episodes, 
                                       pre_stim = pre_stim,
                                       pupil_threshold = pupil_threshold, 
                                       running_speed_threshold = running_speed_threshold, 
                                       metric = 'locomotion')

    episode_n = random.randint(0, episodes.dFoF.shape[0]-1) #chosen randomly 1 
    
    fig, AX = plt.subplots(1, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.8)
    
    start = int(pre_stim*1000)
    end = int(start + episodes.time_duration[0]*1000)

    print("start : ", start)
    print("end : ", end)
    print(episodes.running_speed[episode_n, start:end])
    print("mean : ", episodes.running_speed[episode_n, start:end].mean(axis=0))
    
    start_sec = 0
    end_sec = int(episodes.time_duration[0])
    
    state = ['active' if HMcond[episode_n] else 'resting']
    color = 'orangered' if HMcond[episode_n] else 'grey'
    #color = 'blue' 
    print(f"Specific episode # {episode_n} ({state})") 
    
    AX.plot(episodes.t, episodes.running_speed[episode_n, :],color=color)
    AX.hlines(episodes.running_speed[episode_n, start:end].mean(axis=0),
                  xmin = start_sec,
                  xmax = end_sec,
                  color='dimgray',
                  linestyle=':')
    print("value to compare to threshold : ", episodes.running_speed[episode_n,start:end].mean(axis=0))

    AX.axvspan(0, episodes.time_duration[0], color='lightgrey')
    
    AX.set_xlabel('Time (s)', fontsize=9)
    AX.set_ylabel('locomotion (cm/s)', fontsize=9)
    AX.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
    AX.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
    AX.grid(False)
    AX.tick_params(axis='both', which='both', bottom=True, left=True)
    AX.axhline(0.5, color="crimson")
    
    return fig, AX


#running overview necessary to filter good files. 
#%% Load Data
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_run')
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_run')
SESSIONS = scan_folder_for_NWBfiles(datafolder)


#%%

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

#%%

#Load episodes by protocol!
protocol = "moving-dots"
pre_stim = 1


dataIndex, roiIndex = 2, 0
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(**dFoF_options, verbose=True)


#%%
plot_locomotion(data, 
                protocol='static-patch',
                running_speed_threshold=0.5,
                pupil_threshold=2.9, 
                pre_stim = 1)


#%%
fig = plot_average_visually_evoked_activity_NDNF(data, roiIndex=roiIndex, pupil_threshold=2.9, running_speed_threshold=0.1, metric='locomotion')
fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/behavior_different_stim.png", dpi=300, bbox_inches='tight')