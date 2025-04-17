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
#     display_name: Python (Miniforge Base)
#     language: python
#     name: base
# ---

# %%
# general python modules for scientific analysis
import sys, pathlib, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# add the physion path:
sys.path.append('../../../physion/src')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from physion.utils import plot_tools as pt

from physion.analysis import tools

sys.path.append('../scripts')
from distinct_rest_vs_active import compute_high_movement_cond

import random
from matplotlib.ticker import MultipleLocator


# %% jupyter={"source_hidden": true}
def plot_behavior_in_episodes(data,
                              ax=None,
                              running_speed_threshold=0.1,
                              pupil_threshold=2.9, 
                              metric=None, 
                              mylabel=False):
    
    protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(2,1.3))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
    else:
        fig = None

    HAcount, LAcount = [], []
    for p, protocol in enumerate(protocols):
        try:
            
            behav_episodes = EpisodeData(data, 
                                 quantities=['dFoF', 'Pupil', 'Running-Speed'],
                                 protocol_name=protocol,
                                 prestim_duration=0,
                                 verbose=False)
        except:
            try: 
                behav_episodes = EpisodeData(data, 
                                 quantities=['dFoF', 'Running-Speed'],
                                 protocol_name=protocol,
                                 prestim_duration=0,
                                 verbose=False)
            except: 
                print("error when computing behavioral episodes")

        
        # HAcond: high arousal condition
        HAcond = compute_high_movement_cond(behav_episodes, pupil_threshold, running_speed_threshold, metric=metric)

        ax.scatter(behav_episodes.pupil_diameter.mean(axis=1)[~HAcond],
                   behav_episodes.running_speed.mean(axis=1)[~HAcond],
                   color='grey',
                   s=2)
        ax.scatter(behav_episodes.pupil_diameter.mean(axis=1)[HAcond],
                   behav_episodes.running_speed.mean(axis=1)[HAcond],
                   color='orangered',
                   s=2)
        
        HAcount.append(np.sum(HAcond))
        LAcount.append(np.sum(~HAcond))
        
    ax.set_ylabel('run. speed (cm/s)', fontsize=9)
    ax.set_xlabel('pupil size (mm)', fontsize=9)
    ax.annotate('\n n=%i ep.' % np.sum(HAcount), (0, 1), color='orangered', xycoords='axes fraction', va='top', fontsize=7)
    ax.annotate(' n=%i ep.' % np.sum(LAcount), (0, 1), color='grey', xycoords='axes fraction', va='top', fontsize=7)
    ax.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', bottom=True, left=True)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_title('Behavior across \n visual stimulation episodes', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    if metric=="locomotion":
        ax.axhline(running_speed_threshold, color = 'black', label = 'threshold', linewidth=0.6)
    elif metric=="pupil":
        ax.axvline(pupil_threshold, color = 'black', label = 'threshold', linewidth=0.6)  
    if mylabel:
        ax.annotate('\n high arousal', (1, 1), color='orangered', xycoords='axes fraction', va='top', fontsize=7)
        ax.annotate(' low arousal', (1, 1), color='grey', xycoords='axes fraction', va='top', fontsize=7)
    
    
    return fig, ax


# %% jupyter={"source_hidden": true}
def plot_average_visually_evoked_activity_NDNF(data,
                                           roiIndex=None,
                                           pupil_threshold=2.9,
                                           running_speed_threshold=0.1, 
                                           metric=None):
    
    protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

    # prepare array for final results (averaged over sessions)
    RESULTS = {}
    for protocol in protocols:
        RESULTS[protocol] = {'significant':[], 'response':[], 'session':[]}


    fig, AX = plt.subplots(5, len(protocols),
                              figsize=(7,5))
    pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for p, protocol in enumerate(protocols):

        behav_episodes = EpisodeData(data, 
                             quantities=['Pupil', 'Running-Speed'],
                             protocol_name=protocol,
                             prestim_duration=0,
                             verbose=False)
        
        # HMcond: high movement condition
        HMcond = compute_high_movement_cond(behav_episodes, pupil_threshold, running_speed_threshold, metric=metric)
        
        episodes = EpisodeData(data,
                               quantities=['dFoF'],
                               protocol_name=protocol,
                               verbose=False)
        varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
        varied_values = [episodes.varied_parameters[k] for k in varied_keys]

        AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                          (0.5,1.4),
                          xycoords='axes fraction', ha='center', fontsize=7)

        i=0
        for values in itertools.product(*varied_values):

            stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

            #inset
            inset = pt.inset(AX[i][p], [0.55, 0.9, 0.3, 0.2])
            rest = sum(stim_cond & (~HMcond))
            run  = sum(stim_cond & HMcond)
            inset.pie([rest, run], 
                       autopct='%1.1f%%', 
                       colors = ['grey', 'orangered'], 
                       textprops={'fontsize': 5}, 
                       #labeldistance=0.5)  # Move labels outside
                       pctdistance=2  )  # Move percentage text closer to the edge

            #rest trace
            if np.sum(~HMcond & stim_cond)>1:
                plot_trial_average(episodes, 
                                   roiIndex=roiIndex,
                                   condition=stim_cond & (~HMcond),
                                   with_std=False, with_stim=False,
                                   color='grey',
                                   AX=[[AX[i][p]]], no_set=True)
            #running trace
            if np.sum(HMcond & stim_cond)>1:
                plot_trial_average(episodes, 
                                   roiIndex=roiIndex,
                                   condition=stim_cond & HMcond,
                                   with_std=False, with_stim=True,
                                   color='orangered',
                                   AX=[[AX[i][p]]], no_set=True)

            RESULTS[protocol]['significant'].append([])
            RESULTS[protocol]['response'].append([])
            RESULTS[protocol]['session'].append([])
            i+=1

    if roiIndex is None:
        AX[-1][0].annotate('single session ,   rec: %s \n --> mean$\pm$s.d. over n=%i ROIs' % (\
                                                        data.filename.replace('.nwb',''),data.nROIs),
                           (0, 0),
                          xycoords='axes fraction')
    else:
        AX[-1][0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data.filename.replace('.nwb','')),
                           (0, 0), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
        if np.isfinite(ax.dataLim.x0) and ax==AX[0][0]:
            pt.draw_bar_scales(ax,
                               Xbar=1., Xbar_label='1s',
                               Ybar=1, Ybar_label='1$\Delta$F/F', fontsize=7)
    pt.set_common_xlims(AX)
    
    return fig


# %% jupyter={"source_hidden": true}
def plot_average_visually_evoked_activity(data,
                                           roiIndex=None,
                                           pupil_threshold=2.9,
                                           running_speed_threshold=0.1,
                                           metric=None, 
                                           with_axis=False):
    
    protocol = data.protocols
    RESULTS = {'significant':[], 'response':[], 'session':[]}

    fig, AX = plt.subplots(2, 1, figsize=(4, 2), squeeze=False)
    pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)
    AX[0][0].annotate(protocol[0].replace('ff-gratings-8orientation-2contrasts-15repeats', '8ori-2contrasts'),
                     (0.5,1.4),
                      xycoords='axes fraction', 
                      ha='center')
    try :    
        episodes = EpisodeData(data,
                               quantities=['dFoF', 'Pupil', 'Running-Speed'],
                               protocol_name=protocol,
                               prestim_duration=1,
                               verbose=False)
    except: 
        try: 
            episodes = EpisodeData(data,
                               quantities=['dFoF', 'Running-Speed'],
                               protocol_name=protocol,
                               prestim_duration=1,
                               verbose=False)
        except: 
            try: 
                episodes = EpisodeData(data,
                               quantities=['dFoF', 'Pupil'],
                               protocol_name=protocol,
                               prestim_duration=1,
                               verbose=False)
            except : 
                print("behavioral data missing")

    
    
    HMcond = compute_high_movement_cond(episodes, pupil_threshold, running_speed_threshold, metric="locomotion")
    #HMcond = compute_high_movement_cond(behav_episodes, pupil_threshold, running_speed_threshold, metric="locomotion")

    
    init_val = episodes.data.dFoF.mean(axis=0)[10631]
    max_val = np.max(episodes.data.dFoF.mean(axis=0))
    vdFoF = max_val - init_val
    
    
    #rest trace
    plot_trial_average(episodes, 
                       roiIndex=roiIndex,
                       condition=(~HMcond),
                       with_std=False, with_stim=False,
                       color='grey',
                       AX=[[AX[0][0]]], 
                       no_set=True, 
                       with_axis=with_axis)
       
    #running trace
    plot_trial_average(episodes, 
                       roiIndex=roiIndex,
                       condition=(HMcond),
                       with_std=False, with_stim=True,
                       color='orangered',
                       AX=[[AX[0][0]]], 
                       no_set=True, 
                       with_axis=with_axis)

    AX[0][0].set_xlabel("Time (0.5 ms)? check")
    AX[0][0].set_ylabel("dFoF ")
    
    #inset
    inset = pt.inset(AX[0][0], [0.55, 0.9, 0.3, 0.2])
    rest = sum(~HMcond)
    run  = sum(HMcond)
    inset.pie([rest, run], 
               autopct='%1.1f%%', 
               colors = ['grey', 'orangered'], 
               textprops={'fontsize': 5}, 
               pctdistance=2  )  # Move percentage text closer to the edge
    
    RESULTS['significant'].append([])
    RESULTS['response'].append([])
    RESULTS['session'].append([])
        

    if roiIndex is None:
        AX[0][0].annotate('single session ,   rec: %s \n --> mean$\pm$s.d. over n=%i ROIs' % (\
                                                        data.filename.replace('.nwb',''),data.nROIs),
                           (0, 0),
                          xycoords='axes fraction')
    else:
        AX[0][0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data.filename.replace('.nwb','')),
                           (0, 1.2), xycoords='axes fraction')


    #plot traces running speed
    
    AX[1][0].plot(np.arange(0,len(episodes.running_speed[0]),1), episodes.running_speed[~HMcond][roiIndex], color= 'grey')
    AX[1][0].plot(np.arange(0,len(episodes.running_speed[0]),1), episodes.running_speed[HMcond][roiIndex], color= 'orangered')
    AX[1][0].set_xlabel("Time (0.5 ms)? check")
    AX[1][0].set_ylabel("Speed (cm/s)")

    
    return fig


# %% jupyter={"source_hidden": true}
def plot_average_visually_evoked_activity_(data,
                                           roiIndex=None,
                                           spe_episode = None,
                                           pupil_threshold=2.9,
                                           running_speed_threshold=0.1,
                                           metric=None, 
                                           with_axis=False):
    
    protocol = data.protocols
    RESULTS = {'significant':[], 'response':[], 'session':[]}

    fig, AX = plt.subplots(2, 1, figsize=(4, 2), squeeze=False)
    pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)
    AX[0][0].annotate(protocol[0].replace('ff-gratings-8orientation-2contrasts-15repeats', '8ori-2contrasts'),
                     (0.5,1.4),
                      xycoords='axes fraction', 
                      ha='center')

    if spe_episode is not None: 
        print("only 1 episode")

    else: 
        try :    
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Pupil', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)
        except: 
            try: 
                episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)
            except: 
                try: 
                    episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Pupil'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)
                except : 
                    print("behavioral data missing")
    
        
        
        HMcond = compute_high_movement_cond(episodes, pupil_threshold, running_speed_threshold, metric="locomotion")
        #HMcond = compute_high_movement_cond(behav_episodes, pupil_threshold, running_speed_threshold, metric="locomotion")
    
        
        init_val = episodes.data.dFoF.mean(axis=0)[10631]
        max_val = np.max(episodes.data.dFoF.mean(axis=0))
        vdFoF = max_val - init_val
        
        
        #rest trace
        plot_trial_average(episodes, 
                           roiIndex=roiIndex,
                           condition=(~HMcond),
                           with_std=False, with_stim=False,
                           color='grey',
                           AX=[[AX[0][0]]], 
                           no_set=True, 
                           with_axis=with_axis)
           
        #running trace
        plot_trial_average(episodes, 
                           roiIndex=roiIndex,
                           condition=(HMcond),
                           with_std=False, with_stim=True,
                           color='orangered',
                           AX=[[AX[0][0]]], 
                           no_set=True, 
                           with_axis=with_axis)
    
        AX[0][0].set_xlabel("Time (0.5 ms)? check")
        AX[0][0].set_ylabel("dFoF ")
        
        #inset
        inset = pt.inset(AX[0][0], [0.55, 0.9, 0.3, 0.2])
        rest = sum(~HMcond)
        run  = sum(HMcond)
        inset.pie([rest, run], 
                   autopct='%1.1f%%', 
                   colors = ['grey', 'orangered'], 
                   textprops={'fontsize': 5}, 
                   pctdistance=2  )  # Move percentage text closer to the edge
        
        RESULTS['significant'].append([])
        RESULTS['response'].append([])
        RESULTS['session'].append([])
            
    
        if roiIndex is None:
            AX[0][0].annotate('single session ,   rec: %s \n --> mean$\pm$s.d. over n=%i ROIs' % (\
                                                            data.filename.replace('.nwb',''),data.nROIs),
                               (0, 0),
                              xycoords='axes fraction')
        else:
            AX[0][0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data.filename.replace('.nwb','')),
                               (0, 1.2), xycoords='axes fraction')
    
    
        #plot traces running speed
        
        AX[1][0].plot(episodes.t, episodes.running_speed[~HMcond][roiIndex], color= 'grey')
        AX[1][0].plot(episodes.t, episodes.running_speed[HMcond][roiIndex], color= 'orangered')
        AX[1][0].set_xlabel("Time (s)")
        AX[1][0].set_ylabel("Speed (cm/s)")

    
    return fig


# %% jupyter={"source_hidden": true}
def plot_dFoF_locomotion(episodes, 
                         HMcond, 
                         roi_n=None, 
                         episode_n = None, 
                         general=True, 
                         active=True, 
                         resting=True):

    fig, AX = plt.subplots(2, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.8)

    if episode_n is None: #average of all episodes 
     
        print(f"average of {episodes.dFoF.shape[0]} episodes ({np.sum(HMcond)} active, {len(HMcond)-np.sum(HMcond)} resting)")
        if roi_n is None:
            print(f"average of {episodes.dFoF.shape[1]} ROIs")
            if general:
                AX[0].plot(ep.t, ep.dFoF[:, :, :].mean(axis=0).mean(axis=0), color='blue') 
                AX[1].plot(ep.t, ep.running_speed[:, :].mean(axis=0), color='blue')
            if active: 
                AX[0].plot(ep.t, ep.dFoF[HMcond, :, :].mean(axis=0).mean(axis=0), color='orangered') 
                AX[1].plot(ep.t, ep.running_speed[HMcond, :].mean(axis=0), color="orangered")
            if resting: 
                AX[0].plot(ep.t, ep.dFoF[~HMcond, :, :].mean(axis=0).mean(axis=0), color = 'grey') 
                AX[1].plot(ep.t, ep.running_speed[~HMcond, :].mean(axis=0), color = "grey")
             
        else: 
            print(f"Specific ROI # {roi_n}")
            if general:
                AX[0].plot(ep.t, ep.dFoF[:, roi_n, :].mean(axis=0), color='blue') 
                AX[1].plot(ep.t, ep.running_speed[:, :].mean(axis=0), color='blue')
            if active: 
                AX[0].plot(ep.t, ep.dFoF[HMcond, roi_n, :].mean(axis=0), color='orangered') 
                AX[1].plot(ep.t, ep.running_speed[HMcond, :].mean(axis=0), color="orangered")
            if resting: 
                AX[0].plot(ep.t, ep.dFoF[~HMcond, roi_n, :].mean(axis=0), color = 'grey') 
                AX[1].plot(ep.t, ep.running_speed[~HMcond, :].mean(axis=0), color = "grey")
                
    else: #specific episode
        state = ['active' if HMcond[episode_n] else 'resting']
        #color = 'orangered' if HMcond[episode_n] else 'grey'
        color = 'blue' 
        print(f"Specific episode # {episode_n} ({state})")
        
        if roi_n is None: 
            print(f"average of {episodes.dFoF.shape[1]} ROIs")
            AX[0].plot(ep.t, ep.dFoF[episode_n, :, :].mean(axis=0), color=color) 
            AX[1].plot(ep.t, ep.running_speed[episode_n, :], color=color)
        else: 
            print(f"Specific ROI # {roi_n}")
            AX[0].plot(ep.t, ep.dFoF[episode_n, roi_n, :], color=color) 
            AX[1].plot(ep.t, ep.running_speed[episode_n, :], color=color)
            
    AX[0].set_ylabel('dFoF', fontsize=9)
    AX[1].set_ylabel('locomotion (cm/s)', fontsize=9)

    for ax in AX:
        ax.axvspan(0, 2, color='lightgrey')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
        ax.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', bottom=True, left=True)

    return 0


# %% jupyter={"source_hidden": true}
def plot_dFoF_locomotion_all(all_episodes, 
                         all_HMcond, 
                         general=True, 
                         active=True, 
                         resting=True):

    fig, AX = plt.subplots(2, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.9)

    num_epi_all = 0
    num_roi_all = 0
    for i in range(len(all_episodes)):
        temp_epi = all_episodes[i].dFoF.shape[0]
        num_epi_all += temp_epi
        temp_roi = all_episodes[i].dFoF.shape[1]
        num_roi_all += temp_roi
    print(f"{len(all_episodes)} files")
    print(f"average of {num_epi_all} episodes ({np.sum(np.concatenate(all_HMcond))} active, {len(np.concatenate(all_HMcond))-np.sum(np.concatenate(all_HMcond))} resting)")
    print(f"average of {num_roi_all} ROIs")

    mean_dFoF = np.mean([ep.dFoF.mean(axis=(0, 1)) for ep in all_ep], axis=0)
    mean_running_speed = np.mean([ep.running_speed.mean(axis=0) for ep in all_ep], axis=0)
    
    mean_dFoF_active = np.mean([ep.dFoF[cond, :, :].mean(axis=(0, 1)) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    mean_running_active = np.mean([ep.running_speed[cond, :].mean(axis=0) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    
    mean_dFoF_rest = np.mean([ep.dFoF[~cond, :, :].mean(axis=(0, 1)) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    mean_running_rest = np.mean([ep.running_speed[~cond, :].mean(axis=0) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
  
    if general:
        AX[0].plot(all_ep[0].t, mean_dFoF, color='blue') 
        AX[1].plot(all_ep[0].t, mean_running_speed, color='blue')
    if active: 
        AX[0].plot(all_ep[0].t, mean_dFoF_active, color='orangered') 
        AX[1].plot(all_ep[0].t, mean_running_active, color="orangered")
    if resting: 
        AX[0].plot(all_ep[0].t, mean_dFoF_rest, color = 'grey') 
        AX[1].plot(all_ep[0].t, mean_running_rest, color = "grey")
             
    
    AX[0].set_ylabel('dFoF', fontsize=9)
    AX[1].set_ylabel('locomotion (cm/s)', fontsize=9)

    for ax in AX:
        ax.axvspan(0, 2, color='lightgrey')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
        ax.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', bottom=True, left=True)
    
    return 0

# %% [markdown]
# # NDNF 2022

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %% [markdown]
# ## Example with 1 file (check index below)

# %%
index = 2
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)
data.build_dFoF(verbose=False)

# %%
data.protocols

# %%
EpMvDots = EpisodeData(data, quantities=['dFoF'], protocol_name='looming-stim')

# %%
EpMvDots.dFoF.shape

# %% jupyter={"outputs_hidden": true}
running_speed_threshold=0.1
pupil_threshold = 2.9
fig, ax = plot_behavior_in_episodes(data, running_speed_threshold=running_speed_threshold, metric="locomotion", mylabel=True)
fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/behavior_locomotion_pupil1.png", dpi=300, bbox_inches='tight')
fig, ax = plot_behavior_in_episodes(data, running_speed_threshold=running_speed_threshold, metric="pupil", mylabel=True)
fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/behavior_locomotion_pupil2.png", dpi=300, bbox_inches='tight')

# %% jupyter={"outputs_hidden": true}
dataIndex, roiIndex = 2, 0
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(verbose=False)
fig = plot_average_visually_evoked_activity_NDNF(data, roiIndex=roiIndex, pupil_threshold=2.9, running_speed_threshold=0.1, metric='locomotion')
fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/behavior_different_stim.png", dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Specific ROI, specific episode

# %%
roi = 65 #random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17
epi_num = 34 #random.randint(0, ep.dFoF.shape[0]-1) #chosen randomly  #34 is active

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = epi_num, 
                     general=True, 
                     active=False, 
                     resting=False)


# %% [markdown]
# ### Specific episode, average ROI

# %% jupyter={"outputs_hidden": true}
epi_num = random.randint(0, ep.dFoF.shape[0]-1) #chosen randomly  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = epi_num, 
                     general=True, 
                     active=True, 
                     resting=True)

# %% [markdown]
# ### Specific ROI, average episodes

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=True, 
                     resting=True)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
roi = 14 #random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17  
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% [markdown]
# ### Average ROI, average episodes

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% jupyter={"source_hidden": true, "outputs_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=False, 
                     active=True, 
                     resting=True)

# %% [markdown]
# ## All files

# %% jupyter={"source_hidden": true, "outputs_hidden": true}
rows = 3
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,7))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    plot_behavior_in_episodes(data, ax=AX[int(f/5)][f%5], metric='locomotion', mylabel=False)
    AX[int(f/5)][f%5].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)
    
for i in range(rows*cols-len(SESSIONS['files'])):
    AX[-1][i*(-1) - 1].axis('off')

fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/all_behavior_locomotion_pupil1.png", dpi=300, bbox_inches='tight')

# %% jupyter={"source_hidden": true, "outputs_hidden": true}
rows = 3
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,7))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    plot_behavior_in_episodes(data, ax=AX[int(f/5)][f%5], metric='pupil', mylabel=False)
    AX[int(f/5)][f%5].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)
    
for i in range(rows*cols-len(SESSIONS['files'])):
    AX[-1][i*(-1) - 1].axis('off')

fig.savefig("C:/Users/laura.gonzalez/Output_expe/In_Vivo/NDNF/Behavior/all_behavior_locomotion_pupil2.png", dpi=300, bbox_inches='tight')

# %% [markdown]
# ### ALL files, average ROI, average episodes

# %% jupyter={"outputs_hidden": true}
all_ep = []
all_HMcond = []
df = pd.DataFrame()

for dataIndex in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][dataIndex], verbose=False)
    data.build_dFoF(verbose=False)
    ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])
    all_ep.append(ep)

    HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')
    all_HMcond.append(HMcond)
    
    new_row = pd.DataFrame({
    'File ID': [dataIndex],
    'number of ROIs': [ep.dFoF.shape[1]],
    'number of episodes': [ep.dFoF.shape[0]],
    'number of active episodes': [np.sum(HMcond)], 
    'proportion of active episodes (%)': [(np.sum(HMcond)/ep.dFoF.shape[0])*100]})
    df = pd.concat([df, new_row], ignore_index=True)


# %%
print(df)

# %%
plot_dFoF_locomotion_all(all_ep, 
                         all_HMcond, 
                         general=True, 
                         active=False, 
                         resting=False)

# %% [markdown]
# ---------------------

# %% [markdown]
# # SST Keta vs Saline Cibele and Sofia 2024-2025

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','my_experiments','All_NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %%
dataIndex = 4
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(verbose=False)
data.build_running_speed()
#print(data.pupil_diameter.shape)
print(data.running_speed.shape)

# %%
fig, ax = plot_behavior_in_episodes(data, running_speed_threshold=0.1, metric='locomotion')
ax.set_title('behavior across \n visual stimulation episodes')
ax.annotate('\n high arousal', (1, 1), color='orangered', xycoords='axes fraction', va='top')
ax.annotate(' low arousal', (1, 1), color='grey', xycoords='axes fraction', va='top')

# %%
rows = 4
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,7))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    try:
        data = Data(filename, verbose=False)
        plot_behavior_in_episodes(data, ax=AX[int(f/5)][f%5], running_speed_threshold=0.1, metric='locomotion')
    except Exception as e:
        print(f"Error{e}")
    AX[int(f/5)][f%5].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)


for i in range(rows*cols-len(SESSIONS['files'])):
    AX[-1][i*(-1) - 1].axis('off')

# %%
ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])

HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')

# %% [markdown]
# ### Specific ROI, specific episode

# %% jupyter={"source_hidden": true}
roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17
epi_num = random.randint(0, ep.dFoF.shape[0]-1) #chosen randomly  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = epi_num, 
                     general=True, 
                     active=True, 
                     resting=True)


# %% [markdown]
# ### Specific ROI, average episodes

# %% jupyter={"source_hidden": true}
roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% jupyter={"source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=True, 
                     resting=True)

# %% jupyter={"source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=False, 
                     active=True, 
                     resting=True)

# %% [markdown]
# ### Average ROI, average episodes

# %%
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %%
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=True, 
                     active=True, 
                     resting=True)

# %%
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=False, 
                     active=True, 
                     resting=True)

# %% [markdown]
# ### ALL files, average ROI, average episodes

# %%
all_ep = []
all_HMcond = []

for dataIndex in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][dataIndex], verbose=False)
    data.build_dFoF(verbose=False)
    ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])
    all_ep.append(ep)

    HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')
    all_HMcond.append(HMcond)

all_HMcond = np.concatenate(all_HMcond)

# %%
print(f"{len(SESSIONS['files'])} files")

plot_dFoF_locomotion_all(all_ep, 
                         all_HMcond, 
                         roi_n=None, 
                         episode_n = None, 
                         general=True, 
                         active=True, 
                         resting=True)

# %% [markdown]
# ---

# %% [markdown]
# # SST Keta vs Saline Taddy Yann 2023  --- not enough data!!!

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','SST-Ketamine-vs-Saline', 'All_NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
dataIndex = 4
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(verbose=False)
data.build_running_speed()
data.build_pupil_diameter()
print(data.dFoF.shape)
print(data.pupil_diameter.shape)
print(data.running_speed.shape)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
fig, ax = plot_behavior_in_episodes(data, pupil_threshold = 2.9, running_speed_threshold=0.1, metric='locomotion')
ax.set_title('behavior across \n visual stimulation episodes')
ax.annotate('\n high arousal', (1, 1), color='orangered', xycoords='axes fraction', va='top')
ax.annotate(' low arousal', (1, 1), color='grey', xycoords='axes fraction', va='top')

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
rows = 1
cols = 5
fig, AX = pt.plt.subplots(rows, cols, figsize=(12,2))
pt.plt.subplots_adjust(wspace=0.4, hspace=0.8)

for f, filename in enumerate(SESSIONS['files']):
    try:
        data = Data(filename, verbose=False)
        plot_behavior_in_episodes(data, ax=AX[f], running_speed_threshold=0.1, metric='locomotion')
    except Exception as e:
        print(f"Error{e}")
    AX[f].set_title(str(f+1)+') '+data.filename.replace('.nwb',''), fontsize=8)


# %% jupyter={"outputs_hidden": true, "source_hidden": true}
ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])

HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17
epi_num = random.randint(0, ep.dFoF.shape[0]-1) #chosen randomly  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = epi_num, 
                     general=True, 
                     active=True, 
                     resting=True)


# %% jupyter={"outputs_hidden": true, "source_hidden": true}
roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly  #roi=17  

plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=roi, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% jupyter={"outputs_hidden": true, "source_hidden": true}
plot_dFoF_locomotion(ep, 
                     HMcond, 
                     roi_n=None, 
                     episode_n = None, 
                     general=True, 
                     active=False, 
                     resting=False)

# %% [markdown]
# ## Useless

# %%
ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])

HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')

# %%
from physion.analysis.dataframe import NWB_to_dataframe
df_ = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)


running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
pupil_size_dFoF_sampled = data.build_pupil_diameter(specific_time_sampling=data.t_dFoF)

df = pd.DataFrame()
df['dFoF'] = data.dFoF.mean(axis=0)   #average of all ROIs
df['running_speed'] = running_dFoF_sampled
df['behav_state'] = df['running_speed'].apply(lambda speed: True if speed > 0.1 else False)
df['stim_cond'] = (~df_['VisStim_grey-10min'])

print(df)
print(df_)

plt.figure(figsize=(8, 2))

fig, ax = pt.plt.subplots(1, 1, figsize=(10,2))

for i in range(len(df) - 1):
    if df['stim_cond'].iloc[i] == False:
        ax.axvspan(i, i+1, color='lightgrey', alpha=0.5)

print(np.sum(df['stim_cond'] == False))
        
ax.scatter(x = df['dFoF'].index, 
           y = df['dFoF'],  
           color = ['grey' if state == False else 'orangered' for state in df['behav_state']],
           s=1)
ax.set_xlabel("time (unit?)")
ax.set_ylabel("dFoF")

import seaborn as sns
# Set plot style
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(5, 2))
g = sns.catplot(data=df, 
                 kind = 'violin',
                 x='stim_cond', 
                 y='dFoF', 
                 hue='behav_state', 
                 palette=['grey', 'orangered'], 
                 linewidth=0.8, 
                 legend_out = True)

g.set_xticklabels(["No visual Stimulus", "Visual stimulus"])
new_title = 'Behavioral state'
g._legend.set_title(new_title)
new_labels = ['rest', 'active']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
    

ax = g.ax  # Get the matplotlib axis
grouped = df.groupby(["stim_cond", "behav_state"]).size().reset_index(name='count')
for i, (x, hue) in enumerate(zip(grouped["stim_cond"], grouped["behav_state"])):
    print(x, hue)
    n = grouped.loc[(grouped["stim_cond"] == x) & (grouped["behav_state"] == hue), "count"].values[0]
    ax.text(i/2-0.2, 0, f"n={n}", ha="center", va="bottom", fontsize=12, color="black")

plt.show()

# %%
'''
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol


STAT_TEST = {}f
for protocol in protocols:
    STAT_TEST[protocol] = dict(interval_pre=[-1,0],
                               interval_post=[1,2],
                               test='ttest',
                               positive=True)
    
STAT_TEST['looming-stim']['interval_post'] = [2, 3]
STAT_TEST['drifting-gratings']['interval_post'] = [1.5, 2.5]
STAT_TEST['moving-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['random-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['static-patch']['interval_post'] = [0.5, 1.5]
'''
