# %% [markdown]
# # Running - Study Variation of dFoF

# %% [markdown]
### Load packages and define constants:
# general python modules for scientific analysis
import sys, pathlib, os, itertools

import sys, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from General_overview_episodes import compute_high_arousal_cond

from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from physion.utils import plot_tools as pt

import scipy.stats as stats_func

import seaborn as sns

running_speed_threshold = 0.5  #cm/s
pre_stim = 1

#%%
def get_variation_dFoF(episodes, cond=None, pre_stim=pre_stim):  #TO CHANGE - automatize boundaries
    
    time_epi = int(episodes.time_duration[0])
    time_epi_p = time_epi * 1000
    ini_p = pre_stim * 1000
    inter_duration_p = int(time_epi_p/5)

    ini_val1 = ini_p - inter_duration_p 
    ini_val2 = ini_p

    final_val1 = ini_p + time_epi_p - inter_duration_p
    final_val2 = ini_p + time_epi_p

    if cond is not None: 
        episodes = episodes.dFoF[np.asarray(cond), :, :]
        ini_val   = episodes[:, :, ini_val1: ini_val2].mean(axis=0).mean(axis=1)
        final_val = episodes[:, :, final_val1: final_val2].mean(axis=0).mean(axis=1)

    else:
        ini_val   = episodes.dFoF[:, :, ini_val1: ini_val2].mean(axis=0).mean(axis=1)
        final_val = episodes.dFoF[:, :, final_val1:final_val2].mean(axis=0).mean(axis=1)
    
    diff = final_val - ini_val

    return diff

def get_vals_plot(episodes, cond):

    traces_act = episodes.dFoF[cond]
    traces_rest = episodes.dFoF[~cond]

    diffs_act = get_variation_dFoF(episodes, cond=cond, pre_stim=pre_stim)
    diffs_rest = get_variation_dFoF(episodes, cond=~cond, pre_stim=pre_stim)

    return traces_act, traces_rest, diffs_act, diffs_rest

def get_stats(all_diffs_act, all_diffs_rest):
    
    t_stats, p_val = stats_func.ttest_ind(all_diffs_act, all_diffs_rest, nan_policy='omit')
    significance = 'ns'                  
    if p_val==np.nan or p_val>0.05:
        significance = 'ns'  # Default is "not significant"
    elif p_val < 0.001:
        significance = '***'
    elif p_val < 0.01:
        significance = '**'
    elif p_val < 0.05:
        significance = '*'

    return t_stats, p_val, significance

def plot_dFoF(traces_act, traces_rest, diffs_act, diffs_rest, protocol, filename, metric):

    cols = 3  # Number of columns per row
    rows = 1  # Compute the required number of rows
    fig, AX = plt.subplots(rows, cols, figsize=(7, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.6)
    #############################################################################################################
    #trace all rois
    plot_trace_vdFoF(fig, AX[0], traces_act, traces_rest)

    if metric =="roi":
        AX[0].set_title(f"File {filename}\n Protocol {protocol},\n average ROIs ( #{len(diffs_act)})")
    elif metric =="sessions":
        AX[0].set_title(f"File {filename}\n Protocol {protocol},\n average sessions ( #{len(diffs_act)})")
    #############################################################################################################
    #barplot all rois
    bar_width = 0.4
    x = np.arange(2)  
    means = [np.nanmean(diffs_act), np.nanmean(diffs_rest)]
    AX[1].bar(x, means, width=bar_width, color=['orangered', 'grey'], edgecolor='black')
    
    x_act = np.full_like(diffs_act, x[0])
    x_rest = np.full_like(diffs_act, x[1])

    if metric == 'roi': 
        jitter_strength = 0.2  # Adjust for more/less jitter
        x_act_jitter = x_act + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_act))
        x_rest_jitter = x_rest + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_rest))

        AX[1].scatter(x_act_jitter, diffs_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
        AX[1].scatter(x_rest_jitter, diffs_rest, color='black', zorder=4, label="Resting", alpha=0.7)
    
    elif metric == "sessions":
        AX[1].scatter(x_act, diffs_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
        AX[1].scatter(x_rest, diffs_rest, color='black', zorder=4, label="Resting", alpha=0.7)
        for i in range(len(diffs_act)):
            color = 'coral' if diffs_rest[i] < diffs_act[i] else 'darkcyan'  # Blue for decrease, Red for increase
            AX[1].plot([x_act[i], x_rest[i]], [diffs_act[i], diffs_rest[i]], color=color, alpha=0.7, lw=1.5)


    AX[1].set_xticks(x, ['Active', 'Resting'])
    AX[1].set_xlabel("Behavioral state")
    AX[1].set_ylabel("Variation of dFoF")
    #AX[1].set_title(f"all ROIs for 1 file \n n ROIs = {len(diffs_act)}")

    t_stats, p_val, significance = get_stats(diffs_act, diffs_rest)
    AX[1].plot([x[0], x[1]], [np.max([means[0], means[1]]) + 5] * 2, color='black', lw=0.8)  # Line above bars
    AX[1].plot([x[0], x[0]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[1].plot([x[1], x[1]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[1].text(np.mean(x), np.max([means[0], means[1]]) + 5.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[1].set_ylim([-0.5,9])
    
    # Annotate each bar with its mean value
    for i in range(2):
        AX[1].text(i, np.max(means) + 3, f'mean {means[i]:.3f}', ha='center', fontsize=6)
    
    print("ALL ROIs for 1 file")
    print("number of ROIs :", len(diffs_act))
    print(f"active mean : {means[0]:.3f}, resting mean : {means[1]:.3f}")
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")

    
    #############################################################################################################
    #violing all rois for 1 file
    
    d = {'active': diffs_act, 'resting': diffs_rest}
    df = pd.DataFrame(data=d)
    df_melted = df.melt(var_name="Behavioral state", value_name="Variation of dFoF")
    sns.violinplot(data=df_melted, 
                   x="Behavioral state",
                   hue="Behavioral state", 
                   y="Variation of dFoF", 
                   inner="quart", 
                   palette={"active": "orangered", "resting": "grey"}, 
                   ax=AX[2], 
                   legend=False)
    
    t_stats, p_val, significance = get_stats(diffs_act, diffs_rest)
    AX[2].plot([x[0]+0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 3] * 2, color='black', lw=0.8)  # Line above bars
    AX[2].plot([x[0]+0.1, x[0]+0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[2].plot([x[1]-0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[2].text(np.mean(x), np.max([means[0], means[1]]) + 3.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[2].set_title(f"all ROIs for 1 file \n n ROIs = {len(diffs_act)}")
    #AX[2].set_ylim([-0.5,9])
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")
    
    # Calculate mean values for labels
    means_ = df_melted.groupby("Behavioral state")["Variation of dFoF"].mean()
    
    # Annotate each bar with its mean value
    for i, mean in enumerate(means_):
        AX[2].text(i, np.max(means_)-0.1, f'mean {mean:.3f}', ha='center', fontsize=6)

    min_plot = np.min(diffs_act)
    max_plot = np.max(diffs_act)
    AX[1].set_ylim([-4,10])
    AX[2].set_ylim([-4,10])
    
    return 0

def plot_trace_vdFoF(fig, ax, traces_act, traces_rest):

    mean_act = np.nanmean(traces_act, axis=(0))
    sem_act  = np.nanstd(traces_act, axis=(0)) / np.sqrt(np.sum(~np.isnan(traces_act), axis=(0)))

    mean_rest = np.nanmean(traces_rest, axis=(0))
    sem_rest  = np.nanstd(traces_rest, axis=(0)) / np.sqrt(np.sum(~np.isnan(traces_rest), axis=(0)))

    # Plot traces +- SEM
    ax.plot(episodes.t, mean_act, color="firebrick", label="Active")
    ax.plot(episodes.t, mean_rest, color="grey", label="Rest")
    ax.fill_between(episodes.t, mean_act - sem_act, mean_act + sem_act,
                    color="firebrick", alpha=0.2)
    ax.fill_between(episodes.t, mean_rest - sem_rest, mean_rest + sem_rest,
                    color="grey", alpha=0.2)

    # Formatting
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dFoF")
    ax.set_xticks(np.arange(0, episodes.t.max() + 1, 1))
    ax.fill_between(np.array([0, episodes.time_duration[0]]), y1=3,
                    color="grey", alpha=0.25)
    ax.fill_between(np.array([-episodes.time_duration[0]/5, 0]), y1=2.8,
                    color="orange", alpha=0.25)
    ax.fill_between(np.array([episodes.time_duration[0] - episodes.time_duration[0]/5,
                              episodes.time_duration[0]]), y1=2.8,
                    color="orange", alpha=0.25)

    return 0

###################################################################################################################
#%% Load Data

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]


dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

#%% ######################################################################################
######################################## TEST 1 FILE #####################################
##########################################################################################
#protocol = "Natural-Images-4-repeats"
protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
#protocol = "static-patch"
#protocol = "looming-stim"

index = 1
filename = SESSIONS['files'][index]
filename_ = os.path.basename(filename)
data = Data(filename,
            verbose=False)
data.build_dFoF(**dFoF_options, verbose=False)
data.build_running_speed()

episodes = EpisodeData(data, 
                       quantities=['dFoF', 'Pupil', 'Running-Speed'],
                       protocol_name=protocol,
                       prestim_duration=pre_stim, 
                       verbose=False)

HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")

traces_act, traces_rest, diffs_act, diffs_rest = get_vals_plot(episodes, cond= HMcond)

flattened_act = [row for arr in traces_act for row in arr]
flattened_rest = [row for arr in traces_rest for row in arr]

plot_dFoF(flattened_act, flattened_rest, diffs_act, diffs_rest, protocol=protocol, filename=filename_, metric = "roi")


#%% ######################################################################################
################################# TEST ALL FILES (ROI combined) ##########################
##########################################################################################
#protocol = "Natural-Images-4-repeats"
#protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
protocol = "static-patch"
#protocol = "looming-stim"

traces_act_s, traces_rest_s, diffs_act_s, diffs_rest_s = [], [], [], []

for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    filename_ = os.path.basename(filename)
    print(filename_)
    data = Data(filename,
                verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()


    episodes = EpisodeData(data, 
                        quantities=['dFoF', 'Pupil', 'Running-Speed'],
                        protocol_name=protocol,
                        prestim_duration=pre_stim, 
                        verbose=False)

    HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")

    traces_act, traces_rest, diffs_act, diffs_rest = get_vals_plot(episodes, cond= HMcond)

    traces_act_s.append(traces_act)
    traces_rest_s.append(traces_rest)
    diffs_act_s.append(diffs_act)
    diffs_rest_s.append(diffs_rest)

temp_act = [np.nanmean(traces_act_s[i], axis=0 ) for i in range(len(traces_act_s))]
flattened_act = [row for arr in temp_act for row in arr]

temp_rest = [np.nanmean(traces_rest_s[i], axis=0 ) for i in range(len(traces_rest_s))]
flattened_rest = [row for arr in temp_rest for row in arr]

#combine ROIs
diffs_act_all = np.concatenate(diffs_act_s) 
diffs_rest_all = np.concatenate(diffs_rest_s)   

plot_dFoF(flattened_act, flattened_rest, diffs_act_all, diffs_rest_all, protocol=protocol, filename="ALL recordings", metric="roi")


#%% ######################################################################################
################ TEST ALL FILES (sessions combined) ######################################
##########################################################################################
#protocol = "Natural-Images-4-repeats"
protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
#protocol = "static-patch"
#protocol = "looming-stim"


traces_act_s, traces_rest_s, diffs_act_s, diffs_rest_s = [], [], [], []

for index in range(len(SESSIONS['files'])):
    filename = SESSIONS['files'][index]
    filename_ = os.path.basename(filename)
    data = Data(filename,
                verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()

    episodes = EpisodeData(data, 
                        quantities=['dFoF', 'Pupil', 'Running-Speed'],
                        protocol_name=protocol,
                        prestim_duration=pre_stim, 
                        verbose=False)

    HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")

    traces_act, traces_rest, diffs_act, diffs_rest = get_vals_plot(episodes, cond= HMcond)

    traces_act_s.append(traces_act)
    traces_rest_s.append(traces_rest)
    diffs_act_s.append(diffs_act)
    diffs_rest_s.append(diffs_rest)

temp_act = [np.nanmean(traces_act_s[i], axis=(0,1)) for i in range(len(traces_act_s))]
temp_rest = [np.nanmean(traces_rest_s[i], axis=(0,1)) for i in range(len(traces_rest_s))]

diff_act_means  = [np.nanmean(diffs_act_s[i]) for i in range(len(diffs_act_s))]
diff_rest_means = [np.nanmean(diffs_rest_s[i]) for i in range(len(diffs_rest_s))]

plot_dFoF(temp_act, temp_rest, diff_act_means, diff_rest_means, protocol=protocol, filename="ALL recordings", metric = "sessions")


########################################################################################
########################################################################################
########################################################################################

#%%
####################################################################
'''

def get_vals(episodes_):
    all_diffs_act = []
    all_diffs_rest = []
    variations_act = []
    variations_rest = []
    
    for i, episodes in enumerate(episodes_):
        
        # HMcond: high movement condition
        HMcond = compute_high_arousal_cond(episodes, pre_stim = pre_stim, running_speed_threshold=0.5, metric="locomotion")
        
        #active
        episodes_act = episodes.dFoF[HMcond]
        episodes_rest = episodes.dFoF[~HMcond]
    
        minsize_subgroup_episodes = int(0.05*len(episodes.dFoF))#~11, 12  #there is a minimum of 5% of total episodes that has to be present in a subgroup to be able to compare  (can be discussed)!!
        
        if (len(episodes_act) > minsize_subgroup_episodes) and (len(episodes_rest) > minsize_subgroup_episodes):  
            diffs_act = []
            n_roi = len(episodes_act[:,:,:].mean(axis=0))
            for i in range(n_roi):
                ini_val = episodes_act[:,i,0].mean(axis=0)
                max_val = np.max(episodes_act[:,i,:].mean(axis=0))
                diff = max_val - ini_val
                diffs_act.append(diff)
                all_diffs_act.append(diff)
                
            variations_act.append(np.mean(diffs_act))    
    
            diffs_rest = []
            n_roi = len(episodes_rest[:,:,:].mean(axis=0))
            for roi in range(n_roi):
                ini_val = episodes_rest[:,roi,0].mean(axis=0)
                max_val = np.max(episodes_rest[:,roi,:].mean(axis=0))
                diff = max_val - ini_val
                diffs_rest.append(diff)
                all_diffs_rest.append(diff)
            variations_rest.append(np.mean(diffs_rest))  
            
    return all_diffs_act, all_diffs_rest, variations_act, variations_rest

def plots_dFoF(all_diffs_act, all_diffs_rest, variations_act, variations_rest, ylim1 = [-0.5,9], ylim2 = [-0.5,3]):
    cols = 2  # Number of columns per row
    rows = 2  # Compute the required number of rows
    #fig, AX = pt.figure(axes=(cols, rows), hspace=2, figsize=(2, 2))
    fig, AX = plt.subplots(rows, cols, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    
    #############################################################################################################
    #barplot all rois for all recordings
    bar_width = 0.4
    x = np.arange(2)  
    means = [np.nanmean(all_diffs_act), np.nanmean(all_diffs_rest)]
    AX[0][0].bar(x, means, width=bar_width, color=['orangered', 'grey'], edgecolor='black')
    jitter_strength = 0.2  # Adjust for more/less jitter
    x_act = np.full_like(all_diffs_act, x[0])
    x_act_jitter = x_act + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_act))
    x_rest = np.full_like(all_diffs_act, x[1])
    x_rest_jitter = x_rest + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_rest))
    AX[0][0].scatter(x_act_jitter, all_diffs_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
    AX[0][0].scatter(x_rest_jitter, all_diffs_rest, color='black', zorder=4, label="Resting", alpha=0.7)
    AX[0][0].set_xticks(x, ['Active', 'Resting'])
    AX[0][0].set_xlabel("Behavioral state")
    AX[0][0].set_ylabel("Variation of dFoF")
    AX[0][0].set_title(f"all ROIs for all recordings\n n = {len(all_diffs_act)}")
    
    t_stats, p_val, significance = get_stats(all_diffs_act, all_diffs_rest)
    AX[0][0].plot([x[0], x[1]], [np.max([means[0], means[1]]) + 5] * 2, color='black', lw=0.8)  # Line above bars
    AX[0][0].plot([x[0], x[0]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[0][0].plot([x[1], x[1]], [np.max([means[0], means[1]]) + 4.8, np.max([means[0], means[1]]) + 5] , color='black', lw=0.8)
    AX[0][0].text(np.mean(x), np.max([means[0], means[1]]) + 5.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[0][0].set_ylim([-0.5,9])
    AX[0][0].set_ylim(ylim1)
    
    # Annotate each bar with its mean value
    for i in range(2):
        AX[0][0].text(i, means[i] + 4, f'mean {means[i]:.3f}', ha='center', fontsize=6)

    #############################################################################################################
    #violing all rois for all recordings
    
    d = {'active': all_diffs_act, 'resting': all_diffs_rest}
    df = pd.DataFrame(data=d)
    df_melted = df.melt(var_name="Behavioral state", value_name="Variation of dFoF")
    sns.violinplot(data=df_melted, 
                   x="Behavioral state", 
                   hue = "Behavioral state", 
                   y="Variation of dFoF", 
                   inner="quart", 
                   palette={"active": "orangered", "resting": "grey"}, 
                   ax=AX[0][1])
    
    t_stats, p_val, significance = get_stats(all_diffs_act, all_diffs_rest)
    AX[0][1].plot([x[0]+0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 3] * 2, color='black', lw=0.8)  # Line above bars
    AX[0][1].plot([x[0]+0.1, x[0]+0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[0][1].plot([x[1]-0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 2.8, np.max([means[0], means[1]]) + 3] , color='black', lw=0.8)
    AX[0][1].text(np.mean(x), np.max([means[0], means[1]]) + 3.1, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    AX[0][1].set_title(f"all ROIs for all recordings\n n = {len(all_diffs_act)}")
    #AX[0][1].set_ylim([-0.5,9])
    AX[0][1].set_ylim(ylim1)
    
    print("ALL ROIs for all files ")
    print("number of ROIs :", len(all_diffs_act))
    print(f"active mean : {means[0]:.3f}, resting mean : {means[1]:.3f}")
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")
    
    # Calculate mean values for labels
    means_ = df_melted.groupby("Behavioral state")["Variation of dFoF"].mean()
    
    # Annotate each bar with its mean value
    for i, mean in enumerate(means_):
        AX[0][1].text(i, mean-0.1, f'mean {mean:.3f}', ha='center', fontsize=6)
    
    
    #############################################################################################################
    
    # Bar plot average ROI for all recordings
    bar_width = 0.4
    x = np.arange(2)  
    means = [np.nanmean(variations_act), np.nanmean(variations_rest)]
    
    AX[1][0].bar(x, means, width=bar_width, color=['orangered', 'grey'], edgecolor='black')
    
    # Jittered scatter points
    #jitter_strength = 0.2
    x_act = np.full_like(variations_act, x[0]) #+ np.random.uniform(-jitter_strength, jitter_strength, size=len(variations_act))
    #x_act_jitter = x_act + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_act))
    x_rest = np.full_like(variations_rest, x[1]) #+ np.random.uniform(-jitter_strength, jitter_strength, size=len(variations_rest))
    #x_rest_jitter = x_rest + np.random.uniform(-jitter_strength, jitter_strength, size=len(x_rest))
    AX[1][0].scatter(x_act, variations_act, color='firebrick', zorder=4, label="Active", alpha=0.7)
    AX[1][0].scatter(x_rest, variations_rest, color='black', zorder=4, label="Resting", alpha=0.7)
    
    # Add lines between corresponding points
    for i in range(len(variations_act)):
        color = 'coral' if variations_rest[i] < variations_act[i] else 'darkcyan'  # Blue for decrease, Red for increase
        AX[1][0].plot([x_act[i], x_rest[i]], [variations_act[i], variations_rest[i]], color=color, alpha=0.7, lw=1.5)
    
    # X-axis labels
    AX[1][0].set_xticks(x)
    AX[1][0].set_xticklabels(['Active', 'Resting'])
    AX[1][0].set_xlabel("Behavioral State")
    AX[1][0].set_ylabel("Variation of dFoF")
    AX[1][0].set_title(f"Average of ROIs for All Recordings\n n Files = {len(variations_act)}")
    
    t_stats, p_val, significance = get_stats(variations_act, variations_rest)
    
    AX[1][0].plot([x[0], x[1]], [np.max([means[0], means[1]]) + 1.3] * 2, color='black', lw=0.8)  # Line above bars
    AX[1][0].plot([x[0], x[0]], [np.max([means[0], means[1]]) + 1.25, np.max([means[0], means[1]]) + 1.3] , color='black', lw=0.8)
    AX[1][0].plot([x[1], x[1]], [np.max([means[0], means[1]]) + 1.25, np.max([means[0], means[1]]) + 1.3] , color='black', lw=0.8)
    AX[1][0].text(np.mean(x), np.max([means[0], means[1]]) + 1.31, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[1][0].set_ylim([-0.5,3])
    AX[1][0].set_ylim(ylim2)
    
    # Annotate each bar with its mean value
    for i in range(2):
        AX[1][0].text(i, np.max(means) + 1, f'mean {means[i]:.3f}', ha='center', fontsize=6)
    
    ################################################################################################################
    #violin plot average ROIs for all recordings
    
    d = {'active': variations_act, 'resting': variations_rest}
    df = pd.DataFrame(data=d)
    df_melted = df.melt(var_name="Behavioral state", value_name="Variation of dFoF")
    
    sns.violinplot(data=df_melted, 
                   x="Behavioral state", 
                   hue = "Behavioral state",
                   y="Variation of dFoF", 
                   inner="quart", 
                   palette={"active": "orangered", "resting": "grey"}, 
                   ax=AX[1][1], 
                   legend=False)
    AX[1][1].set_title(f"average of ROIs for all recordings\n n= {len(variations_act)}")
    t_stats, p_val, significance = get_stats(variations_act, variations_rest)
    AX[1][1].plot([x[0]+0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 0.9] * 2, color='black', lw=0.8)  # Line above bars
    AX[1][1].plot([x[0]+0.1, x[0]+0.1], [np.max([means[0], means[1]]) + 0.8, np.max([means[0], means[1]]) + 0.9] , color='black', lw=0.8)
    AX[1][1].plot([x[1]-0.1, x[1]-0.1], [np.max([means[0], means[1]]) + 0.8, np.max([means[0], means[1]]) + 0.9] , color='black', lw=0.8)
    AX[1][1].text(np.mean(x), np.max([means[0], means[1]]) + 0.91, f"{significance}    p = {p_val:.3f}", ha='center', va='bottom', fontsize=8)
    #AX[1][1].set_ylim([-0.5,3])
    AX[1][1].set_ylim(ylim2)
    
    
    # Calculate mean values for labels
    means_ = df_melted.groupby("Behavioral state")["Variation of dFoF"].mean()
    # Annotate each bar with its mean value
    for i, mean in enumerate(means_):
        AX[1][1].text(i, mean+0.01, f'mean {mean:.3f}', ha='center', fontsize=6)
    
    print(f"\n Average ROIs for all files")
    print("number of files :", len(variations_act))
    print(f"active mean : {means[0]:.3f}, resting mean : {means[1]:.3f}")
    print(f"t_stats : {t_stats:.3f}, p_value : {p_val:.3f}, significance : {significance}")
    return 0

def get_episodes(SESSIONS, dFoF_options, protocol, verbose=True):
    episodes_ = []

    for index in range(len(SESSIONS['files'])):
        
        filename = SESSIONS['files'][index]
        data = Data(filename,
                    verbose=verbose)
        data.build_dFoF(**dFoF_options, verbose=verbose)
        data.build_pupil_diameter()
        data.build_running_speed()
        
        print(data.running_speed)

        episodes = EpisodeData(data, 
                           quantities=['dFoF', 'Running-Speed'],
                           protocol_name=protocol,
                           prestim_duration=0,  ##important !!
                           verbose=False)
        episodes_.append(episodes)
        
    return episodes_

protocol = "looming-stim"
episodes_ = get_episodes(SESSIONS, dFoF_options, protocol, verbose=False)
all_diffs_act, all_diffs_rest, variations_act, variations_rest = get_vals(episodes_)
plots_dFoF(all_diffs_act, all_diffs_rest, variations_act, variations_rest)
'''

#%%
#####################################################################################################
#protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]

#for p in protocols: 
#    print(protocol)
#    episodes_ = get_episodes(SESSIONS, dFoF_options, p, verbose=False)
#    all_diffs_act, all_diffs_rest, variations_act, variations_rest = get_vals(episodes_)
#    plots_dFoF(all_diffs_act, all_diffs_rest, variations_act, variations_rest)