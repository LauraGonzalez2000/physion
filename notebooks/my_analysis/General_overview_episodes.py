# %% [markdown]
# #General overview episodes

# %%
# load packages:
import os, sys
sys.path += ['../../src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils  import plot_tools as pt
import numpy as np
import matplotlib as plt
import itertools
from physion.dataviz.episodes.trial_average import plot as plot_trial_average, get_trial_average_trace

# %%
def compute_high_arousal_cond(episodes, 
                              pre_stim = 0,
                              pupil_threshold = 0.29, 
                              running_speed_threshold = 0.1, 
                              metric = None):
    """
    Calculates wether the episodes are aroused/active or calm/resting.

    Args:
        episodes (array of Episode): (Episode#, ROI#, dFoF_values (0.5ms sampling rate)).
        pupil_threshold (float) : The threshold to discriminate calm state and aroused state
        running_speed_threshold (float): The threshold to discriminate resting state and active state.
        metric (string) : metric used to split calm/rest and aroused/active states. ("pupil" or "locomotion")

    Returns:
        np.array : HMcond is True when active/aroused and false when resting/calm
    """
    cond = []
    
    if metric=="pupil":
        '''
        if pupil_threshold is not None:
            cond = (episodes.pupil_diameter.mean(axis=1)>pupil_threshold)
        else:
            print("pupil_threshold not given")
        '''
        if pupil_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.pupil_diameter[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > pupil_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("pupil_threshold not given")
            


    if metric=="locomotion":
        
        if running_speed_threshold is not None: 
            start = int(pre_stim*1000)
            end = int(start + episodes.time_duration[0]*1000)
            values = episodes.running_speed[:, start:end]  ## check if these boundaries cause problem #1000:3001
            for value in values: 
                if (np.mean(value) > running_speed_threshold):
                    cond.append(True)
                else: 
                    cond.append(False)
            cond = np.array(cond) 
    
        else: 
            print("running_speed_threshold not given")

    return cond

#%%
def plot_dFoF_per_protocol(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.5, 
                           metric=None, 
                           protocols = [], 
                           subplots_n=5):
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    

    fig, AX = pt.figure(axes_extents=[[ [1,1] for _ in protocols ] for _ in range(subplots_n)])  #generalize 9 

    for p, protocol in enumerate(protocols):
        session_traces = []

        for data in data_s:
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
            else:
                cond = episodes.find_episode_cond()
            
            varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]

            i = 0
            for values in itertools.product(*varied_values):
                stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

                mean_trace, sem_trace = get_trial_average_trace(
                    episodes,
                    roiIndex=roiIndex,
                    condition=stim_cond & cond
                )
                
                if mean_trace is not None:
                    session_traces.append((i, mean_trace, sem_trace))
                i += 1
        
        # plotting
        n_conditions = len(list(itertools.product(*varied_values)))

        for j in range(n_conditions):
            traces = [tr for idx, tr, _ in session_traces if idx == j]
            sems   = [se for idx, _, se in session_traces if idx == j]
            
            if len(traces) == 0:
                continue  # nothing to plot for this condition

            if mode == "single":
                mean_trace = traces[0]
                sem_trace  = sems[0]
            else:
                mean_trace = np.mean(traces, axis=0)
                sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

            
            AX[j][p].plot(mean_trace, color='k')
            AX[j][p].fill_between(np.arange(len(mean_trace)),
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color='k', alpha=0.3)
            AX[j][p].axvspan(1000, 1000+1000*episodes.time_duration[0], color='lightgrey', alpha=0.5, zorder=0)

        AX[0][p].set_title(f'{protocol.replace('Natural-Images-4-repeats','natural-images')}')   
        #AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
        #                  (0.5,1.4),
        #                  xycoords='axes fraction', ha='center', fontsize=7)
    
    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX[-1][0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
        else:
            AX[-1][0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
    else:
        if mode == "single":
            AX[-1][0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
        else:
            AX[-1][0].annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
    pt.set_common_xlims(AX)
    
    return fig, AX

def plot_dFoF_per_protocol2(data_s,
                           dataIndex=None,
                           roiIndex=None,
                           pupil_threshold=2.9,
                           running_speed_threshold=0.1, 
                           metric=None, 
                           found=True):
    """
    Plot dFoF per protocol for a single session or across multiple sessions.

    Parameters
    ----------
    data_list : list
        List of sessions.
    dataIndex : int or None
        If int, plot only that session from data_list.
        If None, average across all sessions.
    roiIndex : int or None
        If int, plot a specific ROI.
        If None, average across all ROIs.
    pupil_threshold : float
        Threshold for pupil dilation (arousal condition).
    running_speed_threshold : float
        Threshold for running speed (arousal condition).
    metric : str or None
        Metric to split high/low arousal conditions.
    """
    # select sessions
    if dataIndex is not None:
        mode = "single"
    else:
        mode = "average"
    
    # protocols (assume same across sessions)
    protocols = [p for p in data_s[0].protocols 
                 if (p != 'grey-10min') and (p != 'black-2min') and (p != 'quick-spatial-mapping')]
    


    fig, AX = pt.figure(axes = (len(protocols),1))

    for p, protocol in enumerate(protocols):
        session_traces = []

        for data in data_s:
            episodes = EpisodeData(data,
                                   quantities=['dFoF', 'Running-Speed'],
                                   protocol_name=protocol,
                                   prestim_duration=1,
                                   verbose=False)

            if metric is not None:
                cond = compute_high_arousal_cond(episodes, pupil_threshold, running_speed_threshold, metric=metric)
            else:
                cond = episodes.find_episode_cond()
            

            # TO FIX : find a better solution
            varied_keys = [k for k in episodes.varied_parameters.keys() if (k != 'repeat') and (k != 'angle') and (k != 'contrast') and (k != 'speed') and (k != 'Image-ID') and (k != 'seed')]
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]


            i = 0
            for values in itertools.product(*varied_values):
                stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)

                mean_trace, sem_trace = get_trial_average_trace(
                    episodes,
                    roiIndex=roiIndex,
                    condition=stim_cond & cond
                )
                if mean_trace is not None:
                    session_traces.append((i, mean_trace, sem_trace))
                i += 1

        # plotting
        n_conditions = len(protocols)
        for j in range(n_conditions):
            traces = [tr for idx, tr, _ in session_traces if idx == j]
            sems   = [se for idx, _, se in session_traces if idx == j]

            if len(traces) == 0:
                continue  # nothing to plot for this condition

            if mode == "single":
                mean_trace = traces[0]
                sem_trace  = sems[0]
            else:
                mean_trace = np.mean(traces, axis=0)
                sem_trace  = np.std(traces, axis=0) / np.sqrt(len(traces))

            AX[p].plot(mean_trace, color='k', linewidth=0.1)
            AX[p].fill_between(np.arange(len(mean_trace)),
                                mean_trace - sem_trace,
                                mean_trace + sem_trace,
                                color='k', alpha=0.3)
            AX[p].axvspan(1000, 1000+1000*episodes.time_duration[0], color='lightgrey', alpha=0.5, zorder=0)

        AX[p].set_title(f'{protocol.replace('Natural-Images-4-repeats','natural-images')}')    
        
        #AX[p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
        #                  (0.5,1.4),
        #                  xycoords='axes fraction', ha='center', fontsize=7)
    
    # annotate session or ROI info
    if roiIndex is None:
        if mode == "single":
            AX[0].annotate('single session: %s ,   n=%i ROIs' %
                               (data_s[0].filename.replace('.nwb',''), data_s[0].nROIs),
                               (0, -0.2), xycoords='axes fraction')
            
        else:
            AX[0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (0, -0.2), xycoords='axes fraction')
            
    else:
        if mode == "single":
            AX[0].annotate('roi #%i ,   rec: %s' % (1+roiIndex, data_s[0].filename.replace('.nwb','')),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX[0].annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)
        else:

            AX[0].annotate('roi #%i , average over %i sessions' % (1+roiIndex, len(data_s)),
                               (0, -0.2), xycoords='axes fraction', fontsize=7)
            
            if not found: 
                AX[0].annotate('Responsive roi not found, took ns ROI',
                                (0, -0.4), xycoords='axes fraction', fontsize=7)

    pt.set_common_ylims(AX)
    for ax in pt.flatten(AX):
        ax.axis('off')
    pt.set_common_xlims(AX)
    
    return fig, AX

#%% [markdown]
# ### Load data
# %%

def generate_data_s(datafolder):
    SESSIONS = scan_folder_for_NWBfiles(datafolder)
    dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                    'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                    'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                    'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                    'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence
    data_s = []
    for i in range(len(SESSIONS['files'])):
        data = Data(SESSIONS['files'][i], verbose=False)
        data.build_dFoF(**dFoF_options, verbose=True)
        data_s.append(data)
    return data_s

def heavy():
    #  [markdown]
    # ### 1 single ROI of 1 single session
    # 
    datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch1','NWBs')
    data_s = generate_data_s(datafolder)
    dataIndex, roiIndex = 4, 2
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex = dataIndex, roiIndex=roiIndex, metric=None)

    #  [markdown]
    # ### 1 single session (average ROIs)
    # 
    dataIndex, roiIndex = 2, None
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex=dataIndex, roiIndex=roiIndex, metric=None)

    # [markdown]
    # ### Average sessions (and average ROIs)
    # 
    dataIndex, roiIndex = None, None
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex=dataIndex, roiIndex=roiIndex, metric=None)
    return 0
################################################## YANN's DATA ####################################################

#%% [markdown]
# ### Load data
# %%
def heavy2():


    datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-WT-Dec-2022','NWBs')
    data_s = generate_data_s(datafolder)

    #  [markdown]
    # ### 1 single ROI of 1 single session
    # 
    dataIndex, roiIndex = 2, 2
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex = dataIndex, roiIndex=roiIndex, metric=None)

    #  [markdown]
    # ### 1 single session (average ROIs)
    # 
    dataIndex, roiIndex = 2, None
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex=dataIndex, roiIndex=roiIndex, metric=None)

    #  [markdown]
    # ### Average sessions (and average ROIs)
    # 
    dataIndex, roiIndex = None, None
    fig, AX = plot_dFoF_per_protocol(data_s=data_s, dataIndex=dataIndex, roiIndex=roiIndex, metric=None)
    return 0 

#%%
#heavy()