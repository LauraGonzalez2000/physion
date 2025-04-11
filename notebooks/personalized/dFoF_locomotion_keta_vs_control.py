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

# %% [markdown]
# # dFoF vs locomotion (keta vs control conditions)

# %%
import sys, os
import numpy as np
import matplotlib.pylab as plt

from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData

sys.path.append('../scripts')
from distinct_rest_vs_active import compute_high_movement_cond
from 

# %%
datafolder_keta = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','my_experiments','NWBs_keta')
SESSIONS_keta = scan_folder_for_NWBfiles(datafolder_keta)
SESSIONS_keta['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %%
all_ep_keta = []
all_HMcond_keta = []

for dataIndex in range(len(SESSIONS_keta['files'])):
    data = Data(SESSIONS_keta['files'][dataIndex], verbose=False)
    data.build_dFoF(verbose=False)
    ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])
    all_ep_keta.append(ep)

    HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')
    all_HMcond_keta.append(HMcond)

#all_HMcond_keta = np.concatenate(all_HMcond_keta)

# %%
print(all_ep_keta)
print(all_HMcond_keta)

# %%
datafolder_saline = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','my_experiments','NWBs_saline')
SESSIONS_saline = scan_folder_for_NWBfiles(datafolder_saline)
SESSIONS_saline['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %%
all_ep_saline = []
all_HMcond_saline = []

for dataIndex in range(len(SESSIONS_saline['files'])):
    data = Data(SESSIONS_saline['files'][dataIndex], verbose=False)
    data.build_dFoF(verbose=False)
    ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])
    all_ep_saline.append(ep)

    HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')
    all_HMcond_saline.append(HMcond)

#all_HMcond_saline = np.concatenate(all_HMcond_saline)



# %%
print(all_ep_keta)
print(all_ep_saline[0])
print(len(all_HMcond_keta[0]))
print(len(all_HMcond_saline[2]))


# %%
def plot_dFoF_locomotion_all(all_episodes_saline, 
                             all_episodes_keta,
                             all_HMcond_saline,
                             all_HMcond_keta,
                             general=True, 
                             active=True, 
                             resting=True):
    
    fig, AX = plt.subplots(2, 2, figsize=(8, 4)) 
    fig.subplots_adjust(hspace=0.8)

    i=0

    for label, all_episodes, all_HMcond, ax_col in [("saline", all_episodes_saline, all_HMcond_saline, 0),
                                                    ("keta", all_episodes_keta, all_HMcond_keta, 1)]:
        
        ep = all_episodes[i]
        HMcond = all_HMcond[i]
    
        if general:
            AX[0][ax_col].plot(ep.t, ep.dFoF[:, :, :].mean(axis=0).mean(axis=0), color='blue') 
            AX[1][ax_col].plot(ep.t, ep.running_speed[:, :].mean(axis=0), color='blue')

        if active:
            mask = HMcond
            print(mask.shape)
            if len(mask) != ep.dFoF.shape[0]:
                print(f"Skipping episode {i}: mask length {len(mask)} != dFoF frames {ep.dFoF.shape[0]}")
                continue  # skip this one

            signal = ep.dFoF[mask].mean(axis=(0, 1))
            if len(signal) == len(ep.t):
                AX[0][ax_col].plot(ep.t, ep.dFoF[HMcond, :, :].mean(axis=0).mean(axis=0), color='orangered') 
                AX[1][ax_col].plot(ep.t, ep.running_speed[HMcond, :].mean(axis=0), color="orangered")
            else:
                print(f"Skipping episode {i}: signal length {len(signal)} != time length {len(ep.t)}")

        if resting:
            mask = ~HMcond
            print(mask.shape)
            if len(mask) != ep.dFoF.shape[0]:
                print(f"Skipping episode {i}: mask length {len(mask)} != dFoF frames {ep.dFoF.shape[0]}")
                continue  # skip this one

            signal = ep.dFoF[mask].mean(axis=(0, 1))
            if len(signal) == len(ep.t):
                AX[0][ax_col].plot(ep.t, ep.dFoF[~HMcond, :, :].mean(axis=0).mean(axis=0), color='grey') 
                AX[1][ax_col].plot(ep.t, ep.running_speed[~HMcond, :].mean(axis=0), color="grey")
            else:
                print(f"Skipping episode {i}: signal length {len(signal)} != time length {len(ep.t)}")

        AX[0][ax_col].set_title(f"dFoF for {label} group n = {len(all_episodes)}")
        AX[1][ax_col].set_title(f"locomotion for {label} group n = {len(all_episodes)}")
        
        i+=1

    AX[0][0].axvspan(0, 2, color='lightgrey')
    AX[0][0].set_ylabel('dFoF')
    AX[0][0].set_xlabel('time (s)')
    AX[0][0].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')

    AX[1][0].axvspan(0, 2, color='lightgrey')
    AX[1][0].set_ylabel('locomotion (cm/s)')
    AX[1][0].set_xlabel('time (s)')
    AX[1][0].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')

    AX[0][1].axvspan(0, 2, color='lightgrey')
    AX[0][1].set_ylabel('dFoF')
    AX[0][1].set_xlabel('time (s)')
    AX[0][1].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')

    AX[1][1].axvspan(0, 2, color='lightgrey')
    AX[1][1].set_ylabel('locomotion (cm/s)')
    AX[1][1].set_xlabel('time (s)')
    AX[1][1].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')
    
    return 0


# %%
def plot_dFoF_locomotion_all(all_episodes_saline, 
                             all_episodes_keta,
                             all_HMcond_saline,
                             all_HMcond_keta,
                             general=True, 
                             active=True, 
                             resting=True):
    
    import numpy as np
    import matplotlib.pyplot as plt

    fig, AX = plt.subplots(2, 2, figsize=(8, 4)) 
    fig.subplots_adjust(hspace=0.8)

    for label, all_episodes, all_HMconds, ax_col in [
        ("saline", all_episodes_saline, all_HMcond_saline, 0),
        ("keta", all_episodes_keta, all_HMcond_keta, 1)
    ]:
        # General (no condition)
        if general:

            # Determine target shape based on the first episode
            target_frames = all_episodes[0].dFoF.shape[0]
            
            dFoF_all = []
            speed_all = []
            for ep in all_episodes:
                if ep.dFoF.shape[0] != target_frames:
                    #print(f"Skipping: frame count {ep.dFoF.shape[0]} != target {target_frames}")
                    continue
                dFoF_all.append(ep.dFoF.mean(axis=1))  # mean over ROIs
                speed_all.append(ep.running_speed)
            
            if dFoF_all:
                dFoF_avg = np.mean(np.stack(dFoF_all), axis=0).mean(axis=0)
                speed_avg = np.mean(np.stack(speed_all), axis=0).mean(axis=0)
                AX[0][ax_col].plot(ep.t, dFoF_avg, color='blue')
                AX[1][ax_col].plot(ep.t, speed_avg, color='blue')


        # Active
        if active:

            # Determine target shape based on the first episode
            target_frames = all_episodes[0].dFoF.shape[0]
            
            dFoF_active = []
            speed_active = []

            for ep, HM in zip(all_episodes, all_HMconds):

                if ep.dFoF.shape[0] != target_frames:
                    #print(f"Skipping: frame count {ep.dFoF.shape[0]} != target {target_frames}")
                    continue
                #dFoF_active.append(ep.dFoF[HM].mean(axis=1))  # mean over ROIs
                #speed_active.append(ep.running_speed[HM])

                dFoF_active.append(ep.dFoF[HM].mean(axis=0))  # shape = (num_ROIs,)
                speed_active.append(ep.running_speed[HM].mean(axis=0))  # shape = ()
                
            if dFoF_active:
                print("x", len(dFoF_active[1]))
                #print("y", dFoF_avg)
                #dFoF_avg = np.mean(np.stack(dFoF_active), axis=0).mean(axis=0)
                #speed_avg = np.mean(np.stack(speed_active), axis=0).mean(axis=0)

                dFoF_avg = np.mean(np.stack(dFoF_active), axis=0)  # shape = (num_ROIs,)
                speed_avg = np.mean(speed_active)  # scalar
                print("x", ep.t)
                print("y", dFoF_avg)
                AX[0][ax_col].plot(ep.t, dFoF_avg, color='orangered')
                AX[1][ax_col].plot(ep.t, speed_avg, color='orangered')

        '''
        # Resting
        if resting:

            # Determine target shape based on the first episode
            target_frames = all_episodes[0].dFoF.shape[0]
            
            dFoF_rest = []
            speed_rest = []
            
            for ep, HM in zip(all_episodes, all_HMconds):
                
                if ep.dFoF.shape[0] != target_frames:
                    #print(f"Skipping: frame count {ep.dFoF.shape[0]} != target {target_frames}")
                    continue
                dFoF_rest.append(ep.dFoF[~HM].mean(axis=1))  # mean over ROIs
                speed_rest.append(ep.running_speed[~HM])
                
            if dFoF_rest:
                dFoF_avg = np.mean(np.stack(dFoF_rest), axis=0).mean(axis=0)
                speed_avg = np.mean(np.stack(speed_rest), axis=0).mean(axis=0)
                AX[0][ax_col].plot(ep.t, dFoF_avg, color='grey')
                AX[1][ax_col].plot(ep.t, speed_avg, color='grey')
        '''
        AX[0][ax_col].set_title(f"dFoF for {label} group n = {len(all_episodes)}")
        AX[1][ax_col].set_title(f"locomotion for {label} group n = {len(all_episodes)}")

    # Shared plot formatting
    for i in range(2):
        for j in range(2):
            AX[i][j].axvspan(0, 2, color='lightgrey')
            AX[i][j].set_xlabel('time (s)')
    AX[0][0].set_ylabel('dFoF')
    AX[1][0].set_ylabel('locomotion (cm/s)')
    AX[0][0].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')
    AX[1][0].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')
    AX[0][1].annotate('Visual stim', (0.30, 1), color='black', xycoords='axes fraction', va='top')

    return fig, AX


# %%
plot_dFoF_locomotion_all(all_ep_saline, 
                         all_ep_keta, 
                         all_HMcond_saline,
                         all_HMcond_keta,
                         general=True, 
                         active=True, 
                         resting=False)
