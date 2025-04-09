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
# ## Cross-validation

# %% [markdown]
# Used to estimate how accurately a predictive model will perform in practice.

# %%
import sys
print(sys.executable)

# general python modules for scientific analysis
import sys, pathlib, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

sys.path.append('../../src')
from sklearn.model_selection import train_test_split
from physion.analysis.read_NWB import Data
from physion.analysis.dataframe import NWB_to_dataframe, extract_stim_keys
from physion.utils import plot_tools as pt

sys.path.append('../../src/physion/analysis')
from cross_validation import TwoFold_train_test_split_basic
from cross_validation import TwoFold_train_test_split


# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
filename = os.path.join(datafolder, '2022_12_14-13-27-41.nwb') #for example
data = Data(filename)
df = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)

# %%
from physion.analysis.process_NWB import EpisodeData

def compute_high_movement_cond(behav_episodes,
                               running_speed_threshold):
    # HMcond: high movement condition
    if running_speed_threshold is not None:
        HMcond = (behav_episodes.running_speed.mean(axis=1)>running_speed_threshold) 
    return HMcond

behav_episodes = EpisodeData(data, 
                             quantities=['Pupil', 'Running-Speed'],
                             #protocol_name=protocol,
                             prestim_duration=0,
                             verbose=False)
        
# HMcond: high movement condition
HMcond = compute_high_movement_cond(behav_episodes, running_speed_threshold=0.1)
fig = plt.figure(figsize=(5, 4))
plt.plot(behav_episodes.pupil_diameter.mean(axis=1)[~HMcond],
        behav_episodes.running_speed.mean(axis=1)[~HMcond], 'o', color='tab:blue')
plt.plot(behav_episodes.pupil_diameter.mean(axis=1)[HMcond],
        behav_episodes.running_speed.mean(axis=1)[HMcond], 'o', color='tab:orange')


# %%
cvIndices = TwoFold_train_test_split_basic(df, spont_act_key='VisStim_grey-10min')
#print(cvIndices)

# Plot
fig, ax = plt.subplots(figsize=(7,2))
ii = 13

#spontaneous
for train, test in zip(cvIndices['spont_train_sets'], cvIndices['spont_test_sets']):
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=8)
    ii-=2
ax.annotate('spont act.    \n(grey screen)        ', (0, ii+3), ha='right')
ii-=1

#evoked
for train, test in zip(cvIndices['stim_train_sets'], cvIndices['stim_test_sets']):
    
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test] , ii+0.5+np.zeros(len(df['time'][test])),  c='tab:blue',marker="_", lw=8)
    ii-=2
ax.annotate('visual stim.    \nperiods        ', (0, ii+3), ha='right')
ii-=1

#stim id
stim_keys = [k for k in df if ('VisStim' in k)]
stimID = 0*df['time']
stim_cond = (~df['VisStim_grey-10min'])

for i, k in enumerate(stim_keys):
        stimID[df[k]] = i+1
        #print(stimID[df[k]])
print(np.unique(stimID[stim_cond]))
ax.scatter(df['time'][stim_cond], [ii-0.5] * np.sum(stim_cond), c=stimID[stim_cond], marker="_", lw=8, cmap=plt.cm.tab20)
ax.annotate('visual stim. ID  ', (0, ii-1), ha='right')
ii-=3

#movement
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)  
speed_bool = pd.Series((running_dFoF_sampled >= 0.60).astype(int)) 
ax.scatter(x = df['time'][speed_bool == 1], 
           y = ii + speed_bool[speed_bool == 1] - 0.5, 
           c='tab:orange', 
           marker="_", 
           lw=8, 
           label="Above Threshold")
ax.annotate('movement  ', (0, ii), ha='right')

#final arangements
ax.annotate('training set', (.8,.9), color='tab:red', xycoords='axes fraction')
ax.annotate('test set\n', (.8,.9), color='tab:blue', xycoords='axes fraction')
ax.axis('off')
ax.set_xlabel("time (s)")
ax.set_title('2-Fold Cross-Validation strategy\n ')
ax.axes.get_xaxis().set_visible(True)

# %% [markdown]
# Check how many movement and quiet there is in training and test set 

# %%
#2 different state (moving vs quiet)
print(len(np.unique(speed_bool)))

# %%
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)

for i in range(2):
    print(f"{i} CV")
    for subset in cvIndices.keys():
        speed_bool = pd.Series((running_dFoF_sampled >= 0.10).astype(int))[cvIndices[subset][i]]
        print(f"In the subset {subset}, there is {sum(speed_bool)} moments moving out of {len(speed_bool)}")
        

# %% [markdown]
# Check composition of stimuli in training and test sets.

# %%
#19 different stimuli
print(np.unique(stim_keys))


# %%
cvIndices = TwoFold_train_test_split(filename, data, df, spont_act_key='VisStim_grey-10min')
sets = ['stim_train_sets', 'stim_test_sets']
keys = np.unique(stim_keys)
final = []

for i in range(2):
    print(f'\n CV fold {i}')
    for my_set in sets:
        subset = df[cvIndices[my_set][i][0]:cvIndices[my_set][i][-1]]
        for key in keys:
            stim_cond = (subset[key])
            final.append(np.sum(stim_cond))
            print(f'There is {np.sum(stim_cond)} stimulus {key}')

    #for j in range(len(keys)):
    #    print(f'\nStimulus {keys[j]} :\n {final[j]} in {sets[0]} and {final[j*2]} in {sets[1]}')    

# %%
cvIndices = TwoFold_train_test_split_basic( df, spont_act_key='VisStim_grey-10min')
#print(cvIndices)

# Plot
fig, ax = plt.subplots(figsize=(7,2))
ii = 13

#spontaneous
for train, test in zip(cvIndices['spont_train_sets'], cvIndices['spont_test_sets']):
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=8)
    ii-=2
ax.annotate('spont act.    \n(grey screen)        ', (0, ii+3), ha='right')
ii-=1

#evoked
for train, test in zip(cvIndices['stim_train_sets'], cvIndices['stim_test_sets']):
    
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test] , ii+0.5+np.zeros(len(df['time'][test])),  c='tab:blue',marker="_", lw=8)
    ii-=2
ax.annotate('visual stim.    \nperiods        ', (0, ii+3), ha='right')
ii-=1

#stim id
stim_keys = [k for k in df if ('VisStim' in k)]
stimID = 0*df['time']
stim_cond = (~df['VisStim_grey-10min'])

for i, k in enumerate(stim_keys):
        stimID[df[k]] = i+1
        #print(stimID[df[k]])
#print(np.unique(stimID[stim_cond]))
ax.scatter(df['time'][stim_cond], [ii-0.5] * np.sum(stim_cond), c=stimID[stim_cond], marker="_", lw=8, cmap=plt.cm.tab20)
ax.annotate('visual stim. ID  ', (0, ii-1), ha='right')
ii-=3

#movement
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)  
speed_bool = pd.Series((running_dFoF_sampled >= 0.60).astype(int)) 
ax.scatter(x = df['time'][speed_bool == 1], 
           y = ii + speed_bool[speed_bool == 1] - 0.5, 
           c='tab:orange', 
           marker="_", 
           lw=8, 
           label="Above Threshold")
ax.annotate('movement  ', (0, ii), ha='right')

#final arangements
ax.annotate('training set', (.8,.9), color='tab:red', xycoords='axes fraction')
ax.annotate('test set\n', (.8,.9), color='tab:blue', xycoords='axes fraction')
ax.axis('off')
ax.set_xlabel("time (s)")
ax.set_title('2-Fold Cross-Validation strategy\n ')
ax.axes.get_xaxis().set_visible(True)

running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)

# Debugging: Print the number of running/not running instances
for i in range(2):
    print(f"\nFold {i + 1}")
    for subset in cvIndices.keys():
        speed_bool = pd.Series((running_dFoF_sampled >= 0.10).astype(int))[cvIndices[subset][i]]
        print(f"{subset} ({len(speed_bool)})\n Running {sum(speed_bool)} Non-running {len(speed_bool)- sum(speed_bool) }")


# %%
cvIndices = TwoFold_train_test_split(filename, data, df, spont_act_key='VisStim_grey-10min')
#print(cvIndices)

# Plot
fig, ax = plt.subplots(figsize=(7,2))
ii = 13

#spontaneous
for train, test in zip(cvIndices['spont_train_sets'], cvIndices['spont_test_sets']):
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=8)
    ii-=2
ax.annotate('spont act.    \n(grey screen)        ', (0, ii+3), ha='right')
ii-=1

#evoked
for train, test in zip(cvIndices['stim_train_sets'], cvIndices['stim_test_sets']):
    
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
    ax.scatter(df['time'][test] , ii+0.5+np.zeros(len(df['time'][test])),  c='tab:blue',marker="_", lw=8)
    ii-=2
ax.annotate('visual stim.    \nperiods        ', (0, ii+3), ha='right')
ii-=1

#stim id
stim_keys = [k for k in df if ('VisStim' in k)]
stimID = 0*df['time']
stim_cond = (~df['VisStim_grey-10min'])

for i, k in enumerate(stim_keys):
        stimID[df[k]] = i+1
        #print(stimID[df[k]])
#print(np.unique(stimID[stim_cond]))
ax.scatter(df['time'][stim_cond], [ii-0.5] * np.sum(stim_cond), c=stimID[stim_cond], marker="_", lw=8, cmap=plt.cm.tab20)
ax.annotate('visual stim. ID  ', (0, ii-1), ha='right')
ii-=3

#movement
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)  
speed_bool = pd.Series((running_dFoF_sampled >= 0.60).astype(int)) 
ax.scatter(x = df['time'][speed_bool == 1], 
           y = ii + speed_bool[speed_bool == 1] - 0.5, 
           c='tab:orange', 
           marker="_", 
           lw=8, 
           label="Above Threshold")
ax.annotate('movement  ', (0, ii), ha='right')

#final arangements
ax.annotate('training set', (.8,.9), color='tab:red', xycoords='axes fraction')
ax.annotate('test set\n', (.8,.9), color='tab:blue', xycoords='axes fraction')
ax.axis('off')
ax.set_xlabel("time (s)")
ax.set_title('2-Fold Cross-Validation strategy\n ')
ax.axes.get_xaxis().set_visible(True)

running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)

# Debugging: Print the number of running/not running instances
for i in range(2):
    print(f"\nFold {i + 1}")
    for subset in cvIndices.keys():
        speed_bool = pd.Series((running_dFoF_sampled >= 0.10).astype(int))[cvIndices[subset][i]]
        print(f"{subset} ({len(speed_bool)})\n Running {sum(speed_bool)} Non-running {len(speed_bool)- sum(speed_bool) }")

