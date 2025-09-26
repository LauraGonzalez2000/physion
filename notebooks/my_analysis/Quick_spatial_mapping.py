# %% [markdown]
# # Quick spatial Mapping

# %%
# load packages:
import os, sys
sys.path += ['../src'] # add src code directory for physion
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.utils  import plot_tools as pt
import numpy as np

# %% [markdown]
# ## Load data

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%%
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

index = 2  #for example this file 
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=True)
data.build_pupil_diameter()
data.build_running_speed()

# %%
quantities = ['dFoF']
protocol = "quick-spatial-mapping"
ep = EpisodeData(data, 
                 quantities = quantities, 
                 protocol_name = protocol, 
                 verbose=True)

#%% [markdown]
# ### Description quick spatial mapping
# 9 static patches at 9 positions x=(-36,0,36) and y=(-26,0,26)
#%%
varied_keys = [k for k in ep.varied_parameters.keys() if k!='repeat']
varied_values = [ep.varied_parameters[k] for k in varied_keys]
print(varied_keys)
print(varied_values)

#%%
# show image here?
#%%
import itertools
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
import matplotlib.pyplot as plt

fig, AX = plt.subplots(3, 3, figsize = (10,10))  # 3x3 grid

i=0
j=2
for values in itertools.product(*varied_values):
    print(values)
    stim_cond = ep.find_episode_cond(key=varied_keys, value=values)
    response = ep.get_response2D(quantity='dFoF',
                                 episode_cond=stim_cond)
    
    AX[j][i].plot(ep.t, response.mean(axis=0))
    
    print(j,i)
    
    if j<2:
        j+=1
    else :
        i+=1
        j=0
        





#%%
def plot(response, title=''):
    fig, AX = pt.figure(figsize=(1,1))
    for r in response:
        AX.plot(ep.t, r, lw=0.4, color='dimgray')
    AX.plot(ep.t, np.mean(response, axis=0), lw=2, color='k')
    pt.set_plot(AX, xlabel='time from start (s)', ylabel='dFoF',
                title=title)

# 3 dimensions (dFoF) - roi = None - averaging dimensions = ROIs [DEFAULT !]
response = ep.get_response2D(quantity="dFoF", averaging_dimension='ROIs')
plot(response, 'mean over ROIs, n=%i eps' % response.shape[0])



fig, AX = pt.figure(axes_extents=[[ [1,1], [1,1], [1,1]],
                                  [ [1,1], [1,1], [1,1]],
                                  [ [1,1], [1,1], [1,1]]])
print(AX)
#AX = AX.flatten()
AX_flat = np.array(AX).flatten()  # shape (9,)
print(AX_flat)
i=0
j=0
for values in itertools.product(*varied_values):
    #print(values)
    #print(stim_cond)

    stim_cond = ep.find_episode_cond(key=varied_keys, value=values)
    
    plot_trial_average(episodes = ep, quantity='dFoF', condition=stim_cond, fig=fig, AX=AX_flat)
    i+=1
pt.show()
# %%
