# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis

import sys, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion


import numpy as np
import physion.utils.plot_tools as pt
pt.set_style('dark')
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from physion.dataviz.raw import plot as plot_raw

# %%
# load a datafile
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs-test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                 'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                 'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                 'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                 'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence

#%%
index = 1
filename = SESSIONS['files'][index]

data = Data(filename,
            verbose=False)

data.build_dFoF(**dFoF_options, verbose=False)
data.build_rawFluo(verbose=False)

from physion.analysis.process_NWB import EpisodeData

#%%
protocol = 'drifting-gratings'
prestim = 1
episodes = EpisodeData(data, 
                        quantities=['dFoF', 'rawFluo'],
                        protocol_name=protocol,
                        prestim_duration=prestim, 
                        verbose=False)

# %% [markdown]
# # Show Raster
#%%
from physion.dataviz.episodes.evoked_raster import plot_evoked_pattern 

quantities = ['dFoF', 'rawFluo']
protocol = "Natural-Images-4-repeats"
#protocol = "drifting-gratings"
#protocol = 'moving-dots' 
#protocol = 'random-dots'
#protocol = "static-patch"
#protocol = "looming-stim"
episodes = EpisodeData(data, 
                        quantities=quantities,
                        protocol_name=protocol,
                        prestim_duration=prestim, 
                        verbose=False)
episodes.init_visual_stim(data)
episodes.set_quantities(data, quantities = quantities)

print("protocol_id : ", episodes.index)

#%%
for index in range(len(np.unique(episodes.index))):
    print(index)
    pattern_cond = [True if episodes.index[i]==index else False for i in range(len(episodes.index))]
    print(pattern_cond)
    pt.set_style('manuscript')
    plot_evoked_pattern(episodes,
                        pattern_cond=pattern_cond,
                        with_stim_inset=False,
                        with_mean_trace=True,
                        figsize=(3,1))

