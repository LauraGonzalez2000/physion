# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules for scientific analysis
#import sys, pathlib, os
import numpy as np

# add the python path:
#sys.path.append('../../src')
#from physion.utils import plot_tools as pt
#from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
#from physion.dataviz.raw import plot as plot_raw

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

    
# %%
