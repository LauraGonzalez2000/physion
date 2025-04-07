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

def compute_high_movement_cond(episodes, running_speed_threshold):
    """
    Calculates wether the episodes are active or resting.

    Args:
        episodes (array of Episode): (Episode#, ROI#, dFoF_values (0.5ms sampling rate)).
        running_speed_threshold (float): The threshold to discriminate resting state and active state.

    Returns:
        np.array : HMcond is True when active and false when resting
    """
 
    if running_speed_threshold is not None:
        run_speed_bool = (episodes.running_speed > running_speed_threshold)  
        
        HMcond = []
        prop_thresh = 0.75*len(run_speed_bool[0]) #if 75% values during stimulation are above 0.1, then the animal is "active"
        
        for i in range(len(run_speed_bool)):
            if np.sum(run_speed_bool[i])> prop_thresh:  
                HMcond.append(True)
            else: 
                HMcond.append(False)
                
        HMcond = np.array(HMcond) 

    else: 
        print("running_speed_threshold not given")
    
    return HMcond
# %%
