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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import sys, pathlib
import numpy as np
sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'In_Vivo','physion', 'src'))
from physion.analysis.read_NWB import Data
from physion.dataviz.raw import plot as plot_raw

from physion.dataviz.raw import plot_modif as plot_raw_modif


# %%
def find_available_settings(data):

    settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'Pupil': {'fig_fraction': 2,
                                                  'subsampling': 1,
                                                  'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                                                   'color': '#2ca02c'}}

    

    attributes = ['facemotion', 'pupil_diameter', 'dFoF']
    
    missing = [attr for attr in attributes if not hasattr(data, attr)]  # for objects
    
    if missing:
        print(f"Missing attributes: {missing}")
    else:
        print("All attributes exist")
    
    if missing==['pupil_diameter']:
        print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': '#1f77b4'},
                        'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                        'CaImaging': {'fig_fraction': 10,
                                                   'subsampling': 1,
                                                   'subquantity': 'dF/F',
                                                   'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                                                   'color': '#2ca02c'}}
    
    if missing==['facemotion']:
        print("only pupil diameter missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                        'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'},
                        'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                                               'color': '#2ca02c'}}
        
    if missing==['dFoF']:
        print("only Ca imaging missing")
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'FaceMotion': {'fig_fraction': 1,
                                                   'subsampling': 1,
                                                   'color': 'purple'},
                    'Pupil': {'fig_fraction': 2,
                                          'subsampling': 1,
                                          'color': '#d62728'}}
                        
    
    if missing==['facemotion', 'pupil_diameter']:
        print('facemotion and pupil diameter missing')
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                                               'color': '#2ca02c'}}
        

    return settings

# %%
item = '2025_01_17-15-09-44.nwb'
root = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments', 'my_experiments','All_NWBs')
filename = os.path.join(root, item)
data = Data(filename, verbose=False)
data.build_dFoF()

settings=find_available_settings(data)

plot_raw_modif(data, tlim=[0, data.t_dFoF[-1]], settings=settings, figsize=(9,3))
