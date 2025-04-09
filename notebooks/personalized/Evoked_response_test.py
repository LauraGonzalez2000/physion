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

# %% [markdown]
# # Analyse ketamine effect on trace
#

# %%
import os
import sys, pathlib
import numpy as np

sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'In_Vivo','physion', 'src'))

from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
#from physion.dataviz.raw import plot_test
from physion.dataviz import tools as dv_tools

import matplotlib.pyplot as plt

base_path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments', 'my_experiments', 'All_NWBs')


# %%
# Function to generate full paths
def generate_file_paths(filenames_list, base_path):
    return [os.path.join(base_path, filename) for filename in filenames_list]

# Function to load data and process
def load_and_process_data(filenames):  #not really used, make separate load and process? 
    data_list = []
    for filename in filenames:
        print(filename)
        data = Data(filename, verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile',
                        verbose=False)
        data_list.append(data)
    return data_list



# %%
# Define filenames
names_monitoring_keta = ["2024_10_07-16-26-15.nwb", "2024_10_11-14-13-22.nwb", "2024_10_11-15-46-32.nwb", "2024_10_11-17-26-55.nwb"]
names_8ori_keta       = ["2024_09_12-14-57-34.nwb", "2024_09_12-15-24-47.nwb", "2024_09_12-15-50-12.nwb", "2024_10_07-15-03-40.nwb", 
                         "2024_10_07-17-18-53.nwb", "2024_10_11-14-57-27.nwb", "2024_10_11-16-44-26.nwb", "2024_10_11-18-24-27.nwb"]
names_8ori_saline     = ["2024_08_27-12-21-14.nwb", "2024_08_27-12-46-41.nwb"]

# Generate file paths
filenames_monitoring_keta = generate_file_paths(names_monitoring_keta, base_path)
filenames_8ori_keta = generate_file_paths(names_8ori_keta, base_path)
filenames_8ori_saline = generate_file_paths(names_8ori_saline, base_path)

# Load and process data
DATA_monitoring_keta = load_and_process_data(filenames_monitoring_keta)
DATA_8ori_keta = load_and_process_data(filenames_8ori_keta)
DATA_8ori_saline = load_and_process_data(filenames_8ori_saline)


# %% [markdown]
# # Observe CaImaging response per group per recording per ROI

# %%
def plot1(fns, num_traces):
    num_datasets = len(fns)  # Number of datasets (columns)
    
    fig, axs = plt.subplots(nrows=num_traces, ncols=num_datasets, figsize= (5 * num_datasets, 3 * num_traces), sharex=True, sharey=True)
    
    for i in range(num_datasets):  #loop for each dataset 
        data = Data(fns[i], verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile', 
                        verbose=False)
    
        for j in range(len(data.dFoF[:,:])):
            axs[j,i].plot(data.t_dFoF, data.dFoF[j,:])
            axs[j,i].set_xlabel('Time')
            axs[j,i].set_ylabel('dF/F')
            axs[j,i].set_title(f'ROI {j+1} for Recording {i+1}')
                
    plt.show()
    return 0

def plot12(fns, num_traces):
    num_datasets = len(fns)  # Number of datasets (columns)
    
    fig, axs = plt.subplots(nrows=num_traces, ncols=num_datasets, figsize= (5 * num_datasets, 3 * num_traces), sharex=True, sharey=True)
    
    for i in range(num_datasets):  #loop for each dataset 
        data = Data(fns[i], verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile', 
                        verbose=False,
                        with_correctedFluo_and_F0=True)
    
        for j in range(len(data.dFoF[:,:])):
            axs[j,i].plot(data.t_dFoF, data.dFoF[j,:])
            axs[j,i].set_xlabel('Time')
            axs[j,i].set_ylabel('dF/F')
            axs[j,i].set_title(f'ROI {j+1} for Recording {i+1}')
                
    plt.show()
    return 0

def plot13(fns, num_traces):
    num_datasets = len(fns)  # Number of datasets (columns)
    
    fig, axs = plt.subplots(nrows=num_traces, ncols=num_datasets, figsize= (5 * num_datasets, 3 * num_traces), sharex=True, sharey=True)
    
    for i in range(num_datasets):  #loop for each dataset 
        data = Data(fns[i], verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile', 
                        verbose=False,
                        with_correctedFluo_and_F0=True, 
                        percentile=10)
    
        for j in range(len(data.dFoF[:,:])):
            axs[j,i].plot(data.t_dFoF, data.dFoF[j,:])
            axs[j,i].set_xlabel('Time')
            axs[j,i].set_ylabel('dF/F')
            axs[j,i].set_title(f'ROI {j+1} for Recording {i+1}')
                
    plt.show()
    return 0


# %% [markdown]
# ### monitoring keta

# %%
fns = generate_file_paths(names_monitoring_keta, base_path)
num_traces = 58  # Number of traces (rows)     #find a way to generalize

plot1(fns, num_traces)

# %%
plot12(fns, num_traces)

# %%
plot13(fns, num_traces)

# %% [markdown]
# ### 8ori keta

# %%
fns = generate_file_paths(names_8ori_keta, base_path)
num_traces = 51  # Number of traces (rows)     #find a way to generalize
plot1(fns, num_traces)

# %% [markdown]
# ### 8ori saline

# %%
fns = generate_file_paths(names_8ori_saline, base_path)
num_traces = 27  # Number of traces (rows)     #find a way to generalize
plot1(fns, num_traces)


# %% [markdown]
# # Observe CaImaging response per recording  (ROIs averaged) per group

# %%
def plot2(fns):
    num_datasets = len(fns)  # Number of datasets (columns)
    n_cols = 2
    n_rows = math.ceil(num_datasets / n_cols)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize= (8, 2*n_rows))
    axs = axs.ravel()  # Flatten for easy indexing
    
    for i in range(num_datasets):  #loop for each dataset 
        data = Data(fns[i], verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile', verbose=False)
        axs[i].plot(data.t_dFoF, np.mean(data.dFoF[:,:], axis=0 ))
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('dF/F')
        axs[i].set_title(f'average of all ROIs for Recording {i+1}')
    
    plt.tight_layout()
    plt.show()
    return 0 


# %% [markdown]
# ### monitoring keta

# %%
import math
fns = generate_file_paths(names_monitoring_keta, base_path)
plot2(fns)

# %% [markdown]
# ### 8 ori keta

# %%
fns = generate_file_paths(names_8ori_keta, base_path)
plot2(fns)

# %% [markdown]
# ### 8ori saline

# %%
fns = generate_file_paths(names_8ori_saline, base_path)
plot2(fns)

# %% [markdown]
# ## Calcium imaging per group / other method

# %%
#put the key of what you want to plot
settings = {'CaImaging_mean': {'fig_fraction': 10,
                             'subsampling': 1,
                             'subquantity': 'dF/F', 
                             'color': '#2ca02c'}}

def plot_data(the_list, settings, protocol):
    for data in the_list:
        fig, AX = pt.figure(axes=(1,1), figsize=(3,5), wspace=0.15)
        plot_raw(data, tlim=[0, data.t_dFoF[-1]], settings=settings, ax=AX)
        fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', protocol, f'{data.filename}.png'))


# %%
plot_data(DATA_monitoring_keta, settings, 'Injection_monitoring')

# %%
plot_data(DATA_8ori_keta, settings, '8ori')

# %%
plot_data(DATA_8ori_saline, settings, '8ori')

# %% [markdown]
# # Observe CaImaging response per group  (Recodings averaged, ROIs averaged)

# %%
GROUPS = {'monitoring_keta':{'files':names_monitoring_keta, 'data': DATA_monitoring_keta},  #not using data part
          '8ori_keta':{'files':names_8ori_keta, 'data': DATA_8ori_keta},
          '8ori_saline':{'files':names_8ori_saline, 'data': DATA_8ori_saline}}

fns = [generate_file_paths(GROUPS['monitoring_keta']['files'], base_path),
        generate_file_paths(GROUPS['8ori_keta']['files'], base_path),
        generate_file_paths(GROUPS['8ori_saline']['files'], base_path)]

num_groups = len(GROUPS)  # Number of datasets (columns)

fig, axs = plt.subplots(nrows=1, ncols=num_groups, figsize= (8, 2))

for i in range(num_groups):  #loop for each GROUP
    
    data_all = []
    for j in range(len(fns[i])): #loop for each recording
        data = Data(fns[i][j], verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile', verbose=False)
        data_roi_avg = np.mean(data.dFoF[:,:], axis=0 )
        data_all.append(data_roi_avg)

    max_length = max(arr.shape[0] for arr in data_all)
    padded_arrays = [np.pad(arr, (0, max_length - arr.shape[0]), constant_values=np.nan) for arr in data_all]
    data_avg = np.mean(np.vstack(padded_arrays), axis=0 )
    
    axs[i].plot(data_avg)
    axs[i].set_xlabel('Time')
    axs[i].set_ylabel('dF/F')
    group_name = list(GROUPS.keys())[i]  # Get the group name
    axs[i].set_title(f'{group_name}')
    #axs[i].set_title(f'average of all Recordings for Group {GROUPS[i+1]}')

plt.tight_layout()
plt.show()

