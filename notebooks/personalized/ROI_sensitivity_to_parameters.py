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
from scipy import stats
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
import random
import sklearn
from sklearn.linear_model import LinearRegression

import sys
print(sys.executable)

# %%
#data NWBs

#my_experiments
names_monitoring_keta = ["2024_10_07-16-26-15.nwb", "2024_10_11-14-13-22.nwb", "2024_10_11-15-46-32.nwb", "2024_10_11-17-26-55.nwb"]
names_8ori_keta       = ["2024_09_12-14-57-34.nwb", "2024_09_12-15-24-47.nwb", "2024_09_12-15-50-12.nwb", "2024_10_07-15-03-40.nwb", 
                         "2024_10_07-17-18-53.nwb", "2024_10_11-14-57-27.nwb", "2024_10_11-16-44-26.nwb", "2024_10_11-18-24-27.nwb"]
names_8ori_saline     = ["2024_08_27-12-21-14.nwb", "2024_08_27-12-46-41.nwb"]




# %%
#Methods

# Function to generate full paths
def generate_file_paths(filenames_list, base_path):
    return [os.path.join(base_path, filename) for filename in filenames_list]

# Function to load data and process
def load_and_process_data(filenames):
    data_list = []
    for filename in filenames:
        print(filename)
        data = Data(filename, verbose=False)
        data.build_dFoF(method_for_F0='sliding_percentile',
                        verbose=False)
        data_list.append(data)
    return data_list

def linear_regression(x, y, ax):
    #linear regression
    #x = np.array(run_speed_sampled).reshape((-1,1))
    #y = np.array(data.dFoF[roi,:])
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x,y)
    fit = model.coef_*x + model.intercept_ 
    ax.plot(x, fit, color='red')
    r_sq = model.score(x,y)
    ax.annotate(text=fr"$R^2 = {r_sq:.3f}$", xy=[0.5,32], fontsize=6, color='red')
    #print(f"coefficient of determination Rsquared: {r_sq}" )
    #print(f"intercept : {model.intercept_}")
    #print(f"slope : {model.coef_}") 
    return r_sq

def lr_dFoF_running(run_speed_sampled, dFoF, r_sq_s):
    
    n_ori = dFoF.shape[0]  # Total number of plots
    cols = 8  # Number of columns per row
    rows = (n_ori + cols - 1) // cols  # Compute the required number of rows
    ROIs = np.arange(0,n_ori,1)
    
    fig, AX = pt.figure(axes=(cols, rows), hspace=2)  # Create a grid with (rows, cols)
    
    # Ensure AX is a list (some plotting libraries return nested lists)
    AX = AX if isinstance(AX, list) else np.ravel(AX)
    AX = np.array(AX).flatten()
    
    for roi, ax in zip(ROIs, AX):
        ax.scatter(run_speed_sampled, dFoF[roi, :])  # Now using ax.scatter()
        ax.set_xlabel("running speed (cm/s)")
        # Show y-axis label only at the start of each row
        
        if roi % cols == 0:  
            ax.set_ylabel("dFoF")
        else:
            ax.set_yticklabels([])  # Hide y-axis labels
            
        #ax.set_ylabel("dFoF")
        ax.set_title(f'ROI #{roi+1}')

        x= np.array(run_speed_sampled).reshape((-1, 1))
        y= np.array(dFoF[roi, :])
        r_sq_s.append(linear_regression(x=x, 
                                        y=y, 
                                        ax=ax))
    # Set common y-limits
    pt.set_common_ylims(AX)
    
    # Hide any unused axes (if n_ori isn't a multiple of cols)
    for ax in AX[n_ori:]:
        ax.set_visible(False)
        
    return 0

def lr_dFoF_running_avg(run_speed_sampled, dFoF, r_sq_s):
    
    fig, AX = pt.figure(axes=(1, 1), hspace=2)  # Create a grid with (rows, cols)
    
    ax.scatter(run_speed_sampled, dFoF[:, :].mean(axis=0))
    ax.set_xlabel("running speed (cm/s)")
    ax.set_ylabel("dFoF")
    ax.set_title('Average all ROIs')

    x= np.array(run_speed_sampled).reshape((-1, 1))
    y= np.array(dFoF[roi, :])
    r_sq_s.append(linear_regression(x=x, 
                                    y=y, 
                                    ax=ax))
    return 0

def explained_var_roi(run_speed_sampled, dFoF, roi):
    x= np.array(run_speed_sampled).reshape((-1, 1))
    y= np.array(dFoF[roi, :])
    model = LinearRegression()
    model.fit(x,y)
    r_sq = model.score(x,y)
    fit = model.coef_*x + model.intercept_ 
    r_sq = model.score(x,y)
    
    
    return r_sq

# %% [markdown]
# ## DATA

# %% [markdown]
# ### Distribution dFoF as a function of running speed

# %%
base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments','my_experiments', 'All_NWBs')

# %% [markdown]
# ## Monitoring ketamine

# %%
fns = generate_file_paths(names_monitoring_keta, base_path)
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    r_sq_s = []
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    print(len(data.dFoF[0]))
    print(len(running_dFoF_sampled))
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF, r_sq_s= r_sq_s)
    r_sq = explained_var_roi(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF, roi = i)
    r_sq_s.append(r_sq)

# %%
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    n_ori = data.dFoF.shape[0]
    print(n_ori)
    x = np.arange(0,n_ori,1)
    plt.plot(x,r_sq_s)

# %% [markdown]
# ## 8ori2contrasts Ketamine

# %%
fns = generate_file_paths(names_8ori_keta, base_path)
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF)
    

# %% [markdown]
# ## 8ori_saline

# %%
fns = generate_file_paths(names_8ori_saline, base_path)
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF)
    

# %% [markdown]
# # NDNF 2022

# %%
base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments','NDNF-WT-Dec-2022', 'NWBs')

# %%
files_NDNF = ['2022_12_14-13-27-41.nwb', 
              '2022_12_15-18-13-25.nwb', '2022_12_15-18-49-40.nwb', 
              '2022_12_16-10-15-42.nwb','2022_12_16-11-00-09.nwb', '2022_12_16-12-03-30.nwb', 
              '2022_12_16-12-47-57.nwb', '2022_12_16-13-40-07.nwb','2022_12_16-14-29-38.nwb', 
              '2022_12_20-11-49-18.nwb', '2022_12_20-12-31-08.nwb','2022_12_20-13-18-42.nwb', 
              '2022_12_20-14-08-45.nwb', '2022_12_20-15-02-20.nwb']

# %%
fns = generate_file_paths(files_NDNF, base_path)

for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF)
    

# %%
base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments','SST-Ketamine-vs-Saline', 'Ketamine')
files_SST_keta = []
fns = generate_file_paths(files_SST_keta, base_path)
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF)

# %%
base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments','SST-Ketamine-vs-Saline', 'Saline')
files_SST_sal = []
fns = generate_file_paths(files_SST_sal, base_path)
for i in range(len(fns)):
    data = Data(fns[i], verbose=False)
    data.build_dFoF()
    data.t_dFoF[-1]
    data.build_pupil_diameter()
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
    lr_dFoF_running(run_speed_sampled=running_dFoF_sampled, dFoF = data.dFoF)
