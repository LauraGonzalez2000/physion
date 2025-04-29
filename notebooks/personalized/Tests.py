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

from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
#from physion.dataviz.raw import plot_test
from physion.dataviz import tools as dv_tools

import matplotlib.pyplot as plt

from scipy import stats
from physion.analysis.process_NWB import EpisodeData

base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments','my_experiments', 'All_NWBs')

from physion.dataviz.episodes.trial_average import plot as plot_trial_average

import random


# %%
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

def process_data(data):
    data.build_dFoF(method_for_F0='sliding_percentile',
                    verbose=False)
    setattr(data, 'new', 0)
    
def get_dFoF(the_list):
    dFoF_all = []
    for data in DATA_monitoring_keta:
        dFoF_all.append(data.dFoF)
    return dFoF_all

def check_group(the_list): #prints (number ROIs, number datapoints)
    for i in range(0, len(the_list)):
        print(the_list[i].dFoF.shape)

def average_ROIs(the_list):
    list_mean_ROIs = []
    for i in range(0, len(the_list)):
        mean_ROIs = np.mean(the_list[i].dFoF, axis=0)
        list_mean_ROIs.append(mean_ROIs)
        #print(mean_ROIs)
        #print(mean_ROIs.shape)
    return list_mean_ROIs

def pad_list(n_array):
    max_length = max(len(arr) for arr in n_array)
    padded_arrays = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan) for arr in n_array])
    #print(padded_arrays.shape)
    return padded_arrays



# %%
names_monitoring_keta = ["2024_10_07-16-26-15.nwb", "2024_10_11-14-13-22.nwb", "2024_10_11-15-46-32.nwb", "2024_10_11-17-26-55.nwb"]
names_8ori_keta       = ["2024_09_12-14-57-34.nwb", "2024_09_12-15-24-47.nwb", "2024_09_12-15-50-12.nwb", "2024_10_07-15-03-40.nwb", 
                         "2024_10_07-17-18-53.nwb", "2024_10_11-14-57-27.nwb", "2024_10_11-16-44-26.nwb", "2024_10_11-18-24-27.nwb"]
names_8ori_saline     = ["2024_08_27-12-21-14.nwb", "2024_08_27-12-46-41.nwb"]

# %% [markdown]
# # Monitoring keta

# %%
fns = generate_file_paths(names_monitoring_keta, base_path)
data = Data(fns[0], verbose=False)

# %%
data.build_dFoF()
data.t_dFoF[-1]
data.build_pupil_diameter()

running_FaceCamera_sampled = data.build_running_speed(specific_time_sampling=data.t_rawFluo)
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
pt.figure(figsize=(2,2))

plt.scatter(data.dFoF[0,:], running_dFoF_sampled)
plt.xlabel("dFoF")
plt.ylabel("running speed (cm/s)")

# %% [markdown]
# ## NDNF dataset

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %%
index = 6
filename = SESSIONS['files'][index]
data = Data(filename,
            verbose=False)

# %%
data.build_dFoF()
data.t_dFoF[-1]
data.build_pupil_diameter()

running_FaceCamera_sampled = data.build_running_speed(specific_time_sampling=data.t_rawFluo)
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
pt.figure(figsize=(2,2))

roi_n = random.randint(0, data.dFoF.shape[0]-1) 
print('Dataset : NDNF')
print('File : ', index)
print('ROI : ', roi_n)
plt.scatter(running_dFoF_sampled, data.dFoF[roi_n,:])
plt.xlabel("dFoF")
plt.ylabel("running speed (cm/s)")
plt.axhline(0.1, color = 'black', label = 'threshold', linewidth=0.6)

# %%
data.build_dFoF()
data.t_dFoF[-1]
data.build_pupil_diameter()
data.build_facemotion()
data.facemotion
plt.plot(data.facemotion)
#running_pupil_diameter_sampled = data.build_pupil_diameter()
#running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
#pt.figure(figsize=(2,2))
#plt.scatter(data., running_dFoF_sampled)

# %% [markdown]
# ### example dFoF of one roi accros time

# %%
from matplotlib.pyplot import figure
fig = figure(figsize=(8, 2))
plt.plot(data.t_dFoF, data.dFoF[25])

# %%
print(data.dFoF.shape) # (rois, time samples)
n_episodes = data.dFoF.shape[1]
bins = 40
roi = 1

print(n_episodes)
#bin_size = int(np.round(n_episodes/bins))
bin_size = 8


fig, ax = pt.figure(figsize=(2, 2.5))
for i in range(bins):
    print("bin : [", i*bin_size," ; ", (1+i)*bin_size, "]")
    print(data.dFoF[roi,i*bin_size:(1+i)*bin_size])
    temp = data.dFoF[roi,i*bin_size:(1+i)*bin_size]
    ax.plot(data.t_dFoF[i*bin_size:(1+i)*bin_size] ,temp.mean(axis=0), color=pt.get_linear_colormap('y', 'b')(i/4))


# %%
data.metadata

# %%
data.init_visual_stim()

# %% [markdown]
# ## Episode visualization

# %% [markdown]
# An episode is the presentation of a stimulus. In this case there are 8 episodes (images of gratings with 8 different angles)

# %%
ep = EpisodeData(data,
                 prestim_duration=2,
                 protocol_id=0,
                 quantities=['dFoF'])

# %% [markdown]
# Specific ROI, 8 orientations

# %%
n_ori = 8
fig, AX = pt.figure(axes=(n_ori,1))
roi=15
for angle, ax in zip(np.unique(ep.angle), AX):
    pt.plot(ep.t, 
            ep.dFoF[ep.angle==angle, roi, :].mean(axis=0), 
            sy=ep.dFoF[ep.angle==angle, roi, :].std(axis=0), 
            ax=ax, title='%.1fdeg'%angle)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dFoF")
    ax.set_title('%.1f°' % angle)
    
pt.set_common_ylims(AX)


# %% [markdown]
# All ROI averaged, 8 orientations

# %%
n_ori = 8
fig, AX = pt.figure(axes=(n_ori,1))
for angle, ax in zip(np.unique(ep.angle), AX):
    pt.plot(ep.t, 
            ep.dFoF[ep.angle==angle, :, :].mean(axis=1).mean(axis=0), 
            sy=ep.dFoF[ep.angle==angle, :, :].mean(axis=1).std(axis=0), 
            ax=ax, title='%.1fdeg'%angle)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("dFoF")
    ax.set_title('%.1f°' % angle)
    
pt.set_common_ylims(AX)

#plot_trial_average(ep, column_key='angle')    #the same but visual is less good

# %% [markdown]
# All ROI averaged

# %%
print(ep.dFoF.shape) # (trials, rois, time samples)
n_episodes = ep.dFoF.shape[0]
#roi = random.randint(0, ep.dFoF.shape[1]-1)  #chosen randomly
roi=2
print(roi)

fig, ax = pt.figure(figsize=(2, 2.5))
temp = ep.dFoF[:,roi,:]
print(temp.shape)
ax.plot(ep.t, temp.mean(axis=0)) #mean of all the episodes
ax.axvspan(0,2, color='lightgrey')
ax.set_xlabel("time (s)")
ax.set_ylabel("dFoF")

# %% [markdown]
# ## Plot episodes averaging with bins

# %% [markdown]
# See evolution with time of dFoF

# %%
print(ep.dFoF.shape) # (trials, rois, time samples)
n_episodes = ep.dFoF.shape[0]
roi = 3 # random.randint(0, 26)

bin_size = 8
bins = int(n_episodes/bin_size)

norm_cond = (ep.t>-0.1) & (ep.t<0)
fig, ax = pt.figure(figsize=(2, 2.5))
print(roi)

for i in range(bins):
    #print("bin : [", i*bin_size," ; ", (1+i)*bin_size, "]")
    #temp = ep.dFoF[i*bin_size:(1+i)*bin_size,:,:].mean(axis=1).mean(axis=0)
    temp = ep.dFoF[i*bin_size:(1+i)*bin_size,roi,:].mean(axis=0)
    ax.plot(ep.t, temp-temp[norm_cond].mean(), color=pt.get_linear_colormap('y', 'b')(i/bins))  #align at stimulus presentation
    #ax.plot(ep.t, temp, color=pt.get_linear_colormap('y', 'b')(i/bins))
    ax.axvspan(0, 2, color='lightgrey')
ax.set_xlabel("time (s)")
ax.set_ylabel("dFoF")


# %%
def find_peak(ep, duration, roi=0):
    return ep.dFoF[:, roi, (ep.t>0) & (ep.t<duration) ].mean(axis=0).max()


# %%
find_peak(ep, duration=2, roi=0)

# %%
tStart = 0
tStop = data.metadata['presentation-duration']

# %%
data = Data(fns[2], verbose=False)

# %%
data.build_dFoF(method_for_F0='sliding_percentile',
                verbose=True, 
                with_correctedFluo_and_F0=True, 
                percentile=10,
                sliding_window=5*60) #10min

# %%
fig, AX = pt.figure(axes=(5,1))
for i in range(5):
    AX[i].plot(data.dFoF[i,:])

# %%
