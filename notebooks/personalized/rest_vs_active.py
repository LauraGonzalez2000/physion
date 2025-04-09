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
#from physion.dataviz.raw import plot2 as plot_raw2, find_default_plot_settings
#from physion.dataviz.raw import plot_test
from physion.dataviz import tools as dv_tools

import matplotlib.pyplot as plt

base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments', 'NDNF-WT-Dec-2022', 'NWBs')

# %%
filename = os.path.join(base_path, "2022_12_14-13-27-41.nwb")
data = Data(filename, verbose=False)
data.build_dFoF(method_for_F0='sliding_percentile',verbose=False)

# %%
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
                           'color': '#2ca02c'}
           }

plot_raw(data=data, tlim=[0, data.t_dFoF[-1]], settings=settings)

# %%
data.build_running_speed
print(data.running_speed)

threshold = 0.5 #choose threshold

# %%
plt.plot(data.running_speed)
plt.axhline(y = threshold, color = 'r', linestyle = '-') 

# %%
running_bool = []

for instance in data.running_speed:
    if instance >= threshold:  
        running_bool.append(True)
    else : running_bool.append(False)

print(running_bool)

# %%



filename = os.path.join(base_path, "2022_12_14-13-27-41.nwb")
data = Data(filename, verbose=False)
data.build_dFoF(method_for_F0='sliding_percentile',verbose=False)
data.build_gaze_movement
data.build_running_speed



# %%
settings = {'Locomotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': '#1f77b4'},
             'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                           'color': '#2ca02c'}
           }


plot_raw(data, tlim=[0, data.t_dFoF[-1]], settings=settings)


# %%
def add_CaImaging(data, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., color='green',
                  subquantity='Fluorescence', 
                  roiIndices='all', dFoF_args={},
                  scale_side='left',
                  vicinity_factor=1, 
                  subsampling=1, 
                  name='[Ca] imaging',
                  annotation_side='left'):

    if (subquantity in ['dF/F', 'dFoF']) and (not hasattr(data, 'dFoF')):
        data.build_dFoF(**dFoF_args)
        
    if (type(roiIndices)==str) and roiIndices=='all':
        roiIndices = data.valid_roiIndices
        
    if color=='tab':
        COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
    else:
        COLORS = [str(color) for n in range(len(roiIndices))]

    i1, i2 = dv_tools.convert_times_to_indices(*tlim, data.Neuropil, axis=1)
    t = np.array(data.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]

    for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):

        ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+\
                fig_fraction_start # bottom position

        if (subquantity in ['dF/F', 'dFoF']):
            y = data.dFoF[ir, np.arange(i1,i2)][::subsampling]
            dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 1.,
                              ax_fraction_extent=fig_fraction/len(roiIndices),
                              ax_fraction_start=ypos,
                              color=color, 
                              scale_side=scale_side,
                             scale_unit_string=('%.0f$\\Delta$F/F' if (n==0) else ' '))
        else:
            y = data.rawFluo[ir,np.arange(i1,i2)][::subsampling]
            dv_tools.plot_scaled_signal(data, ax, t, y, tlim, 1.,
                   ax_fraction_extent=fig_fraction/len(roiIndices),
                   ax_fraction_start=ypos, color=color,
                   scale_side=scale_side,
                   scale_unit_string=('fluo (a.u.)' if (n==0) else ''))

        if annotation_side!='':
            dv_tools.add_name_annotation(data, ax, 
                    'roi #%i'%(ir+1), tlim, fig_fraction/len(roiIndices),
                                         ypos, color=color, 
                                         side=annotation_side)
        
        

def add_Locomotion(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2,
                   speed_scale_bar=1, # cm/s
                   scale_side='left',
                   color='#1f77b4', name='run. speed'):

    if not hasattr(data, 'running_speed'):
        data.build_running_speed()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.acquisition['Running-Speed'])
    x, y = data.t_running_speed[i1:i2][::subsampling], data.running_speed[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data, ax, x, y,
                                tlim, speed_scale_bar,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                color=color, scale_unit_string='%icm/s')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)




# %%
def plot_state(data, 
               tlim=[0,100], 
               settings={}, 
               figsize=(9,6), 
               Tbar=0., 
               zoom_area=None,
               ax=None,
               active=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    fig_fraction_full, fstart = np.sum([settings[key]['fig_fraction'] for key in settings]), 0

    for key in settings:
        settings[key]['fig_fraction_start'] = fstart
        settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
        fstart += settings[key]['fig_fraction']

    for key in settings:
        exec('add_%s(data, tlim, ax, **settings[key])' % key)

    # time scale bar
    if Tbar==0.:
        Tbar = np.max([int((tlim[1]-tlim[0])/30.), 1])

    #ax.plot([dv_tools.shifted_start(tlim), dv_tools.shifted_start(tlim)+Tbar], [1.,1.], lw=1, color='k')
    #ax.plot(data.running_speed)
    #ax.axhline(y = threshold, color = 'r', linestyle = '-') 
    
    ax.annotate((' %is' % Tbar if Tbar>=1 else  '%.1fs' % Tbar) ,
                [dv_tools.shifted_start(tlim), 1.02], color='k')#, fontsize=9)

    ax.axis('off')
    ax.set_xlim([dv_tools.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
    ax.set_ylim([-0.05,1.05])

    if zoom_area is not None:
        ax.fill_between(zoom_area, [0,0], [1,1],  color='k', alpha=.2, lw=0)

    return fig, ax


# %%
plot_state(data, tlim=[0, data.t_dFoF[-1]], settings=settings)

# %%
plt.plot(data.running_speed)

# %%

threshold_cond = (data.running_speed>threshold)

plt.plot(data.t_running_speed[~threshold_cond], data.running_speed[~threshold_cond])  

# %%
settings = {'Locomotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': '#1f77b4'},
             'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                           'color': '#2ca02c'}
           }


plot_raw2(data, tlim=[0, data.t_dFoF[-1]], settings=settings)

# %%
filename = os.path.join(base_path, "2022_12_14-13-27-41.nwb")
data = Data(filename, verbose=False)
data.build_dFoF(method_for_F0='sliding_percentile',verbose=False)
data.build_gaze_movement
data.build_running_speed

# %%
base_path = os.path.join(os.path.expanduser('~'), 'DATA','In_Vivo_experiments', 'SST-Ketamine-vs-Saline', 'All_NWBs')
filename = os.path.join(base_path, "2023_01_18-15-08-51.nwb")
data = Data(filename, verbose=False)
data.build_dFoF(method_for_F0='sliding_percentile',verbose=False)
data.build_gaze_movement
data.build_running_speed

# %%
data.metadata
