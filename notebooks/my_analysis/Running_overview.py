# %% [markdown]
# # Running overview 

# %%
# load packages:
import sys, os
physion_path = os.path.join(os.path.expanduser('~'), 'Programming/In_Vivo/physion/src')
sys.path += [physion_path]
import physion

import numpy as np

from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from physion.analysis.behavior import population_analysis
import matplotlib.pyplot as plt

from physion.analysis.process_NWB import EpisodeData


#%% Load Data
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_final')
#datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)

#%% [markdown]
## Minimum time running 10% cutoff

#%%
run_path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_run')

#run_path = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs_run')
if not os.path.exists(run_path):
    os.makedirs(run_path)
    
#%%
fracs_running = []
running_speed_threshold = 0.1

for f, filename in enumerate(SESSIONS['files']):
        data = Data(filename, verbose=False)

        if (data.nwbfile is not None) and ('Running-Speed' in data.nwbfile.acquisition):
            speed = data.nwbfile.acquisition['Running-Speed'].data[:]
            frac = 100*np.sum(speed>running_speed_threshold)/len(speed)
            fracs_running.append(frac)
            if frac>=10:
                save_path = os.path.join(run_path, os.path.basename(filename))
                # Example: save NWB file copy
                with open(filename, 'rb') as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
                
            #print(f"file {filename} :\n fraction running {frac}")


x= np.arange(0,len(fracs_running),1)
y = fracs_running
threshold = 10
colors = ['r' if val > threshold else 'blue' for val in y]
plt.bar(x, y, color=colors)
plt.axhline(threshold, c='r')
plt.xlabel('Recording #')
plt.ylabel('Frac. running (%)')

#%%
import numpy as np

from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt

def population_analysis(FILES,
                        min_time_minutes=2,
                        exclude_subjects=[],
                        ax=None,
                        running_speed_threshold=0.1):

    times, fracs_running, subjects = [], [], []
    if ax is None:
        fig, ax = pt.figure((1, 1), figsize=(5,5))
    else:
        fig = None

    for f in FILES:
        data = Data(f, verbose=False)
        if (data.nwbfile is not None) and ('Running-Speed' in data.nwbfile.acquisition):
            speed = data.nwbfile.acquisition['Running-Speed'].data[:]
            max_time = len(speed)/data.nwbfile.acquisition['Running-Speed'].rate
            
            if (max_time>60*min_time_minutes) and (data.metadata['subject_ID'] not in exclude_subjects):
                times.append(max_time)
                fracs_running.append(100*np.sum(speed>running_speed_threshold)/len(speed))
                subjects.append(data.metadata['subject_ID'])

    i=-1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ax.bar(np.arange(1+i, i+1+np.sum(s_cond)),
               np.array(fracs_running)[s_cond]+1,
               width=.75, color=pt.plt.cm.tab10(c%10))
        i+=np.sum(s_cond)+1
    ax.bar([i+2], [np.mean(fracs_running)], yerr=[np.std(fracs_running)],
           width=1.5, color='grey')
    ax.annotate('frac. running:\n%.1f+/-%.1f %%' % (np.mean(fracs_running), np.std(fracs_running)),
                (i+3, np.mean(fracs_running)), xycoords='data')
    ax.set_xticks([])
    ax.set_xlabel('Recording #')
    ax.set_ylabel('Frac. running (%)')

    threshold = 10
    ax.axhline(threshold, color='r')


    ymax, i = ax.get_ylim()[1], -1
    for c, s in enumerate(np.unique(subjects)):
        s_cond = np.array(subjects)==s
        ax.annotate(s, (1+i, ymax), rotation=90, color=pt.plt.cm.tab10(c%10), xycoords='data')
        i+=np.sum(s_cond)+1
    return fig, ax

#%%
population_analysis(SESSIONS['files'])
# %%
