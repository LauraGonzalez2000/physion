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

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','my_experiments','All_NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %%
dataIndex = 4
data = Data(SESSIONS['files'][dataIndex], verbose=False)
data.build_dFoF(verbose=False)
data.build_running_speed()
#print(data.pupil_diameter.shape)
print(data.running_speed.shape)

# %%
all_ep = []
all_HMcond = []

for dataIndex in range(len(SESSIONS['files'])):
    data = Data(SESSIONS['files'][dataIndex], verbose=False)
    data.build_dFoF(verbose=False)
    ep = EpisodeData(data,
                 prestim_duration=0,
                 protocol_id=0,
                 quantities=['dFoF', 'running_speed'])
    all_ep.append(ep)

    HMcond = compute_high_movement_cond(ep, 
                                    pupil_threshold = 0.29, 
                                    running_speed_threshold = 0.1, 
                                    metric = 'locomotion')
    all_HMcond.append(HMcond)

all_HMcond = np.concatenate(all_HMcond)

# %%
print(f"{len(SESSIONS['files'])} files")

plot_dFoF_locomotion_all(all_ep, 
                         all_HMcond, 
                         roi_n=None, 
                         episode_n = None, 
                         general=True, 
                         active=True, 
                         resting=True)

# %%

# %%

# %%
