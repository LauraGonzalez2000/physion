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
import pandas as pd
import os
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.dataframe import NWB_to_dataframe
import matplotlib.pylab as plt

import seaborn as sns
from physion.utils import plot_tools as pt



# %%
from physion.analysis.process_NWB import EpisodeData

import sys
sys.path.append('../scripts')
from distinct_rest_vs_active import compute_high_movement_cond

import numpy as np
import pandas as pd

sys.path.append('../../src/physion/analysis')
from cross_validation import TwoFold_train_test_split_basic

from sklearn.model_selection import train_test_split


# %% jupyter={"source_hidden": true}
def plot_dFoF_locomotion_all(all_episodes, 
                         all_HMcond, 
                         general=True, 
                         active=True, 
                         resting=True):

    fig, AX = plt.subplots(2, 1, figsize=(4, 2)) 
    fig.subplots_adjust(hspace=0.9)

    num_epi_all = 0
    num_roi_all = 0
    for i in range(len(all_episodes)):
        temp_epi = all_episodes[i].dFoF.shape[0]
        num_epi_all += temp_epi
        temp_roi = all_episodes[i].dFoF.shape[1]
        num_roi_all += temp_roi
    print(f"{len(all_episodes)} files")
    print(f"average of {num_epi_all} episodes ({np.sum(np.concatenate(all_HMcond))} active, {len(np.concatenate(all_HMcond))-np.sum(np.concatenate(all_HMcond))} resting)")
    print(f"average of {num_roi_all} ROIs")

    mean_dFoF = np.mean([ep.dFoF.mean(axis=(0, 1)) for ep in all_ep], axis=0)
    mean_running_speed = np.mean([ep.running_speed.mean(axis=0) for ep in all_ep], axis=0)
    
    mean_dFoF_active = np.mean([ep.dFoF[cond, :, :].mean(axis=(0, 1)) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    mean_running_active = np.mean([ep.running_speed[cond, :].mean(axis=0) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    
    mean_dFoF_rest = np.mean([ep.dFoF[~cond, :, :].mean(axis=(0, 1)) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
    mean_running_rest = np.mean([ep.running_speed[~cond, :].mean(axis=0) for ep, cond in zip(all_ep, all_HMcond) if np.any(cond)], axis=0)
  
    if general:
        AX[0].plot(all_ep[0].t, mean_dFoF, color='blue') 
        AX[1].plot(all_ep[0].t, mean_running_speed, color='blue')
    if active: 
        AX[0].plot(all_ep[0].t, mean_dFoF_active, color='orangered') 
        AX[1].plot(all_ep[0].t, mean_running_active, color="orangered")
    if resting: 
        AX[0].plot(all_ep[0].t, mean_dFoF_rest, color = 'grey') 
        AX[1].plot(all_ep[0].t, mean_running_rest, color = "grey")
             
    
    AX[0].set_ylabel('dFoF', fontsize=9)
    AX[1].set_ylabel('locomotion (cm/s)', fontsize=9)

    for ax in AX:
        ax.axvspan(0, 2, color='lightgrey')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.annotate('Visual stimulation', (0.30, 1), color='black', xycoords='axes fraction', va='top', fontsize=7)
        ax.tick_params(axis='both', labelsize=7, pad=1, direction='out', length=4, width=1)
        ax.grid(False)
        ax.tick_params(axis='both', which='both', bottom=True, left=True)
    
    return 0

# %% [markdown]
# # NDNF dataset

# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

# %% jupyter={"outputs_hidden": true}
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


# %%
plot_dFoF_locomotion_all(all_ep, 
                         all_HMcond, 
                         general=True, 
                         active=True, 
                         resting=True)

# %% jupyter={"outputs_hidden": true}
print(len(all_ep))
print(len(all_HMcond))

for i in range(14):
    print(all_ep[i].dFoF.shape)
    print(all_ep[i].running_speed.shape)
    print(all_HMcond[i].shape)
    print("")

# %% [markdown]
# ### Prepare data for the model

# %%
rows = []

for file_idx, file in enumerate(all_ep):
    
    for ep_idx in range(file.dFoF.shape[0]):
        
        dFoF_ = file.dFoF[ep_idx].mean(axis=0)
        running_speed_ = file.running_speed[ep_idx].mean(axis=0)
        all_HMcond_ = all_HMcond[file_idx][ep_idx]
        
        row = {'File idx': file_idx,
               'Episode idx': ep_idx,  #would be good to put episode ID, because here it doesn't make sense
               'dFoF_meanROIs': dFoF_.mean(axis=0), 
               'running_speed': running_speed_,
               'behav_state': all_HMcond_}

        for roi_idx in range(file.dFoF[ep_idx].shape[0]):
            row[f'dFoF_ROI_{roi_idx}'] = file.dFoF[ep_idx][roi_idx].mean(axis=0)
            
        rows.append(row)

df_unfolded = pd.DataFrame(rows)


# %%
df_unfolded

# %% [markdown]
# ### divide train and test sets

# %%
df_train, df_test = train_test_split(df_unfolded,
                                     test_size=0.2,
                                     stratify=df_unfolded['behav_state'],
                                     random_state=42)

# %%
#to make sure
print("Train:")
print(f"{df_train['behav_state'].value_counts()} \n")
print(df_train['behav_state'].value_counts(normalize=True))

print("\n\nTest:")
print(f"{df_test['behav_state'].value_counts()} \n")
print(df_test['behav_state'].value_counts(normalize=True))


# %% [markdown]
# ### create and fit the model

# %%
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

# %% [markdown]
# Mixed linear model

# %% [markdown]
# dFoF_meanROIs ~ running_speed  + (1 | File idx) <br>
# The model is trying to predict dFoF using the running_speed variable, and accounting for random shifts by file

# %%
model_mlm = smf.mixedlm("dFoF_meanROIs ~ running_speed",
                    df_train,
                    groups=df_train["File idx"])
result_mlm = model_mlm.fit()
print(result_mlm.summary())

# %%
df_test["_mlm_predicted_dFoF"] = result_mlm.predict(df_test)

# %% [markdown]
# Linear regression

# %%
model_lr = LinearRegression()

x = df_train["running_speed"].values.reshape(-1, 1)
y = df_train["dFoF_meanROIs"].values
result_lr = model_lr.fit(x, y)

df_test["lr_predicted_dFoF"] = model_lr.predict(df_test["running_speed"].values.reshape(-1, 1))

# %% [markdown]
# ## Evaluate the predictions

# %% [markdown]
# ### Mixed linear model

# %% [markdown]
# ### Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)

# %%
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df_test["dFoF_meanROIs"], df_test["mlm_predicted_dFoF"])
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

# %%
mse = mean_squared_error(df_test["dFoF_meanROIs"], df_test["lr_predicted_dFoF"])
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)

# %% [markdown]
# ### R² Score (Coefficient of Determination)

# %% jupyter={"outputs_hidden": true}
from sklearn.metrics import r2_score

r2 = r2_score(df_test["dFoF_meanROIs"], df_test["mlm_predicted_dFoF"])
print("Mixed linear model R² Score:", r2)


# %%
r2 = r2_score(df_test["dFoF_meanROIs"], df_test["lr_predicted_dFoF"])
print(f"Linear regression R²: {r2:.3f}")

# %% [markdown]
# ### Correlation (Pearson) Between True and Predicted

# %%
corr = np.corrcoef(df_test["dFoF_meanROIs"], df_test["mlm_predicted_dFoF"])[0, 1]
print("Correlation (Pearson):", corr)

# %%
corr = np.corrcoef(df_test["dFoF_meanROIs"], df_test["lr_predicted_dFoF"])[0, 1]
print("Correlation (Pearson):", corr)

# %% [markdown]
# ### Visually: 

# %%
plt.figure(figsize=(3, 3))
plt.scatter(df_test["dFoF_meanROIs"], df_test["mlm_predicted_dFoF"])
plt.plot([0, 10], [0, 10], 'r--')
plt.xlabel('True Values dFoF')
plt.ylabel('Predictions dFoF')
plt.annotate( f"n={len(df_test["dFoF_meanROIs"])}", xy=[1,9])

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.show()

# %%
plt.figure(figsize=(3, 3))
plt.scatter(df_test["dFoF_meanROIs"], df_test["lr_predicted_dFoF"])
plt.plot([0, 10], [0, 10], 'r--')
plt.xlabel('True Values dFoF')
plt.ylabel('Predictions dFoF')
plt.annotate( f"n={len(df_test["dFoF_meanROIs"])}", xy=[1,9])

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.show()

# %%

# %%

# %% [markdown]
# # Useless 

# %% jupyter={"outputs_hidden": true}
index = 8
filename = SESSIONS['files'][index]

df = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)

# %%
cv_indices = TwoFold_train_test_split_basic(df,spont_act_key='VisStim_grey-10min')

# %%
cv_indices

# %%
df

# %%
cv_indices['spont_train_sets']

# %%
md = smf.mixedlm("dFoF ~ running_speed", df, groups = df['behav_state'])
mdf = md.fit()
print(mdf.summary())

# %%

# %%

# %%

# %%

# %%

# %%
index = 8
filename = SESSIONS['files'][index]

# %%
df_ = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)

# %%
df_

# %%
data = Data(filename, verbose=False)
data.build_dFoF()
data.t_dFoF[-1]
running_dFoF_sampled = data.build_running_speed(specific_time_sampling=data.t_dFoF)
pupil_size_dFoF_sampled = data.build_pupil_diameter(specific_time_sampling=data.t_dFoF)

df = pd.DataFrame()
df['dFoF'] = data.dFoF.mean(axis=0)   #average of all ROIs
df['running_speed'] = running_dFoF_sampled
df['behav_state'] = df['running_speed'].apply(lambda speed: True if speed > 0.1 else False)
df['stim_cond'] = (~df_['VisStim_grey-10min'])

print(df)


# %%

# %%
# Set plot style
sns.set(style="whitegrid")

# Create the bar plot
plt.figure(figsize=(5, 2))
g = sns.catplot(data=df, 
                 kind = 'violin',
                 x='stim_cond', 
                 y='dFoF', 
                 hue='behav_state', 
                 palette=['grey', 'orangered'], 
                 linewidth=0.8, 
                 legend_out = True)

g.set_xticklabels(["No visual Stimulus", "Visual stimulus"])
new_title = 'Behavioral state'
g._legend.set_title(new_title)
new_labels = ['rest', 'active']
for t, l in zip(g._legend.texts, new_labels):
    t.set_text(l)
    

ax = g.ax  # Get the matplotlib axis
grouped = df.groupby(["stim_cond", "behav_state"]).size().reset_index(name='count')
for i, (x, hue) in enumerate(zip(grouped["stim_cond"], grouped["behav_state"])):
    print(x, hue)
    n = grouped.loc[(grouped["stim_cond"] == x) & (grouped["behav_state"] == hue), "count"].values[0]
    ax.text(i/2-0.2, 0, f"n={n}", ha="center", va="bottom", fontsize=12, color="black")




plt.show()

# %% [markdown]
# Statistics: Linear Mixed effect model

# %%
md = smf.mixedlm("dFoF ~ behav_state", df, groups=df['stim_cond'])
mdf = md.fit()
print(mdf.summary())

# %% [markdown]
# ## 2 Way-ANOVA

# %% [markdown]
# Assumptions
# 1) Homogeneity of variance (a.k.a. homoscedasticity)
# The variation around the mean for each group being compared should be similar among all groups. If your data don’t meet this assumption, you may be able to use a non-parametric alternative, like the Kruskal-Wallis test.
#
# 2) Independence of observations
# Your independent variables should not be dependent on one another (i.e. one should not cause the other). This is impossible to test with categorical variables – it can only be ensured by good experimental design. <br>
# In addition, your dependent variable should represent unique observations – that is, your observations should not be grouped within locations or individuals.<br>
# If your data don’t meet this assumption (i.e. if you set up experimental treatments within blocks), you can include a blocking variable and/or use a repeated-measures ANOVA.<br>
#
# 3) Normally-distributed dependent variable
# The values of the dependent variable should follow a bell curve (they should be normally distributed). If your data don’t meet this assumption, you can try a data transformation.
#

# %%
# Performing two-way ANOVA 
md = ols('dFoF ~ C(stim_cond) + C(behav_state) + C(stim_cond):C(behav_state)', data=df)
mdf = md.fit()
result = sm.stats.anova_lm(mdf, type=2)  
print(result) 

# %%

# %% [markdown]
# dFoF significantly affected by the behavioral state, stimulus condition, and the interaction between the 2?

# %%
all_ep = []
all_HMcond = []
df = pd.DataFrame()

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
    
    new_row = pd.DataFrame({
    'File ID': [dataIndex],
    'number of ROIs': [ep.dFoF.shape[1]],
    'number of episodes': [ep.dFoF.shape[0]],
    'number of active episodes': [np.sum(HMcond)], 
    'proportion of active episodes (%)': [(np.sum(HMcond)/ep.dFoF.shape[0])*100]})
    df = pd.concat([df, new_row], ignore_index=True)

# %%
print(df)

# %%
print(len(all_ep))
print(len(all_HMcond))
print(all_ep[0].dFoF.shape)
print(all_ep[0].running_speed.shape)
print(all_HMcond[0].shape)

df_test = pd.DataFrame()
df_test['dFoF'] = [all_ep[i].dFoF for i in range(len(all_ep))]
df_test['running_speed'] = [all_ep[i].running_speed for i in range(len(all_ep))]
df_test['behav_state'] = all_HMcond

# %%
df_test
