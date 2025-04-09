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


#stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols


# %%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
index = 8
filename = SESSIONS['files'][index]

# %%
df_ = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)

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

#df['pupil_size'] = pupil_size_dFoF_sampled
#df['aroused_state'] = df['pupil_size'].apply(lambda pupil_size: True if pupil_size > 2.9 else False)

df['stim_cond'] = (~df_['VisStim_grey-10min'])

print(df)


# %%
plt.figure(figsize=(8, 2))

fig, ax = pt.plt.subplots(1, 1, figsize=(10,2))

for i in range(len(df) - 1):
    if df['stim_cond'].iloc[i] == False:
        ax.axvspan(i, i+1, color='lightgrey', alpha=0.5)
        
ax.scatter(x = df['dFoF'].index, 
           y = df['dFoF'],  
           color = ['grey' if state == False else 'orangered' for state in df['behav_state']],
           s=1)
ax.set_xlabel("time (unit?)")
ax.set_ylabel("dFoF")

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
