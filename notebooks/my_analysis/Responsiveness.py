# %% [markdown]
# # Responsiveness

#%%
import sys, os
sys.path += ['../../src'] # add src code directory for physion
import physion

import numpy as np

from physion.analysis.read_NWB import Data
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

from physion.analysis.behavior import population_analysis
import matplotlib.pyplot as plt

from physion.analysis.process_NWB import EpisodeData

#%%
def study_responsiveness(SESSIONS, index, protocol):

    filename = SESSIONS['files'][index]
    data = Data(filename, verbose=False)
    data.build_dFoF()
   
    protocol = protocol
    stat_test_props = dict(interval_pre=[-1.,0],                                   
                        interval_post=[1.,2.],                                   
                        test='ttest')

    ep = EpisodeData(data,
                    protocol_name=protocol,
                    quantities=['dFoF'])

    print("dFoF shape : ", ep.dFoF.shape)
    print("varied parameters : ", ep.varied_parameters)

    values = []
    significance = []
    colors = []

    for roi_n in range(data.nROIs):
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)

        summary_data = ep.compute_summary_data(stat_test_props,
                                            exclude_keys=['repeat', 'angle', 'contrast'],
                                            response_significance_threshold=0.05,
                                            response_args=dict(roiIndex=roi_n))
        
        if summary_data['significant']: 
            if summary_data['value'] < 0: color = 'red'
            else: color = 'green'
            colors.append(color)
        else: 
            if summary_data['value'] < 0: color = 'pink'
            else: color = 'lime'
            colors.append(color)

        values.append(summary_data['value'].flatten())
        significance.append(summary_data['significant'].flatten())

    fig, AX = plt.subplots(1, 1, figsize=(1, 1))
    x= np.arange(0,len(values),1)
    y = [float(value) for value in values]
    AX.bar(x, y, color=colors)
    AX.set_xlabel('ROI #')
    AX.set_ylabel('Responsiveness')
    AX.set_title(f'Session #{index}')
    print(significance)
    true_indexes = [i for i, val in enumerate(significance) if val]
    false_indexes = [i for i, val in enumerate(significance) if not val]
    print(true_indexes)
    print(f'{len(true_indexes)} significant ROI out of {len(significance)} ROIs')
    return 0

#%%
def study_responsiveness_all(SESSIONS, protocol):

    Colors = []
    Values = []
    Significance = []

    for rec in range(len(SESSIONS['files'])):
        filename = SESSIONS['files'][rec]
        data = Data(filename, verbose=False)
        data.build_dFoF()

        protocol = "static-patch"
        stat_test_props = dict(interval_pre=[-1.,0],                                   
                            interval_post=[1.,2.],                                   
                            test='ttest')
        
        ep = EpisodeData(data,
                        protocol_name=protocol,
                        quantities=['dFoF'], 
                        verbose=False)
        
        values = []
        significance = []
        colors = []

        for roi_n in range(data.nROIs):
            
            summary_data = ep.compute_summary_data(stat_test_props,
                                                exclude_keys=['repeat', 'angle', 'contrast'],
                                                response_significance_threshold=0.05,
                                                response_args=dict(roiIndex=roi_n))
            
            if summary_data['significant']: 
                if summary_data['value'] < 0: color = 'red'
                else: color = 'green'
                colors.append(color)
            else: 
                if summary_data['value'] < 0: color = 'pink'
                else: color = 'lime'
                colors.append(color)

            values.append(summary_data['value'].flatten())
            significance.append(summary_data['significant'].flatten())
        
        Colors.append(colors)
        Values.append(values)
        Significance.append(significance)

    
    fig, AX = plt.subplots(5, 5, figsize=(9, 9))

    i,j = 0,0

    for rec in range(len(SESSIONS['files'])):

        x= np.arange(0,len(Values[rec]),1)
        y = [float(value) for value in Values[rec]]    

        AX[i][j].bar(x, y, color=Colors[rec])
        AX[i][j].set_xlabel('ROI #')
        AX[i][j].set_ylabel('Responsiveness')
        AX[i][j].set_title(f'Session #{rec}')

        # ---- PIE CHART ----
        # Count responsive cells
        n_total = len(Significance[rec])
        true_indexes = [i for i, val in enumerate(Significance[rec]) if val]
        false_indexes = [i for i, val in enumerate(Significance[rec]) if not val]
        
        excit_indexes = sum(1 for v in true_indexes if v > 0)
        inhib_indexes = sum(1 for v in true_indexes if v < 0)
        n_nonresponsive = len(false_indexes)
        n_total = len(Significance[rec])
        
        
        pie_counts = [excit_indexes, inhib_indexes, n_nonresponsive]
        pie_colors = ['g', 'r', 'gray']
        
        # Add as inset
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(AX[i][j], width="30%", height="30%", loc='upper right')
        ax_inset.pie(pie_counts, colors=pie_colors)#, autopct='%1.0f%%')
        #ax_inset.set_title('Responsive cells', fontsize=8)

        #update position next graph
        if j<4:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    for idx in range(len(SESSIONS['files']), len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0


############################################################################################################
############################## PLOT PER SESSION and PER ROI ################################################
############################################################################################################

########################################### STATIC PATCH     ###############################################
#%%
protocol="static-patch"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness(SESSIONS=SESSIONS, index=0, protocol=protocol)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)


########################################### DRIFTING GRATING ###############################################
#%%
protocol="drifting-grating"
#%% DATA ALL
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% DATA CENTERED
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%% YANN'S DATA
protocol="drifting-gratings"
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)



############################################# QUICK SPATIAL MAPPING #########################################
#%%
protocol="quick-spatial-mapping"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_centered')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
study_responsiveness_all(SESSIONS=SESSIONS, protocol=protocol)



############################################################################################################
############################################ Plot per protocol #############################################
############################################################################################################
#%%
def responsiveness_sessions_vs_protocols(SESSIONS, protocols):
    Values_m = []
    X = []

    for p, protocol in enumerate(protocols):
        
        values_per_file = []
        significance_per_file = []
        
        for f, filename in enumerate(SESSIONS['files']):
            data = Data(filename, verbose=False)
            data.build_dFoF()
            stat_test_props = dict(interval_pre=[-1.,0],                                   
                                interval_post=[1.,2.],                                   
                                test='ttest')
            ep = EpisodeData(data,
                            protocol_name=protocol,
                            quantities=['dFoF'], 
                            verbose=False)
        
            summary_data = ep.compute_summary_data(stat_test_props,
                                                exclude_keys=['repeat', 'angle', 'contrast'],
                                                response_significance_threshold=0.05,
                                                response_args={})
            

            values_per_file.append(summary_data['value'].flatten())
            significance_per_file.append(summary_data['significant'].flatten())
        
        X.append(np.arange(0,len(SESSIONS['files']),1))
        Values_m.append(values_per_file)

    # ## plot
   
    i,j = 0,0

    fig, AX = plt.subplots(2, 4, figsize=(10, 5))

    for p, protocol in enumerate(protocols):
        try: 
            AX[i][j].bar(X[p], np.array(Values_m[p]).flatten())
        except Exception as e:
            AX[i][j].bar(X[p], np.array(np.mean(Values_m[p], axis=1)).flatten())
        AX[i][j].axhline(0.0, c='black', linewidth = 0.5)
        AX[i][j].set_xlabel('File #')
        AX[i][j].set_title(f'{protocol}')
        AX[i][j].set_xticks(X[p])

        if j==0:
            AX[i][j].set_ylabel('responsiveness')

        if protocol!='Natural-Images-4-repeats':
            AX[i][j].set_title(f'{protocol}')
        else: 
            AX[i][j].set_title('Natural-Images')

        #update position next graph
        if j<3:
            j+=1
        else: 
            i+=1
            j=0

    AX = AX.flatten()
    n_files = len(protocols)
    print(n_files)
    for idx in range(n_files, len(AX)):
        fig.delaxes(AX[idx])
    fig.tight_layout()
    return 0

#%% MY DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
#%%
protocols = ['static-patch', 'drifting-grating' ,'looming-stim',
             'Natural-Images-4-repeats','moving-dots',
             'drifting-surround','quick-spatial-mapping']
#%%
responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
#######################################################################################################################
#%% YANN's DATA
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
#%%
protocols = [['static-patch', 'drifting-gratings', 'looming-stim',
              'Natural-Images-4-repeats', 'moving-dots', 
              'random-dots']]
# %%
responsiveness_sessions_vs_protocols(SESSIONS=SESSIONS, protocols=protocols)
########################################################################################################################