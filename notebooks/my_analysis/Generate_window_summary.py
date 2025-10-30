# %% [markdown]
# # Generate PNG summary

#%%
import os, sys
sys.path += ['../../src'] # add src code directory for physion

import numpy as np
sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'In_Vivo', 'physion', 'src'))

import physion.utils.plot_tools as pt
from physion.intrinsic.tools import *
from physion.intrinsic.analysis import RetinotopicMapping
import matplotlib.pylab as plt
from PIL import Image

from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from PDF_layout import PDF4


# %% [markdown]
# ## Load data

# %%
dataFolder = os.path.join(os.path.expanduser('~'), 'DATA', 
                        'physion_Demo-Datasets', 'PV-WT', 'retinotopic_mapping',
                        'PVTOM_BB_5')

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','Processed', 'intrinsic_img', '07_05_2025', '15-09-29')

# retinotopic mapping data
maps = np.load(os.path.join(datafolder, 'raw-maps.npy') , 
                allow_pickle=True).item()
#maps = dict(np.load(os.path.join(dataFolder, 'raw-maps.npz') , 
#               allow_pickle=True))

#%%
# vasculature picture
imVasc = np.array(Image.open(os.path.join(dataFolder, 'vasculature.tif')))
fig1, ax = pt.figure(ax_scale=(2,2))
ax.imshow(imVasc**1, cmap=plt.cm.gray) 
plt.axis('off');

# %% [markdown]
# # Retinotopic Maps

# %%
fig2 = plot_retinotopic_maps(maps, map_type='altitude');

# %%
fig3 = plot_retinotopic_maps(maps, map_type='azimuth');

# %% [markdown]
# # Perform Segmentation

# %%
data = build_trial_data(maps)
data['vasculatureMap'] = imVasc[::int(imVasc.shape[0]/data['aziPosMap'].shape[0]),\
                                ::int(imVasc.shape[1]/data['aziPosMap'].shape[1])]
#%%
segmentation_params={'phaseMapFilterSigma': 2.,
                    'signMapFilterSigma': 1.,
                    'signMapThr': 0.8,
                    'eccMapFilterSigma': 10.,
                    'splitLocalMinCutStep': 5.,
                    'mergeOverlapThr': 0.4,
                    'closeIter': 3,
                    'openIter': 3,
                    'dilationIter': 15,
                    'borderWidth': 1,
                    'smallPatchThr': 100,
                    'visualSpacePixelSize': 0.5,
                    'visualSpaceCloseIter': 15,
                    'splitOverlapThr': 1.1}

data['params'] = segmentation_params
trial = RetinotopicMapping.RetinotopicMappingTrial(**data)
trial.processTrial(isPlot=True) # TURN TO TRUE TO VISUALIZE THE SEGMENTATION STEPS
fig4 = trial._getSignMap(onlySMplot=True)

# %%
fig5, ax = pt.figure(ax_scale=(5,5))
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.imshow(imVasc, cmap=pt.plt.cm.gray, 
          vmin=imVasc.min(), vmax=imVasc.max(), 
          extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.axis('off');
#fig.savefig(os.path.join(os.path.expanduser('~'), # 'Desktop', 'fig.svg'))

# %% [markdown]
# # Summary Plot

# %%
fig, AX = pt.figure(axes=(5,1), 
                    ax_scale=(.9*2,.9*3), 
                    wspace=0.4)

for map_type, ax, bounds in zip(['altitude', 'azimuth'], AX[:2],
                                [[-50, 50],[-60, 60]]):
    im = ax.imshow(maps['%s-retinotopy' % map_type], cmap=plt.cm.PRGn,\
                vmin=bounds[0], vmax=bounds[1])
    fig.colorbar(im, ax=ax,
                label='visual e (deg.)')
    ax.set_title('%s map' % map_type)

ax = AX[2]
im = ax.imshow(trial.signMapf, cmap='jet', vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label='phase gradient sign')
ax.set_title('sign map')

ax=AX[-2]
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])
h = RetinotopicMapping.plotPatches(trial.finalPatches, plotaxis=ax)
ax.set_title('w/ vasculature')

ax = AX[-1]
h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, 
        vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])

for ax in AX:
    ax.axis('equal')
    ax.axis('off')    

fig.savefig(os.path.join(os.path.expanduser('~'), 
                        'Desktop', 'fig.svg'))
# %% [markdown]
### Identify the Center (<20˚) of V1
# 
# %%

# Based on the excentricity center
fig6, ax = pt.figure(ax_scale=(2,5))
m = 0*maps['altitude-retinotopy']
cond = ( np.abs(maps['altitude-retinotopy'])<10) &\
          ( np.abs(maps['azimuth-retinotopy'])<10)

# NOT WORKING
centerV1 = trial.finalPatches['patch01'].getCenter()
plt.plot([centerV1[1]], [centerV1[0]], 'ro', 
         alpha=0.3, ms=25)
m[~cond] = 1

h = RetinotopicMapping.plotPatches(trial.finalPatches, 
                                   alpha=0, plotaxis=ax)
ax.imshow(imVasc, cmap=plt.cm.gray, 
        vmin=imVasc.min(), vmax=imVasc.max(), 
        extent=[*ax.get_xlim(), *ax.get_ylim()])
ax.axis('off')
ax.axis('equal');

#%%

def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, cell_type):
    try: 
        pdf1 = PDF4()
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6)
        fig_p1 = pdf1.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Window_Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1

        print("Window PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual window PDF file : {e}")

    return 0

#%%
create_PDF(dict_annotation={}, fig1, fig2, fig3, fig4, fig5, fig6, cell_type='NDNF')

#####################


from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz.imaging import show_CaImaging_FOV

from scipy import stats
base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'NWBs')
import random
from physion.dataviz.raw import plot as plot_raw
from PDF_layout import PDF, PDF2, PDF3, PDF3_
from matplotlib.backends.backend_pdf import PdfPages

from General_overview_episodes import plot_dFoF_per_protocol, plot_dFoF_per_protocol2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random

import time

#%%
def find_available_settings(data, debug=False):

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
                                                   'color': '#2ca02c'}}

    

    attributes = ['facemotion', 'pupil_diameter', 'dFoF']
    
    missing = [attr for attr in attributes if not hasattr(data, attr)]  # for objects

    if debug:
        if missing:
            print(f"Missing attributes: {missing}")
        else:
            print("All attributes exist")
    
    if missing==['pupil_diameter']:
        if debug:
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
                                                   'color': '#2ca02c'}}
    
    if missing==['facemotion']:
        if debug:
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
                                               'color': '#2ca02c'}}
        
    if missing==['dFoF']:
        if debug:
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
        if debug:
            print('facemotion and pupil diameter missing')
        settings = {'Locomotion': {'fig_fraction': 1,
                                               'subsampling': 1,
                                               'color': '#1f77b4'},
                    'CaImaging': {'fig_fraction': 10,
                                               'subsampling': 1,
                                               'subquantity': 'dF/F',
                                               'color': '#2ca02c'}}
        
    return settings
#%%
def figure_to_array(fig):
            """Convert a matplotlib Figure to a NumPy array"""
            canvas = FigureCanvas(fig)
            canvas.draw()
            buf = canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            fig_arr = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            return fig_arr
#%%
def plot_responsiveness_per_protocol(ep, nROIs, AX, idx, p):
    
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(nROIs):
        t0 = max([0, ep.time_duration[0]-1.5])
        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
                                                   exclude_keys=['repeat'],
                                                   #exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
                                                   response_significance_threshold=0.05,
                                                   response_args=dict(roiIndex=roi_n))
    
        session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
        session_summary['value'].append(roi_summary_data['value'][0])

    resp_cond = np.array(session_summary['significant'])

    pos_cond = resp_cond & ([session_summary['value'][i]>0 for i in range(len(session_summary['value']))])
    neg_cond = resp_cond & ([session_summary['value'][i]<0 for i in range(len(session_summary['value']))])

    print(f'{np.sum(resp_cond)} significant ROI ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) out of {len(session_summary['significant'])} ROIs')

    pos_frac = np.sum(pos_cond)/nROIs
    neg_frac = np.sum(neg_cond)/nROIs
    ns_frac = 1-pos_frac-neg_frac

    colors = ['green', 'red', 'grey']

    pt.pie(data=[pos_frac, neg_frac, ns_frac], ax = AX[idx], COLORS = colors)#, pie_labels = ['%.1f%%' % (100*pos),'%.1f%%' % (100*neg), '%.1f%%' % (100*ns)] )
    
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    pt.annotate(AX[idx], '+ resp=%.1f%%' % (100*pos_frac), (1, 0), ha='right', va='top')
    pt.annotate(AX[idx], '- resp=%.1f%%' % (100*neg_frac), (1, -0.2), ha='right', va='top')

    pt.annotate(AX[0], f'{nROIs} ROIs', (1, -0.4), ha='right', va='top')
    return 0
#%%
def plot_barplot_per_protocol(ep, AX, idx, p, subplots_n):
        
    t0 = max([0, ep.time_duration[0]-1.5])
    stat_test_props = dict(interval_pre=[-1.5,0],                                   
                            interval_post=[t0, t0+1.5],                                   
                            test='ttest', 
                            sign='both')
    
    summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
                                            exclude_keys=['repeat'],
                                            response_significance_threshold=0.05,
                                            response_args={})
    
    mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

    # Pad or truncate to 9 elements
    target_len = subplots_n #generalize
    if len(mean_vals) < target_len:
        mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
    else:
        mean_vals = mean_vals[:target_len]


    x = np.arange(target_len) #generalize
    AX[idx].bar(x, mean_vals, alpha=0.8, capsize=4)
    AX[idx].set_xticks(x)
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    AX[idx].axhline(0, color='black', linewidth=0.8)

    if idx==0:
        AX[0].set_ylabel('variation dFoF')

    return 0
#%%
def get_roiIndex(data, type='pos'):
    session_summary = {'significant':[], 'value':[]}

    for roi_n in range(data.nROIs):
        ep = EpisodeData(data,
                        protocol_name='Natural-Images-4-repeats',
                        quantities=['dFoF'], 
                        verbose=False)

        t0 = max([0, ep.time_duration[0]-1.5])
        stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                interval_post=[t0, t0+1.5],                                   
                                test='ttest', 
                                sign='both')

        roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
                                                    #exclude_keys=['repeat'],
                                                    exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
                                                    response_significance_threshold=0.05,
                                                    response_args=dict(roiIndex=roi_n))
        
        print(roi_summary_data)
        session_summary['significant'].append(bool(roi_summary_data['significant']))
        session_summary['value'].append(roi_summary_data['value'])

    for i in range(len(session_summary['significant'])):

        resp_cond = np.array(session_summary['significant'][i])
        pos_cond = resp_cond & (roi_summary_data['value'][0]>0)
        neg_cond = resp_cond & (roi_summary_data['value'][0]<0)

        pos_roi = np.arange(data.nROIs)[pos_cond]
        neg_roi = np.arange(data.nROIs)[neg_cond]
        ns_roi = np.arange(data.nROIs)[~resp_cond]


    resp_cond = np.array(session_summary['significant'])
    pos_cond = resp_cond & (roi_summary_data['value'][0]>0)
    neg_cond = resp_cond & (roi_summary_data['value'][0]<0)

    pos_roi = np.arange(data.nROIs)[pos_cond]
    neg_roi = np.arange(data.nROIs)[neg_cond]
    ns_roi = np.arange(data.nROIs)[~resp_cond]

    found = True

    if type=='pos':
        if len(pos_roi) > 0:
            roiIndex = random.choice(np.arange(data.nROIs)[pos_cond])
        else:
            print("No positive ROIs found — choosing from non-significant set instead.")
            found = False
            roiIndex = random.choice(ns_roi)

    
    elif type=='neg':
        if len(neg_roi) > 0:
            roiIndex = random.choice(np.arange(data.nROIs)[neg_cond])
        else:
            print("No negative ROIs found — choosing from non-significant set instead.")
            roiIndex = random.choice(ns_roi)
            found = False

    elif type=='ns':
        roiIndex = random.choice(ns_roi) 

    return roiIndex, found
#%%
def generate_figures(data_s, cell_type='nan', subplots_n=9, data_type = 'Sofia'):
    start_time = time.time()
    # Generate figures per session 

    for idx, data in enumerate(data_s):

        dict_annotation = {
            'name': data.filename,
            'Subject_ID': data.metadata['subject_ID'],
            'protocol': data.metadata['protocol']
        }

        settings = find_available_settings(data)
        protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min')]

        ################################# FIGURES ###############################################################
        
        fig1, AX1 = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[0])
        show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX1[1])
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2,  ax=AX1[2])
        
        if hasattr(data, "dFoF") and data.dFoF is not None and len(data.dFoF) > 0:
            '''
            data_cropped = data
            
            max_n = 10
            n_rois_total = len(data_cropped.dFoF)

            if n_rois_total > max_n:
                idx = np.random.choice(n_rois_total, max_n, replace=False)
                data_cropped.dFoF = data_cropped.dFoF[idx, :]
            '''
            if data_type=='Sofia':
                fig2, _ = plot_raw(data, 
                                    tlim=[0, data.t_dFoF[-1]], 
                                    settings=settings, 
                                    figsize=(9,3),
                                    zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])],
                                    grey=True, 
                                    black=True,
                                    grey_co=[638, 1238],
                                    black_co=[466, 586])
            else:
                fig2, _ = plot_raw(data, 
                                    tlim=[0, data.t_dFoF[-1]], 
                                    settings=settings, 
                                    figsize=(9,3),
                                    zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                                ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
            
            fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                            settings=settings, figsize=(9,3))
            
            fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]],
                            settings=settings, figsize=(9,3))
            
              
        fig5, _ = plot_dFoF_per_protocol(data_s=[data], protocols=protocols, subplots_n=subplots_n)
        roiIndex, found = get_roiIndex(data, type='pos')
        fig6, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        roiIndex, found = get_roiIndex(data, type='neg')
        fig7, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        roiIndex, found = get_roiIndex(data, type='ns')
        fig8, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roiIndex, found=found)
        fig9, AX9 = pt.figure(axes = (len(protocols),1))
        fig10, AX10 = pt.figure(axes = (len(protocols),1))
  
        nROIs = data.nROIs

        for idx, p in enumerate(protocols):
            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
            plot_responsiveness_per_protocol(ep, nROIs, AX9, idx, p)
            plot_barplot_per_protocol(ep, AX10, idx, p, subplots_n)
        
        fig1 = figure_to_array(fig1)
        fig2 = figure_to_array(fig2)
        fig3 = figure_to_array(fig3)
        fig4 = figure_to_array(fig4)
        fig5 = figure_to_array(fig5)
        fig6 = figure_to_array(fig6)
        fig7 = figure_to_array(fig7)
        fig8 = figure_to_array(fig8)
        fig9 = figure_to_array(fig9)
        fig10 = figure_to_array(fig10)

        create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type)

    elapsed = time.time() - start_time
    print(f"Execution time: {elapsed:.2f} seconds")
        
    return 0    
#%%
def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type):
    try: 
        pdf1 = PDF()
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4)
        fig_p1 = pdf1.fig

        pdf2 = PDF2()
        pdf2.fill_PDF2(fig5)
        fig_p2 = pdf2.fig

        pdf3 = PDF3()
        pdf3.fill_PDF3(fig6, fig7, fig8, fig9, fig10)
        fig_p3 = pdf3.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2
                pdf.savefig(fig_p3, dpi=300, bbox_inches="tight")  # Page 3

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################
#%%
def plot_responsiveness2_per_protocol(data_s, AX,idx,p, type='means'):
    pos_s = []
    neg_s = []

    resp_cond_s = []
    pos_cond_s = []
    neg_cond_s = []

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
        
        sig_list = []
        val_list = []
        nROIs = []

        for roi_n in range(data.nROIs):

            t0 = max([0, ep.time_duration[0]-1.5])
            stat_test_props = dict(
                interval_pre=[-1.5,0],
                interval_post=[t0, t0+1.5],
                test='ttest',
                sign='both'
            )
            roi_summary_data = ep.compute_summary_data(
                stat_test_props=stat_test_props,
                exclude_keys=list(ep.varied_parameters.keys()),
                response_significance_threshold=0.05,
                response_args=dict(roiIndex=roi_n)
            )

            sig_list.append(bool(roi_summary_data['significant'][0]))
            val_list.append(roi_summary_data['value'][0])
            nROIs.append(data.nROIs)

        sig_arr = np.array(sig_list)
        val_arr = np.array(val_list)

        # ✅ Compute per-ROI positive/negative significance
        resp_cond = sig_arr
        pos_cond = sig_arr & (val_arr > 0)
        neg_cond = sig_arr & (val_arr < 0)

        resp_cond_s.append(resp_cond)
        pos_cond_s.append(pos_cond)
        neg_cond_s.append(neg_cond)

        # Compute per-session proportions
        pos = np.sum(pos_cond) / len(sig_arr)
        neg = np.sum(neg_cond) / len(sig_arr)

        pos_s.append(pos)
        neg_s.append(neg)

    if type== 'means':
        final_pos = np.mean(pos_s)
        final_neg = np.mean(neg_s)
        final_ns = 1 - final_pos - final_neg
        AX[0].annotate('average over %i sessions ,   mean$\\pm$SEM across sessions' % len(data_s),
                               (1, -0.6), xycoords='axes fraction')
        
        sem = stats.sem([pos_s, neg_s], axis=1) 

        pt.annotate(AX[idx], 'Pos= %.1f ± %.1f %%' % (100 * final_pos, 100 *sem[0]),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX[idx], 'Neg= %.1f ± %.1f %%' % (100 * final_neg, 100 *sem[1]),
                    (1, -0.2), ha='right', va='top', fontsize=6)
        

    elif type == 'ROI':
        pos_cond_s = np.concatenate(pos_cond_s)
        neg_cond_s = np.concatenate(neg_cond_s)
    
        final_pos = np.mean(pos_cond_s)
        final_neg = np.mean(neg_cond_s)
        final_ns = 1 - final_pos - final_neg
        AX[0].annotate('average over %i ROIs' % np.sum(nROIs),
                               (1, -0.6), xycoords='axes fraction')
        
        pt.annotate(AX[idx], 'Pos= %.1f %%' % (100 * final_pos),
                (1, 0), ha='right', va='top', fontsize=6)
        pt.annotate(AX[idx], 'Neg= %.1f %%' % (100 * final_neg),
                    (1, -0.2), ha='right', va='top', fontsize=6)

    
    pt.pie(data=[final_pos, final_neg, final_ns],
        ax=AX[idx],
        COLORS=['green', 'red', 'grey'])
    

    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    
    return 0
#%%
def plot_barplot2_per_protocol(data_s, AX,idx,  p, subplots_n):
    mean_vals_s = []  # store per-session mean responses

    for data in data_s:
        ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])

        t0 = max([0, ep.time_duration[0] - 1.5])
        stat_test_props = dict(
            interval_pre=[-1.5, 0],
            interval_post=[t0, t0 + 1.5],
            test='ttest',
            sign='both')

        summary_data = ep.compute_summary_data(
            stat_test_props=stat_test_props,
            exclude_keys=['repeat'],
            response_significance_threshold=0.05,
            response_args={})

        # Extract ROI mean values
        mean_vals = [float(np.ravel(v)[0]) if np.size(v) > 0 else np.nan for v in summary_data['value']]

        # Pad/truncate to 5 elements
        target_len = subplots_n
        mean_vals = (mean_vals + [np.nan] * target_len)[:target_len]

        mean_vals_s.append(mean_vals)

    # Compute session-aggregated mean and SEM
    values = np.nanmean(mean_vals_s, axis=0)
    yerr = stats.sem(mean_vals_s, axis=0, nan_policy='omit')

    # Plot directly
    x = np.arange(len(values))
    AX[idx].bar(
        x, values, yerr=yerr,
        alpha=0.8, capsize=0,
        error_kw=dict(linewidth=0.6)
    )
    AX[idx].set_xticks(x)
    AX[idx].set_title(f'{p.replace('Natural-Images-4-repeats','natural-images')}')
    AX[idx].axhline(0, color='black', linewidth=0.8)
    
    if idx==0:
        AX[0].set_ylabel('variation dFoF')
    
    return 0
#%%
def generate_figures_GROUP(data_s, subplots_n):
    start_time = time.time()  

    protocols = [p for p in data_s[0].protocols 
                        if (p != 'grey-10min') and (p != 'black-2min')]

    fig1, _     = plot_dFoF_per_protocol(data_s=data_s, protocols=protocols)

    elapsed = time.time() - start_time
    print(f"Fig 1 ok: {elapsed:.2f} seconds")

    fig2, AX2  = pt.figure(axes = (len(protocols),1))
    fig3, AX3  = pt.figure(axes = (len(protocols),1))
    fig4, AX4 = pt.figure(axes = (len(protocols),1))

    for idx, p in enumerate(protocols):
        plot_responsiveness2_per_protocol(data_s, AX2, idx, p, type='ROI')
        plot_responsiveness2_per_protocol(data_s, AX3, idx, p, type='means')
        plot_barplot2_per_protocol(data_s, AX4, idx, p, subplots_n)

    fig1 = figure_to_array(fig1)
    fig2 = figure_to_array(fig2)
    fig3 = figure_to_array(fig3)
    fig4 = figure_to_array(fig4)

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")
    return fig1, fig2, fig3, fig4
#%%
def create_group_PDF(fig1, fig2, fig3, fig4, cell_type):
    try: 
        
        pdf1 = PDF2()
        pdf1.fill_PDF2(fig1)
        fig_p1 = pdf1.fig
        
        pdf2 = PDF3_()
        pdf2.fill_PDF3(fig2, fig3, fig4)
        fig_p2 = pdf2.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/GROUP_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2

        print("GROUP PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating GROUP PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################
##################################################################################################################
# %% [markdown]
# # Generate final figures
#%% [markdown]
# ## YANN DATASET

#%%
'''
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs-test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for idx, filename in enumerate(SESSIONS['files']):
    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)
'''
#%% [markdown]
# ## All individual files
#%%
'''
generate_figures(data_s, cell_type='NDNF_YANN', subplots_n=5, data_type = 'Yann')
'''
#%% [mardown]
# ## GROUPED ANALYSIS
#%%
#fig1, fig2, fig3, fig4 = generate_figures_GROUP(data_s, subplots_n=5)
#create_group_PDF(fig1, fig2, fig3, fig4, 'NDNF_YANN')

##################################################################################################################################
##################################################################################################################################

#%% [markdown]
# ## NDNF CRE BATCH 1

#%%

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','Processed', 'intrinsic_img', '07_05_2025')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8}

data_s = []
for idx, filename in enumerate(SESSIONS['files']):

    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

#%% [markdown]
# ## All individual files
#%%

generate_figures(data_s, cell_type='NDNF', subplots_n=9, data_type = 'Sofia')

#%% [mardown]
# ## GROUPED ANALYSIS
#%%
'''
fig1, fig2, fig3, fig4 = generate_figures_GROUP(data_s, subplots_n=9)
create_group_PDF(fig1, fig2, fig3, fig4, 'NDNF')
'''
##############################################################################################################
##############################################################################################################
