# %% [markdown]
# # Generate PNG summary

#%%
import os, sys
sys.path += ['../../src'] # add src code directory for physion

import numpy as np
sys.path.append(os.path.join(os.path.expanduser('~'), 'Programming', 'In_Vivo', 'physion', 'src'))

from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.imaging import show_CaImaging_FOV
from physion.dataviz import tools as dv_tools
from physion.dataviz.episodes.trial_average import plot as plot_trial_average
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
from physion.dataviz.imaging import show_CaImaging_FOV

import matplotlib.pyplot as plt
from scipy import stats
base_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'NWBs')
import random
from PIL import Image, ImageDraw, ImageFont
from PIL import Image

from PDF_layout import PDF, PDF2, PDF3

from matplotlib.backends.backend_pdf import PdfPages

from General_overview_episodes import plot_dFoF_per_protocol, plot_dFoF_per_protocol2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import random

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
def plot_responsiveness(data, protocols):
    
    fig, AX = pt.figure(axes = (len(protocols),1))
    
    for idx, p in enumerate(protocols):
        
        ep = EpisodeData(data,
                    protocol_name=p,
                    quantities=['dFoF'])
        
        session_summary = {'significant':[], 'value':[]}

        for roi_n in range(data.nROIs):
            ep = EpisodeData(data,
                            protocol_name=p,
                            quantities=['dFoF'], 
                            verbose=False)

            t0 = max([0, ep.time_duration[0]-1.5])
            stat_test_props = dict(interval_pre=[-1.5,0],                                   
                                    interval_post=[t0, t0+1.5],                                   
                                    test='ttest', 
                                    sign='both')

            roi_summary_data = ep.compute_summary_data(stat_test_props=stat_test_props,
                                                        exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
                                                        response_significance_threshold=0.05,
                                                        response_args=dict(roiIndex=roi_n))
            
            session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
            session_summary['value'].append(roi_summary_data['value'][0])

    
        resp_cond = np.array(session_summary['significant'])
        pos_cond = resp_cond & (roi_summary_data['value'][0]>0)
        neg_cond = resp_cond & (roi_summary_data['value'][0]<0)

        print(f'{np.sum(resp_cond)} significant ROI ({np.sum(pos_cond)} positive, {np.sum(neg_cond)} negative) out of {len(session_summary['significant'])} ROIs')

        pos_frac = np.sum(pos_cond)/len(session_summary['significant'])
        neg_frac = np.sum(neg_cond)/len(session_summary['significant'])
        ns_frac = 1-pos_frac-neg_frac


        colors = ['green', 'red', 'grey']

        pt.pie(data=[pos_frac, neg_frac, ns_frac], ax = AX[idx], COLORS = colors)#, pie_labels = ['%.1f%%' % (100*pos),'%.1f%%' % (100*neg), '%.1f%%' % (100*ns)] )
        
        pt.annotate(AX[idx], 'Pos resp=%.2f%%' % (100*pos_frac), (1, 1.5), ha='right', va='top')
        pt.annotate(AX[idx], 'Neg resp=%.2f%%' % (100*neg_frac), (1, 1.2), ha='right', va='top')
     
        

    return fig, AX
#%%
def plot_responsiveness2(data_s, protocols, type = 'means'):

    fig, AX = pt.figure(axes = (len(protocols),1))

    for idx, p in enumerate(protocols):

        pos_s = []
        neg_s = []

        resp_cond_s = []
        pos_cond_s = []
        neg_cond_s = []

        for data in data_s:
            ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'])
            
            sig_list = []
            val_list = []

            for roi_n in range(data.nROIs):
                ep = EpisodeData(data, protocol_name=p, quantities=['dFoF'], verbose=False)

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

        colors = ['green', 'red', 'grey']


        if type== 'means':
            final_pos = np.mean(pos_s)
            final_neg = np.mean(neg_s)
            final_ns = 1 - final_pos - final_neg

        elif type == 'ROI':
            pos_cond_s = np.concatenate(pos_cond_s)
            neg_cond_s = np.concatenate(neg_cond_s)
        
            final_pos = np.mean(pos_cond_s)
            final_neg = np.mean(neg_cond_s)
            final_ns = 1 - final_pos - final_neg

        sem = stats.sem([pos_s, neg_s], axis=1) 

        pt.pie(data=[final_pos, final_neg, final_ns],
            ax=AX[idx],
            COLORS=colors)

        pt.annotate(AX[idx], 'Pos= %.2f%% ± %.2f SEM' % (100 * final_pos, sem[0]),
                    (1, 1.5), ha='right', va='top', fontsize=4)
        pt.annotate(AX[idx], 'Neg= %.2f%% ± %.2f SEM' % (100 * final_neg, sem[1]),
                    (1, 1.2), ha='right', va='top', fontsize=4)

    return fig, AX
#%%
def plot_barplots(data, protocols):
    
    fig, AX = pt.figure(axes = (len(protocols),1))
    
    for idx, p in enumerate(protocols):
        
        ep = EpisodeData(data,
                    protocol_name=p,
                    quantities=['dFoF'])
        
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
        target_len = 9 #generalize
        if len(mean_vals) < target_len:
            mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
        else:
            mean_vals = mean_vals[:target_len]


        x = np.arange(9) #generalize
        AX[idx].bar(x, mean_vals, alpha=0.8, capsize=4)
        AX[idx].set_xticks(x)
        AX[idx].set_title(f'{p}')
        AX[idx].axhline(0, color='black', linewidth=0.8)

    AX[0].set_ylabel('variation dFoF')

    return fig, AX
#%%
def plot_barplots2(data_s, protocols):

    fig, AX = pt.figure(axes = (len(protocols),1))
    
    final_vals_s = []
    final_sems_s = []
    for idx, p in enumerate(protocols):
        mean_vals_s = []
        sem_vals_s = []
        for data in data_s:
            ep = EpisodeData(data,
                        protocol_name=p,
                        quantities=['dFoF'])
            
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
            target_len = 9 #generalize
            if len(mean_vals) < target_len:
                mean_vals.extend([np.nan] * (target_len - len(mean_vals)))
            else:
                mean_vals = mean_vals[:target_len]

            mean_vals_s.append(mean_vals)

        sem_vals_s = stats.sem([r for r in mean_vals_s], axis=0)

        final_vals_s.append(mean_vals_s)
        final_sems_s.append(sem_vals_s)


    for idx, p in enumerate(protocols):

        values = np.mean(final_vals_s[idx], axis=0)
        yerr = final_sems_s[idx]
        print(yerr)
        x = np.arange(9) #generalize
        AX[idx].bar(x, values, yerr=yerr, alpha=0.8, capsize=0, error_kw=dict(linewidth=0.6))
        AX[idx].set_xticks(x)
        AX[idx].set_title(f'{p}')
        AX[idx].axhline(0, color='black', linewidth=0.8)

    AX[0].set_ylabel('variation dFoF')
    
    return fig, AX
#%%
def generate_figures(SESSIONS, cell_type='nan', dFoF_options={}):

    # Generate figures per session 

    for idx, filename in enumerate(SESSIONS['files']):

        data = Data(filename, verbose=False)
        data.build_dFoF(**dFoF_options, verbose=False)
        data.build_running_speed()
        data.build_facemotion()
        data.build_pupil_diameter()

        print("---------------------")
        print(filename.split('\\')[-1])
        print("---------------------")
     
        dict_annotation = {
            'name': filename.split('\\')[-1],
            'Subject_ID': data.metadata['subject_ID'],
            'protocol': data.metadata['protocol']
        }

        settings = find_available_settings(data)
        protocols = [p for p in data.protocols if (p != 'grey-10min') and (p != 'black-2min')]

        ################################# FIGURES ###############################################################
        try: 
            fig1, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
            show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX[0])
            show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX[1])
            show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2,  ax=AX[2])
            #cropped_img = fig1.crop(box= (80, 80, 1000, 430))
            #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/FOV_cropped.png')
            fig1 = figure_to_array(fig1)
        except Exception as e: 
            print(f'problem with figure 1 FOV : {e}')
            fig1, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
            fig1 = figure_to_array(fig1)
        
        try: 
            
            if hasattr(data, "dFoF") and data.dFoF is not None and len(data.dFoF) > 0:
                fig2, _ = plot_raw(data, 
                                   tlim=[0, data.t_dFoF[-1]], 
                                   settings=settings, 
                                   figsize=(9,3),
                                   zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                              ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
                #fig2 = fig2.crop(box=(80, 40, 1340, 430))
                fig2 = figure_to_array(fig2)
            
                fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                                settings=settings, figsize=(9,3))
                #cropped_img = image3.crop(box= (80, 40, 1340, 500))
                #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom1_cropped.png')
                fig3 = figure_to_array(fig3)
              
                fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]],
                                settings=settings, figsize=(9,3))
                #cropped_img = image4.crop(box= (80, 40, 1340, 430))
                #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom2_cropped.png')
                fig4 = figure_to_array(fig4)
              
        except Exception as e:
            print(f'problem with figure 2-3-4 Traces : {e}')
            fig2, _  = pt.figure(axes=(1,1))
            fig2 = figure_to_array(fig2)
            
            fig3, _  = pt.figure(axes=(1,1))
            fig3 = figure_to_array(fig3)
            
            fig4, _  = pt.figure(axes=(1,1))
            fig4 = figure_to_array(fig4)
            
        try:

            fig5, _ = plot_dFoF_per_protocol(data_s=[data])
            fig5 = figure_to_array(fig5)
        except Exception as e:
            print(f'problem with figure 5 mean per subprotocol : {e}')
            fig5, _  = pt.figure(axes=(len(protocols),9)) #generalize 9
            fig5 = figure_to_array(fig5)
        
        try:
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
                                                            exclude_keys= list(ep.varied_parameters.keys()), # we merge different stimulus properties as repetitions of the stim. type  
                                                            response_significance_threshold=0.05,
                                                            response_args=dict(roiIndex=roi_n))
                
                session_summary['significant'].append(bool(roi_summary_data['significant'][0]))
                session_summary['value'].append(roi_summary_data['value'][0])

            resp_cond = np.array(session_summary['significant'])
            pos_cond = resp_cond & (roi_summary_data['value'][0]>0)
            neg_cond = resp_cond & (roi_summary_data['value'][0]<0)

            pos_roi = np.arange(data.nROIs)[pos_cond]
            neg_roi = np.arange(data.nROIs)[neg_cond]
            ns_roi = np.arange(data.nROIs)[~resp_cond]

            if len(pos_roi) > 0:
                roi_n_pos = random.choice(np.arange(data.nROIs)[pos_cond])
            else:
                print("No significant ROIs found — choosing from non-significant set instead.")
                roi_n_pos = random.choice(ns_roi)

            fig6, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roi_n_pos)
            fig6 = figure_to_array(fig6)

            if len(neg_roi) > 0:
                roi_n_neg = random.choice(np.arange(data.nROIs)[neg_cond])
            else:
                print("No significant ROIs found — choosing from non-significant set instead.")
                roi_n_neg = random.choice(ns_roi)

            fig7, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roi_n_neg)
            fig7 = figure_to_array(fig7)


            roi_n_ns = random.choice(ns_roi) 
            fig8, _ = plot_dFoF_per_protocol2(data_s=[data], roiIndex=roi_n_ns)
            fig8 = figure_to_array(fig8)

        except Exception as e:
            print(f'problem with figure 6 7 8 - ROI significative (pos and neg) and non significative : {e}')
            fig6, _  = pt.figure(axes=(len(protocols),1))
            fig6 = figure_to_array(fig6)
            fig7, _  = pt.figure(axes=(len(protocols),1))
            fig7 = figure_to_array(fig7)
            fig8, _  = pt.figure(axes=(len(protocols),1))
            fig8 = figure_to_array(fig8)
        
        try:
            fig9, _ = plot_responsiveness(data, protocols)
            fig9 = figure_to_array(fig9)
        except Exception as e:
            print(f'problem with figure 9 - pie charts : {e}')
            fig9, _  = pt.figure(axes=(len(protocols),1))
            fig9 = figure_to_array(fig9)

        try:
            fig10, _ = plot_barplots(data, protocols)
            fig10 = figure_to_array(fig10)
        except Exception as e:
            print(f'problem with figure 10 - barplots : {e}')
            fig10, _  = pt.figure(axes=(len(protocols),1))
            fig10 = figure_to_array(fig10)
            
        create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, cell_type)
        

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

        plt.close(fig_p1)
        plt.close(fig_p2)
        plt.close(fig_p3)

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0
#%%
def generate_figures_GROUP(SESSIONS):
    
    dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8
    }
    data_s = []
    for idx, filename in enumerate(SESSIONS['files']):

        data = Data(filename, verbose=False)
        data.build_dFoF(**dFoF_options, verbose=False)
        data.build_running_speed()
        data.build_facemotion()
        data.build_pupil_diameter()
        data_s.append(data)

    #Assume all datfiles have same protocols
    data = Data(SESSIONS['files'][0], verbose=False)
    protocols = [p for p in data.protocols 
                 if (p != 'grey-10min') and (p != 'black-2min') and (p != 'looming-stim') and (p != 'moving-dots') and (p != 'random-dots')]

    try:
            fig5, _ = plot_dFoF_per_protocol(data_s=data_s, protocols=protocols)
            fig5 = figure_to_array(fig5)
    except Exception as e:
            print(f'problem with figure 5 mean per subprotocol : {e}')
            fig5, _  = pt.figure(axes=(len(protocols),9)) #generalize 9
            fig5 = figure_to_array(fig5)
    
    fig6, _  = pt.figure(axes=(len(protocols),1))
    fig6 = figure_to_array(fig6)
    fig7, _  = pt.figure(axes=(len(protocols),1))
    fig7 = figure_to_array(fig7)

    
    ###### ROI
    try:
        fig8, _ = plot_responsiveness2(data_s, protocols, type='ROI')
        fig8 = figure_to_array(fig8)
    except Exception as e:
        print(f'problem with figure 8 - pie charts ROI: {e}')
        fig8, _  = pt.figure(axes=(len(protocols),1))
        fig8 = figure_to_array(fig8)

    ###### means
    try:
        fig9, _ = plot_responsiveness2(data_s, protocols, type='means')
        fig9 = figure_to_array(fig9)
    except Exception as e:
        print(f'problem with figure 9 - pie charts means: {e}')
        fig9, _  = pt.figure(axes=(len(protocols),1))
        fig9 = figure_to_array(fig9)

    try:
        fig10, _ = plot_barplots2(data_s, protocols)
        fig10 = figure_to_array(fig10)
    except Exception as e:
        print(f'problem with figure 10 - barplots : {e}')
        fig10, _  = pt.figure(axes=(len(protocols),1))
        fig10 = figure_to_array(fig10)

    
    return fig5, fig6, fig7, fig8, fig9, fig10
#%%
def create_group_PDF(fig5, fig6, fig7, fig8, fig9, fig10, cell_type):
    try: 
        
        pdf1 = PDF2()
        pdf1.fill_PDF2(fig5)
        fig_p1 = pdf1.fig
        
        pdf2 = PDF3()
        pdf2.fill_PDF3(fig6, fig7, fig8, fig9, fig10)
        fig_p2 = pdf2.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/GROUP_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2

        plt.close(fig_p1)
        plt.close(fig_p2)

        print("GROUP PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating GROUP PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################
##################################################################################################################
# %% [markdown]
# # Generate final figures
#%%
# SETTINGS
dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor': 1.0,
                    'method_for_F0': 'sliding_percentile',
                    'sliding_window': 300.,
                    'percentile': 10.,
                    'neuropil_correction_factor': 0.8}


#%%
#######################################################
############### summarize all files ###################
#######################################################
#%% [markdown]
# # Summarize all files

#%% [markdown]
# ## NDNF CRE BATCH 1

#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_test2')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%% [markdown]
# ## One file
#generate_figures(SESSIONS['files'][0], cell_type='NDNF', dFoF_options=dFoF_options)
#%% [markdown]
# ## All individual files
#%%
#generate_figures(SESSIONS, cell_type='NDNF', dFoF_options=dFoF_options)
#%% [mardown]
# ## GROUPED ANALYSIS

#%%
dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8
    }
data_s = []
for idx, filename in enumerate(SESSIONS['files']):

    data = Data(filename, verbose=False)
    data.build_dFoF(**dFoF_options, verbose=False)
    data.build_running_speed()
    data.build_facemotion()
    data.build_pupil_diameter()
    data_s.append(data)

type='ROI'
#Assume all datfiles have same protocols
data = Data(SESSIONS['files'][0], verbose=False)
protocols = [p for p in data.protocols 
                if (p != 'grey-10min') and (p != 'black-2min')]

# test code here

#%%
'''
fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures_GROUP(SESSIONS)
create_group_PDF(fig5, fig6, fig7, fig8, fig9, fig10, 'NDNF')
'''
##############################################################################################################
##############################################################################################################
#%% [markdown]
# ## YANN DATASET

#%%

datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]

#%% [markdown]
# ## One file
#%%
#generate_figures(SESSIONS['files'][0], cell_type='NDNF_YANN', dFoF_options=dFoF_options)
#%% [markdown]
# ## All individual files
#%%
#generate_figures(SESSIONS, cell_type='NDNF_YANN', dFoF_options=dFoF_options)

#%% [mardown]
# ## GROUPED ANALYSIS
#%%
fig5, fig6, fig7, fig8, fig9, fig10 = generate_figures_GROUP(SESSIONS)
create_group_PDF(fig5, fig6, fig7, fig8, fig9, fig10, 'NDNF_YANN')
