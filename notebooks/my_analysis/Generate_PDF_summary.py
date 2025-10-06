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

from PDF_layout import PDF, PDF2

from matplotlib.backends.backend_pdf import PdfPages

from General_overview_episodes import plot_dFoF_per_protocol, plot_dFoF_per_protocol2

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


#%%
def generate_sub_png(data, settings, dict_annotation, cell_type, grey=False, black=False, grey_co=[], black_co=[]):
    
    fig1, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
    show_CaImaging_FOV(data, key='meanImg', 
                       cmap=pt.get_linear_colormap('k', 'tab:green'),
                       NL=2, # non-linearity to normalize image
                       ax=AX[0])
    show_CaImaging_FOV(data, key='max_proj', 
                       cmap=pt.get_linear_colormap('k', 'tab:green'),
                       NL=2, # non-linearity to normalize image
                       ax=AX[1])
    show_CaImaging_FOV(data, key='meanImg', 
                       cmap=pt.get_linear_colormap('k', 'tab:green'),
                       NL=2,
                       roiIndices=range(data.nROIs), 
                       ax=AX[2])
    fig1.savefig(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo',cell_type, 'pieces', 'FOV.png'), facecolor="white")
    image1 = Image.open(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'FOV.png'))
    image1.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/FOV.png')
    #cropped_img = image1.crop(box= (80, 80, 1000, 430))
    #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/FOV_cropped.png')

    fig2, _ = plot_raw(data, tlim=[0, data.t_dFoF[-1]], settings=settings, figsize=(9,3), grey=grey, black=black, grey_co=grey_co, black_co=black_co, zoom_area = [((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
    fig2.savefig(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo',cell_type, 'pieces', 'full_view.png'), facecolor="white")
    image2 = Image.open(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'full_view.png'))
    image2.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/full_view.png')
    #cropped_img = image2.crop(box= (80, 40, 1340, 430))
    #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/full_view_cropped.png')
    
    fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]], settings=settings, figsize=(9,3), grey=grey, black=black, grey_co=grey_co, black_co=black_co)
    fig3.savefig(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'zoom1.png'), facecolor="white")
    image3 = Image.open(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'zoom1.png'))
    image3.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom1.png')
    #cropped_img = image3.crop(box= (80, 40, 1340, 500))
    #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom1_cropped.png')
    
    fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]], settings=settings, figsize=(9,3), grey=grey, black=black, grey_co=grey_co, black_co=black_co)
    fig4.savefig(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'zoom2.png'), facecolor="white")
    image4 = Image.open(os.path.join(os.path.expanduser('~'), 'Output_expe','In_Vivo', cell_type, 'pieces', 'zoom2.png'))
    image4.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom2.png')
    #cropped_img = image4.crop(box= (80, 40, 1340, 430))
    #cropped_img.save(f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/pieces/zoom2_cropped.png')
    
    return 0

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
                                                   'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
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
                                                   'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
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
                                               'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
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
                                               'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
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
'''
def generate_figures(SESSIONS, cell_type='nan'):
    
    range_ = [0]
    data_s = []
    dFoF_options = {'roi_to_neuropil_fluo_inclusion_factor' : 1.0, # ratio to discard ROIs with weak fluo compared to neuropil
                        'method_for_F0' : 'sliding_percentile', # either 'minimum', 'percentile', 'sliding_minimum', or 'sliding_percentile'
                        'sliding_window' : 300. , # seconds (used only if METHOD= 'sliding_minimum' | 'sliding_percentile')
                        'percentile' : 10. , # for baseline (used only if METHOD= 'percentile' | 'sliding_percentile')
                        'neuropil_correction_factor' : 0.8 }# fraction of neuropil substracted to fluorescence
        
    if isinstance(SESSIONS, str):
        filename = SESSIONS
    
    if not isinstance(SESSIONS, str):
        range_ =  range(len(SESSIONS['files']))
        for i in range_:
            data = Data(SESSIONS['files'][i], verbose=False)
            data.build_dFoF(**dFoF_options, verbose=False)
            data_s.append(data)
        
    for index in range_:
        if not isinstance(SESSIONS, str):
             filename = SESSIONS['files'][index]

        data = Data(filename,verbose=False)
        data.build_dFoF(**dFoF_options,verbose=False )
        data.build_running_speed()
        data.build_facemotion()
        data.build_pupil_diameter()
        data_s.append(data)

        protocol = data.metadata['protocol']
        subject_id = data.metadata['subject_ID']
        
        dict_annotation = {'name': filename.split('\\')[-1],
                           'Subject_ID' : subject_id,
                           'protocol' : protocol}

        settings = find_available_settings(data)
        
        generate_sub_png(data, settings, dict_annotation, cell_type)
                
        fig1, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg', 
                        cmap=pt.get_linear_colormap('k', 'tab:green'),
                        NL=2, # non-linearity to normalize image
                        ax=AX[0])
        show_CaImaging_FOV(data, key='max_proj', 
                        cmap=pt.get_linear_colormap('k', 'tab:green'),
                        NL=2, # non-linearity to normalize image
                        ax=AX[1])
        show_CaImaging_FOV(data, key='meanImg', 
                        cmap=pt.get_linear_colormap('k', 'tab:green'),
                        NL=2,
                        roiIndices=range(data.nROIs), 
                        ax=AX[2])
        fig1 = figure_to_array(fig1)
            
        fig2, _ = plot_raw(data, tlim=[0, data.t_dFoF[-1]], settings=settings, figsize=(9,3), zoom_area = [((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
        fig2 = figure_to_array(fig2)

        fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]], settings=settings, figsize=(9,3))
        fig3 = figure_to_array(fig3)

        fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]], settings=settings, figsize=(9,3))
        fig4 = figure_to_array(fig4)

    for index in range_:
        fig5, _ = plot_dFoF_per_protocol(data_s=data_s, dataIndex = index)
        fig5 = figure_to_array(fig5)

    return dict_annotation, fig1, fig2, fig3, fig4, fig5
'''
def generate_figures(SESSIONS, cell_type='nan'):
    

    dFoF_options = {
        'roi_to_neuropil_fluo_inclusion_factor': 1.0,
        'method_for_F0': 'sliding_percentile',
        'sliding_window': 300.,
        'percentile': 10.,
        'neuropil_correction_factor': 0.8
    }

    data_s = []
    dict_annotation = {}

    # --- Load data once ---
    if isinstance(SESSIONS, str):
        filenames = [SESSIONS]
    else:
        filenames = SESSIONS['files']

    for filename in filenames:
        print(filename)
        data = Data(filename, verbose=False)
        data.build_dFoF(**dFoF_options, verbose=False)
        data.build_running_speed()
        data.build_facemotion()
        data.build_pupil_diameter()
        data_s.append(data)

    print(len(data_s))
    # --- Generate figures per session ---
    for idx, data in enumerate(data_s):
        filename = filenames[idx]
        protocol = data.metadata['protocol']
        subject_id = data.metadata['subject_ID']

        dict_annotation = {
            'name': filename.split('\\')[-1],
            'Subject_ID': subject_id,
            'protocol': protocol
        }

        settings = find_available_settings(data)
        generate_sub_png(data, settings, dict_annotation, cell_type)

        fig1, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX[0])
        show_CaImaging_FOV(data, key='max_proj',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, ax=AX[1])
        show_CaImaging_FOV(data, key='meanImg',cmap=pt.get_linear_colormap('k', 'tab:green'),NL=2, roiIndices=range(data.nROIs), ax=AX[2])
        fig1 = figure_to_array(fig1)

        fig2, _ = plot_raw(data, tlim=[0, data.t_dFoF[-1]], settings=settings, figsize=(9,3),
                           zoom_area=[((2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]),
                                      ((15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1])])
        fig2 = figure_to_array(fig2)

        fig3, _ = plot_raw(data, tlim=[(2/20)*data.t_dFoF[-1], (3/20)*data.t_dFoF[-1]],
                           settings=settings, figsize=(9,3))
        fig3 = figure_to_array(fig3)

        fig4, _ = plot_raw(data, tlim=[(15/20)*data.t_dFoF[-1], (16/20)*data.t_dFoF[-1]],
                           settings=settings, figsize=(9,3))
        fig4 = figure_to_array(fig4)

        print("data : " ,[data])
        fig5, _ = plot_dFoF_per_protocol(data_s=[data])
        fig5 = figure_to_array(fig5)

        fig6, _ = plot_dFoF_per_protocol2(data_s=[data])
        fig6 = figure_to_array(fig6)

        create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, cell_type)

    return 0

        
#%%
def create_PDF(dict_annotation, fig1, fig2, fig3, fig4, fig5, fig6, cell_type):
    try: 

        pdf1 = PDF()
        pdf1.fill_PDF(dict_annotation, fig1, fig2, fig3, fig4)
        fig_p1 = pdf1.fig

        pdf2 = PDF2()
        pdf2.fill_PDF2(fig5, fig6)
        fig_p2 = pdf2.fig

        output_path = f'C:/Users/laura.gonzalez/Output_expe/In_Vivo/{cell_type}/Summary_PDF/{os.path.splitext(dict_annotation['name'])[0]}_summary.pdf'

        with PdfPages(output_path) as pdf:
                pdf.savefig(fig_p1, dpi=300, bbox_inches="tight")  # Page 1
                pdf.savefig(fig_p2, dpi=300, bbox_inches="tight")  # Page 2

        plt.close(fig_p1)
        plt.close(fig_p2)

        print("Individual PDF File saved successfully ")
        
    except Exception as e:
        print(f"Error creating the individual PDF file : {e}")

    return 0

##################################################################################################################
##################################################################################################################
##################################################################################################################
# %% [markdown]
# # Generate final figures

#%% [markdown]
# ## NDNF CRE BATCH 1

#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs_test')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
SESSIONS['nwbfiles'] = [os.path.basename(f) for f in SESSIONS['files']]
#%% [markdown]
# ## One file
#%%
generate_figures(SESSIONS['files'][0], cell_type='NDNF')
#%% [mardown]
# ## All files
#%%
generate_figures(SESSIONS, cell_type='NDNF')
