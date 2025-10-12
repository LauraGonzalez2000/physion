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
protocol="static-patch"
#%%
datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-Cre-batch1','NWBs')
SESSIONS = scan_folder_for_NWBfiles(datafolder)
filename = SESSIONS['files'][0]
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

def plot(response, title=''):
    fig, AX = pt.figure(figsize=(1,1))
    for r in response:
        AX.plot(ep.t, r, lw=0.4, color='dimgray')
    AX.plot(ep.t, np.mean(response, axis=0), lw=2, color='k')
    pt.set_plot(AX, xlabel='time from start (s)', ylabel='dFoF',
                title=title)
    
response = ep.get_response2D(quantity="dFoF", averaging_dimension='ROIs')
plot(response, 'mean over ROIs, n=%i eps' % response.shape[0])