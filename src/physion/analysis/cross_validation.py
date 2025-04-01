import numpy as np
import os
from read_NWB import Data
from dataframe import NWB_to_dataframe
import matplotlib.pylab as plt
import pandas as pd


def TwoFold_train_test_split(filename, data, df, spont_act_key='VisStim_grey-10min', running_speed_threshold=0.1):
    Indices = {}
    indices = np.arange(len(df['time']))

    df = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)
    

    ###################################
    # Compute running activity
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)  
    HMcond_elongated = pd.Series((running_dFoF_sampled >= 0.60).astype(bool)) 

    ###################################
    # First: Split spontaneous activity
    spont_cond = df[spont_act_key]
    spont_indices = indices[spont_cond]  
    spont_running = spont_indices[HMcond_elongated[spont_cond]]
    spont_not_running = spont_indices[~HMcond_elongated[spont_cond]]
   
    # Ensure balanced splits
    Nspont_run = len(spont_running) // 2
    Nspont_norun = len(spont_not_running) // 2
 
    Indices['spont_train_sets'] = [np.concatenate([spont_running[:Nspont_run], spont_not_running[:Nspont_norun]]),
                                   np.concatenate([spont_running[Nspont_run:], spont_not_running[Nspont_norun:]])]
    Indices['spont_test_sets'] = [np.concatenate([spont_running[Nspont_run:], spont_not_running[Nspont_norun:]]),
                                  np.concatenate([spont_running[:Nspont_run], spont_not_running[:Nspont_norun]])]

    ###################################
    # Then: Split stimulus-evoked activity
    stim_keys = [k for k in df if 'VisStim' in k]
    stimID = np.zeros(len(df['time']), dtype=int)

    for i, k in enumerate(stim_keys):
        stimID[df[k]] = i + 1  

    stim_cond = ~df[spont_act_key]
    stim_indices = indices[stim_cond]  

    # Separate running and non-running moments
    stim_running = stim_indices[HMcond_elongated[stim_cond]]
    stim_not_running = stim_indices[~HMcond_elongated[stim_cond]]

    # Ensure balanced splits
    Nstim_run = len(stim_running) // 2
    Nstim_norun = len(stim_not_running) // 2

    Indices['stim_train_sets'] = [np.concatenate([stim_running[:Nstim_run], stim_not_running[:Nstim_norun]]),
                                  np.concatenate([stim_running[Nstim_run:], stim_not_running[Nstim_norun:]])]

    Indices['stim_test_sets'] = [np.concatenate([stim_running[Nstim_run:], stim_not_running[Nstim_norun:]]),
                                 np.concatenate([stim_running[:Nstim_run], stim_not_running[:Nstim_norun]])]


    # Debugging: Print the number of running/not running instances
    for i in range(2):
        print(f"\nFold {i + 1} (Spontaneous Activity):")
        print(f"  Training ({len(Indices['spont_train_sets'][i])}) - Running: {np.sum(HMcond_elongated[Indices['spont_train_sets'][i]])}, Non-Running: {len(Indices['spont_train_sets'][i]) - np.sum(HMcond_elongated[Indices['spont_train_sets'][i]])}")
        print(f"  Test ({len(Indices['spont_test_sets'][i])}) - Running: {np.sum(HMcond_elongated[Indices['spont_test_sets'][i]])}, Non-Running: {len(Indices['spont_test_sets'][i]) - np.sum(HMcond_elongated[Indices['spont_test_sets'][i]])}")


        print(f"\nFold {i + 1} (Stimulus-Evoked Activity):")
        print(f"  Training ({len(Indices['stim_train_sets'][i])}) - Running: {np.sum(HMcond_elongated[Indices['stim_train_sets'][i]])}, Non-Running: {np.sum(~HMcond_elongated[Indices['stim_train_sets'][i]])}")
        print(f"  Test ({len(Indices['stim_test_sets'][i])}) - Running: {np.sum(HMcond_elongated[Indices['stim_test_sets'][i]])}, Non-Running: {np.sum(~HMcond_elongated[Indices['stim_test_sets'][i]])}")

    return Indices

def TwoFold_train_test_split_basic(df,
                             spont_act_key='VisStim_grey-10min'):

    Indices = {}

    indices = np.arange(len(df['time']))
    
    ###################################
    # first spontaneous activity
    spont_cond = df[spont_act_key]
    Nspont = int(np.sum(spont_cond)/2)

    Indices['spont_train_sets'] = [indices[spont_cond][:Nspont],
                                   indices[spont_cond][Nspont:]]
    Indices['spont_test_sets'] = [indices[spont_cond][Nspont:],
                                  indices[spont_cond][:Nspont]]

    ###################################
    # then stimulus evoked activity
    stim_keys = [k for k in df if ('VisStim' in k)]
    stimID = 0*df['time']
    for i, k in enumerate(stim_keys):
        stimID[df[k]] = i+1
    stim_cond = (~df[spont_act_key])
    Nstim = int(np.sum(stim_cond)/2)

    Indices['stim_train_sets'] = [indices[stim_cond][:Nstim],
                                  indices[stim_cond][Nstim:]]
    Indices['stim_test_sets'] = [indices[stim_cond][Nstim:],
                                 indices[stim_cond][:Nstim]]
    

    return Indices

def plot_cross_val(cvIndices):
    
    fig, ax = plt.subplots(figsize=(7,2))
    ii = 13

    #spontaneous
    for train, test in zip(cvIndices['spont_train_sets'], cvIndices['spont_test_sets']):
        ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
        ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=8)
        ii-=2
    ax.annotate('spont act.    \n(grey screen)        ', (0, ii+3), ha='right')
    ii-=1

    #evoked
    for train, test in zip(cvIndices['stim_train_sets'], cvIndices['stim_test_sets']):
        
        ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=8)
        ax.scatter(df['time'][test] , ii+0.5+np.zeros(len(df['time'][test])),  c='tab:blue',marker="_", lw=8)
        ii-=2
    ax.annotate('visual stim.    \nperiods        ', (0, ii+3), ha='right')
    ii-=1

    #stim id
    stim_keys = [k for k in df if ('VisStim' in k)]
    stimID = 0*df['time']
    stim_cond = (~df['VisStim_grey-10min'])

    for i, k in enumerate(stim_keys):
            stimID[df[k]] = i+1
            #print(stimID[df[k]])
    print(np.unique(stimID[stim_cond]))
    ax.scatter(df['time'][stim_cond], [ii-0.5] * np.sum(stim_cond), c=stimID[stim_cond], marker="_", lw=8, cmap=plt.cm.tab20)
    ax.annotate('visual stim. ID  ', (0, ii-1), ha='right')
    ii-=3

    #movement
    running_dFoF_sampled = data.build_running_speed(specific_time_sampling=df['time'].values)  
    speed_bool = pd.Series((running_dFoF_sampled >= 0.60).astype(int)) 
    ax.scatter(x = df['time'][speed_bool == 1], 
            y = ii + speed_bool[speed_bool == 1] - 0.5, 
            c='tab:orange', 
            marker="_", 
            lw=8, 
            label="Above Threshold")
    ax.annotate('movement  ', (0, ii), ha='right')

    #final arangements
    ax.annotate('training set', (.8,.9), color='tab:red', xycoords='axes fraction')
    ax.annotate('test set\n', (.8,.9), color='tab:blue', xycoords='axes fraction')
    ax.axis('off')
    ax.set_xlabel("time (s)")
    ax.set_title('2-Fold Cross-Validation strategy\n ')
    ax.axes.get_xaxis().set_visible(True)
    plt.show()

    return 0


if __name__=='__main__':
    import os
    from read_NWB import Data
    from dataframe import NWB_to_dataframe
    import matplotlib.pylab as plt
    import pandas as pd

    datafolder = os.path.join(os.path.expanduser('~'), 'DATA', 'In_Vivo_experiments','NDNF-WT-Dec-2022','NWBs')
    filename = os.path.join(datafolder, '2022_12_14-13-27-41.nwb') #for example
    data = Data(filename)
    df = NWB_to_dataframe(filename,
                      normalize=['dFoF', 'Pupil-diameter', 'Running-Speed', 'Whisking'],
                      visual_stim_label='per-protocol-and-parameters',
                      verbose=False)

    
    cvIndices = TwoFold_train_test_split(filename, data, df, spont_act_key='VisStim_grey-10min')
    plot_cross_val(cvIndices)
