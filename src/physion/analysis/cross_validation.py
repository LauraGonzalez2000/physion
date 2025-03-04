import numpy as np

def TwoFold_train_test_split(df,
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

