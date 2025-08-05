"""
==========================================
 ---  template for new visual stimuli ---
==========================================

copy this and rename to the desired script name

[!!] need to add the new script to the "stimuli/__init__.py" 
"""
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    STIMULUS TEMPLATE    --- #####
##########################################

params = {"presentation-duration": 2.0,
          "N_repeat": 1,
          "N_deviant": 1,
          "N_redundant": 7,
          "Nmin-successive-redundant": 5,
          "stim-duration": 2.0,
          "seed": 42,
          "stimulus": "grating",
          "angle-redundant": 45.0,
          "angle-deviant": 90.0,
          "spatial-freq": 0.04,
          "contrast": 1.0,
          "speed": 0,
          "bg-color":0.5, 
          'Screen': "Dell-2020"}   

class stim(visual_stim):
    """
    stimulus specific visual stimulation object
    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)
       
        # fix pseudo-random seed
        np.random.seed(int(protocol['seed']))

        # initialize a default sequence:
        NperRepeat = int(protocol['N_deviant']+protocol['N_redundant'])
        params['N-repeat'] = NperRepeat*protocol['N_repeat']

        # that we modify to stick to the oddball protocol
        NmSR = int(protocol['Nmin-successive-redundant'])
        N = int(NperRepeat-NmSR)
        iShifts = np.random.randint(0, N, protocol['N_repeat'])
        angles = [None] * params['N-repeat']
        #self.experiment['angle'] = [None] * params['N-repeat']
        
        for repeat in range(protocol['N_repeat']):
            # first redundants until random iShift
            start1 = repeat*NperRepeat
            end1 = repeat*NperRepeat+NmSR+iShifts[repeat]
            for i in range(start1, end1):
                angles[i] = (params['angle-redundant'])
                #self.experiment['angle'][i] = (params['angle-redundant'])
            
            # deviant at random iShift
            deviant = repeat*NperRepeat+NmSR+iShifts[repeat]
            angles[deviant] = params['angle-deviant']

            #self.experiment['angle'][deviant] = params['angle-deviant']

            # last redundants from random iShift
            start2 = repeat*NperRepeat+NmSR+iShifts[repeat]+1
            end2 = (repeat+1)*NperRepeat
            for i in range(start2,end2):
                angles[i] = params['angle-redundant']
                #self.experiment['angle'][i]  = params['angle-redundant']

        print("final angles : ", angles)
        self.refresh_freq = protocol['movie_refresh_freq']
        #print(self)
        print("attributes", self.__dict__)
        self.experiment['angle'] = angles
        

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        img = init_bg_image(self, index)
        self.add_grating_patch(img,
                               angle=self.experiment['angle'][index],
                               radius=200.,
                               spatial_freq=self.experiment['spatial-freq'][index],
                               contrast=self.experiment['contrast'][index])
        return img


if __name__=='__main__':
    
    from physion.visual_stim.build import get_default_params

    params = get_default_params('oddball')
    params['radius'] = 20.
    params['speed'] = 2.
    params['angle-surround'] = 90.
    params['radius-surround'] = 50.
    params['speed-surround'] = 2.

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break