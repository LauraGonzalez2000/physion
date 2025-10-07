#import matplotlib.pylab as plt

import numpy as np
import pandas as pd
import openpyxl
from scipy import stats as stats_

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from pathlib import Path
from physion.utils  import plot_tools as pt

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from PIL import Image



class PDF:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, _  = pt.figure(figsize=(8.27, 11.69))
        self.AXs = {}

        # build the axes one by one
        #X0, Y0, DX, DY = 0.12, 0.05, 2, 1.6
        #self.AXs['Notes'] = self.create_panel([X0, Y0, DX, DY], 'Notes')
        #self.AXs['Notes'].axis('off')

        X0, Y0, DX, DY = 1.12, 0.05, 6, 1.6
        self.AXs['FOV'] = self.create_panel([X0, Y0, DX, DY], 'FOV')

        X0, Y0, DX, DY = 0.12, 2, 8, 1.6
        self.AXs['Traces_all'] = self.create_panel([X0, Y0, DX, DY], 'Traces_all')

        X0, Y0, DX, DY = 0.12, 4, 8, 1.6
        self.AXs['Traces_zoom1'] = self.create_panel([X0, Y0, DX, DY], 'Traces_zoom1')

        X0, Y0, DX, DY = 0.12, 6, 8, 1.6
        self.AXs['Traces_zoom2'] = self.create_panel([X0, Y0, DX, DY], 'Traces_zoom2')

    def create_panel(self, coords, title=None):
        """ 
        coords: (x0, y0, dx, dy)
                from left to right
                from top to bottom (unlike matplotlib)
        """
        coords[1] = 1-coords[1]-coords[3]
        print(coords)
        ax = pt.inset(self.fig, rect=coords)

        if title:
            ax.set_title(title, loc='left', pad=2, fontsize=8)
        return ax

    def fill_PDF(self, 
                 dict_annotation, 
                 image1, 
                 image2, 
                 image3, 
                 image4): 
        '''
        txt = (
            f"ID file: {dict_annotation['name']}\n"
            f"Protocol: {dict_annotation['protocol']}\n"
            f"Subject ID: {dict_annotation['Subject_ID']}\n"
        )

        self.fig.text(
            0.05,   # x position in figure coordinates (0=left, 1=right)
            0.95,   # y position in figure coordinates (0=bottom, 1=top)
            txt,
            va='top', ha='left',
            fontsize=10,
            wrap=True
        )
        '''
        
    
        for key in self.AXs:
            self.AXs[key].axis('off')

            '''
            if key=='Notes':
                self.AXs[key].axis('off')
                txt = (
                    f"ID file: {dict_annotation['name']}\n"
                    f"Protocol: {dict_annotation['protocol']}\n"
                    f"Subject ID: {dict_annotation['Subject_ID']}\n"
                )
                self.AXs[key].text(0, 1, txt, va='top', ha='left', fontsize=10, wrap=True)
                self.AXs[key].axis('off')
            '''
            if key=='FOV':
                self.AXs[key].imshow(image1)
                
            elif key=='Traces_all':
                self.AXs[key].imshow(image2)
                
            elif key=='Traces_zoom1':
                self.AXs[key].imshow(image3)

            elif key=='Traces_zoom2':
                self.AXs[key].imshow(image4)


class PDF2:

    def __init__(self, 
                 structure_dict={},
                 debug=False):
        
        # figure in A4 format
        self.fig, _  = pt.figure(figsize=(8.27, 11.69))
        self.AXs = {}

        # build the axes one by one
        #X0, Y0, DX, DY = 0.12, 0.05, 2, 1.6
        #self.AXs['Notes'] = self.create_panel([X0, Y0, DX, DY], 'Notes')
        #self.AXs['Notes'].axis('off')

        X0, Y0, DX, DY = 0.12, 0.05, 8, 3
        self.AXs['Resp_protocol'] = self.create_panel([X0, Y0, DX, DY], 'Resp_protocol')

        X0, Y0, DX, DY = 0.12, 3, 8, 3
        self.AXs['ex1_ROI'] = self.create_panel([X0, Y0, DX, DY], 'ex1_EOI')

        X0, Y0, DX, DY = 0.12, 6, 8, 3
        self.AXs['fract_resp'] = self.create_panel([X0, Y0, DX, DY], 'fract_resp')





    def create_panel(self, coords, title=None):
            """ 
            coords: (x0, y0, dx, dy)
                    from left to right
                    from top to bottom (unlike matplotlib)
            """
            coords[1] = 1-coords[1]-coords[3]
            print(coords)
            ax = pt.inset(self.fig, rect=coords)

            if title:
                ax.set_title(title, loc='left', pad=2, fontsize=8)
            return ax
    
    def fill_PDF2(self, 
                 image1, 
                 image2, 
                 image3): 
        for key in self.AXs:
            self.AXs[key].axis('off')
            if key=='Resp_protocol':
                self.AXs[key].imshow(image1)
            
            elif key=='ex1_ROI':
                self.AXs[key].imshow(image2)

            elif key=='fract_resp':
                self.AXs[key].imshow(image3)

            
                