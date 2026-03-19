# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class header:
    
    def __init__(self,flowDir=0,T=None):
        self.T = T
        self.flowDir = flowDir


    
    def write_U(self):
        if not isinstance(self.flowDir, (list, tuple)) or len(self.flowDir) != 3:
            
            if isinstance(self.flowDir, (int, float)):
                angle = np.radians(self.flowDir)
                self.xvel = -1*np.sin(angle)
                self.yvel = -1*np.cos(angle)    
            else:
                logger.error("Could not read wind direction")
                raise TypeError(f'Please provide wind direction {self.flowDir} as a 3-digit list or tuple or as a single number')
        else:
            self.xvel = 1 * self.flowDir[0]
            self.yvel = 1 * self.flowDir[1]
        self.zvel = 0
        Ustr = self.header_template('volVectorField',
                                    'U',
                                    '[0 1 -1 0 0 0 0]',
                                    f'({self.xvel:.3f} {self.yvel:.3f} {self.zvel:.3f})')
        return Ustr
    
    def write_k(self):
        kstr = self.header_template('volScalarField',
                                    'k',
                                    '[0 2 -2 0 0 0 0]',
                                    '0.1')
        return kstr   
    
    def write_omega(self):
        omegastr = self.header_template('volScalarField',
                                    'omega',
                                    '[0 0 -1 0 0 0 0]',
                                    '0.01')
        return omegastr   
    
    def write_nut(self):
        nutstr = self.header_template('volScalarField',
                                    'nut',
                                    '[0 2 -1 0 0 0 0]',
                                    '0')
        return nutstr   
    
    def write_p(self):
        pstr = self.header_template('volScalarField',
                                    'p',
                                    '[0 2 -2 0 0 0 0]',
                                    '0')
        return pstr      
    
    def write_LAD(self):
        LADstr = self.header_template('volScalarField',
                                    'leafAreaDensity',
                                    '[0 -1 0 0 0 0 0]',
                                    '0')
        return LADstr        
    
    def write_plantCd(self):
        plantCdstr = self.header_template('volScalarField',
                                    'plantCd',
                                    '[0 0 0 0 0 0 0]',
                                    '0')
        return plantCdstr        
    
    def write_qPlant(self):
        qPlantstr = self.header_template('volScalarField',
                                    'qPlant',
                                    '[0 2 -3 0 0 0 0]',
                                    '0')
        return qPlantstr        
    
    def write_prgh(self):
        prghstr = self.header_template('volScalarField',
                                    'p_rgh',
                                    '[0 2 -2 0 0 0 0]',
                                    '0')        
        return prghstr    

    def write_T(self,internalField):
        Tstr = self.header_template('volScalarField',
                                    'T',
                                    '[0 0 0 1 0 0 0]',
                                    f'{internalField}')        
        return Tstr

    def write_alphaT(self):
        alphaTstr = self.header_template('volScalarField',
                                    'alphaT',
                                    '[0 2 -1 0 0 0 0]',
                                    '0')         
        return alphaTstr

    def write_tracer(self,name):
        tracerstr = self.header_template('volScalarField',
                                    name,
                                    '[0 0 0 0 0 0 0]',
                                    '0')        
        return tracerstr    
    
    
    