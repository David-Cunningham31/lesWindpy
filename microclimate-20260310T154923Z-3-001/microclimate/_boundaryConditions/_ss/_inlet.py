# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class inlet:
    
    def __init__(self,name,
                 Uref,Zref,flowDir,z0,
                 k=0.1,omega=0.01,p=0,
                 tracers=None):
    
        # the name of the inlet
        self.name = name
        
        # required parameters related to wind
        self.flowDir = flowDir
        if not isinstance(flowDir, (list, tuple)) or len(flowDir) != 3:
            
            if isinstance(flowDir, (int, float)):
                angle = np.radians(flowDir)
                self.xvel = -Uref*np.sin(angle)
                self.yvel = -Uref*np.cos(angle)    
            else:
                logger.error("Could not read wind direction")
                raise TypeError(f'Please provide wind direction {flowDir} as a 3-digit list or tuple or as a single number')
        else:
            self.xvel = Uref * flowDir[0]
            self.yvel = Uref * flowDir[1]
        self.z0 = z0
        self.Uref = Uref
        self.Zref = Zref
        
        # parameters that have defaults if not provided. 
        # Turbulence and pressure
        self.k = k
        self.omega = omega
        self.p = p
        
        # optional parameters
        # tracers. Need to provide a list of strings
        if isinstance(tracers, list):
            for tracer in tracers:
                self.tracer = tracer
        elif not tracer and not isinstance(tracer, list):
            raise TypeError('Please provide tracers as a list of strings')
        
        
    
    def write_U(self):
        
        if isinstance(self.flowDir, (list,tuple,np.ndarray)):
            if len(self.flowDir) == 3:
                internalField = self.flowDir
        else:
            
            # "flips" the flow direction so it's coming toward the center
            # then 90 - X so that cosine is for x-comp, sine is for y-comp
            internalField = (np.cos((90-((180+self.flowDir)%360))*np.pi/180), 
                             np.sin((90-((180+self.flowDir)%360))*np.pi/180),
                             0)
        Ustr = f"\t{self.name}\n\
\t{{\n\
\t\tz0\t\t\tuniform {self.z0};\n\
\t\tzDir\t\t\t(0 0 1);\n\
\t\tvalue\t\t\tuniform ({internalField[0]*self.Uref:.3f} {internalField[1]*self.Uref:.3f} {internalField[2]*self.Uref:.3f});\n\
\t\tzGround\t\t\tuniform 0;\n\
\t\tUref\t\t\t{self.Uref:.3f};\n\
\t\tZref\t\t\t{self.Zref};\n\
\t\ttype\t\t\tatmBoundaryLayerInletVelocity;\n\
\t\tflowDir\t\t\t{self.flowDir};\n\
\t\td\t\t\tuniform 0;\n\
\t}}\n"          
        
        return Ustr
    
    def write_k(self):
        kstr = f"\t{self.name}\n\
\t{{\n\
\t\tz0\t\t\tuniform {self.z0};\n\
\t\tzDir\t\t\t(0 0 1);\n\
\t\tvalue\t\t\tuniform {self.k:.3f};\n\
\t\tzGround\t\t\tuniform 0;\n\
\t\tUref\t\t\t{self.Uref};\n\
\t\tZref\t\t\t{self.Zref};\n\
\t\ttype\t\t\tatmBoundaryLayerInletK;\n\
\t\tflowDir\t\t\t{self.flowDir};\n\
\t\td\t\t\tuniform 0;\n\
\t\tC1\t\t\t0;\n\
\t\tC2\t\t\t1;\n\
\t}}\n"          
        return kstr

    def write_omega(self):
        omegastr = f"\t{self.name}\n\
\t{{\n\
\t\tz0\t\t\tuniform {self.z0};\n\
\t\tzDir\t\t\t(0 0 1);\n\
\t\tvalue\t\t\tuniform {self.omega};\n\
\t\tzGround\t\t\t uniform 0;\n\
\t\tUref\t\t\t{self.Uref};\n\
\t\tZref\t\t\t{self.Zref};\n\
\t\ttype\t\t\tatmBoundaryLayerInletOmega;\n\
\t\tflowDir\t\t\t{self.flowDir};\n\
\t\td\t\t\tuniform 0;\n\
\t}}\n"          
        return omegastr

    def write_nut(self):
        nutstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\t calculated;\n\
\t\tvalue\t\t\t uniform 0;\n\
\t}}\n"          
        return nutstr

    def write_p(self):
        pstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"          
        return pstr
    
    def write_LAD(self):
        LADstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"          
        return LADstr
    
    def write_plantCd(self):
        plantCdstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"          
        return plantCdstr
    
    def write_qPlant(self):
        qPlantstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"          
        return qPlantstr

    def write_prgh(self):
        prghstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedFluxPressure;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\trho\t\t\trhok;\n\
\t}}\n"          
        return prghstr    

    def write_T(self):
        Tstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}}\n"          
        return Tstr

    def write_alphaT(self):
        alphaTstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"          
        return alphaTstr

    def write_tracer(self):
        tracerstr = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}}\n"          
        return tracerstr
