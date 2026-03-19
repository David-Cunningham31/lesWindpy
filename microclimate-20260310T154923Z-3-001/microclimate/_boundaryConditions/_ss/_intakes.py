# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class intakes:
    
    def __init__(self,name,Q=1):
        self.name = name
        self.Q = Q
        
    def write_U(self):
        if self.Q > 0:
            raise ValueError("Intake must have a negative flowrate!")
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tflowRateInletVelocity;\n\
\t\tvalue\t\t\tuniform (0 0 0);\n\
\t\tvolumetricFlowRate\t\t\t{self.Q};\n\
\t}}\n"         
        return str_return
    
    def write_k(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    def write_omega(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    def write_nut(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    def write_p(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tcalculated;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}}\n"         
        return str_return
    
    def write_LAD(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_plantCd(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_qPlant(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_prgh(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedFluxPressure;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\trho\t\t\trhok;\n\
\t}}\n"         
        return str_return
    
    def write_T(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    def write_alphaT(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
        
    def write_tracer(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
