# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class exhausts:
    
    def __init__(self,name,Q=1,k=0.1,omega=0.01,tracer=1,T=300):
        self.name = name
        self.Q = Q
        self.T = T
        self.k = k
        self.omega = omega
        self.tracer = tracer
        
    def write_U(self):
        if self.Q < 0:
            raise ValueError("Exhausts must have a positive flowrate!")
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tflowRateInletVelocity;\n\
\t\tvalue\t\t\tuniform (0 0 0);\n\
\t\tvolumetricFlowRate\t\t\t{self.Q};\n\
\t}}\n"         
        return str_return
    
    def write_k(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform {self.k};\n\
\t}}\n"         
        return str_return
    
    def write_omega(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform {self.omega};\n\
\t}}\n"         
        return str_return
    
    def write_nut(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    def write_p(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tcalculated;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}}\n"         
        return str_return
    
    def write_LAD(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_plantCd(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_qPlant(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_prgh(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedFluxPressure;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\trho\t\t\trhok;\n\
\t}}\n"         
        return str_return
    
    def write_T(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform {self.T};\n\
\t}}\n"         
        return str_return
    
    def write_alphaT(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
        
    def write_tracer(self):
        str_return = f"{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform {self.tracer};\n\
\t}}\n"         
        return str_return
