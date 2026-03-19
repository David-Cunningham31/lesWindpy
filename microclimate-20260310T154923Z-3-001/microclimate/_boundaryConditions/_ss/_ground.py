# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class ground:
    
    def __init__(self,name,k=0.1,omega=0.01,z0=0.03,T=None):
        self.name = name
        self.k = k
        self.omega = omega
        self.z0 = z0
        
    def write_U(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tnoslip;\n\
\t}}\n"         
        return str_return
    
    def write_k(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tkqRWallFunction;\n\
\t\tvalue\t\t\tuniform {self.k:.3f};\n\
\t}}\n"         
        return str_return
    
    def write_omega(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tomegaWallFunction;\n\
\t\tkappa\t\t\t0.41;\n\
\t\tCs\t\t\t0.09;\n\
\t\tvalue\t\t\t{self.omega};\n\
\t}}\n"         
        return str_return
    
    def write_nut(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tatmNutkWallFunction;\n\
\t\tkappa\t\t\t0.41;\n\
\t\tCs\t\t\t0.09;\n\
\t\tz0\t\t\t{self.z0};\n\
\t\tvalue\t\t\t0;\n\
\t}}\n"         
        return str_return
    
    def write_p(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
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
\t\ttype\t\t\tslip;\n\
\t}}\n"         
        return str_return
    
    def write_T(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\ttype\t\t\t{self.T};\n\
\t}}\n"         
        return str_return
    
    def write_alphaT(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\talphatJayatillekeWallFunction;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tPrt\t\t\t0.85;\n\
\t}}\n"         
        return str_return
        
    def write_tracer(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\ttype\t\t\t$internalField;\n\
\t}}\n"         
        return str_return
