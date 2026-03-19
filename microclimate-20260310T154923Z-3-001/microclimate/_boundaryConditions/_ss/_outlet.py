# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class outlet:
    
    def __init__(self,name):
    
        # the name of the inlet
        self.name = name
    
    def write_U(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tinletOutlet;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tinletValue\t\t\tuniform (0 0 0);\n\
\t}}\n"         
        
        return str_return
    
    def write_k(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tinletOutlet;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tinletValue\t\t\tuniform (0 0 0);\n\
\t}}\n"         
        
        return str_return
    
    def write_omega(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tinletOutlet;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tinletValue\t\t\tuniform (0 0 0);\n\
\t}}\n"         
        
        return str_return
    
    def write_nut(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tcalculated;\n\
\t\tvalue\t\t\t0;\n\
\t}}\n"         
        
        return str_return
    
    def write_p(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tuniformFixedValue;\n\
\t\tuniformValue\t\t\tconstant 0;\n\
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
\t\ttype\t\t\ttotalPressure;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\trho\t\t\trhok;\n\
\t\tp0\t\t\tuniform 0;\n\
\t\tphi\t\t\tphi;\n\
\t\tpsi\t\t\tnone;\n\
\t\tgamma\t\t\t1.4;\n\
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
\t\ttype\t\t\tinletOutlet;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tinletValue\t\t\t$internalField;\n\
\t}}\n"         
        
        return str_return
