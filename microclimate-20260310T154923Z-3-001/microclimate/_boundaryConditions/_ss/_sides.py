# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class sides:
    
    def __init__(self,name):
    
        # the name of the sides
        self.name = name
    
    def write_U(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"         
        return str_return
    
    def write_k(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"   
        return str_return
    
    def write_omega(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"    
        return str_return
    
    def write_nut(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"     
        return str_return
    
    def write_p(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"   
        return str_return
    
    def write_LAD(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"      
        return str_return
    
    def write_plantCd(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"    
        return str_return
    
    def write_qPlant(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"    
        return str_return
    
    def write_prgh(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"     
        return str_return   
    
    def write_T(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"  
        return str_return
    
    def write_alphaT(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"  
        return str_return
        
    def write_tracer(self):
        str_return = f"\t{self.name[0]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n\
\t{self.name[1]}\n\
\t{{\n\
\t\ttype\t\t\tsymmetry;\n\
\t}}\n"  
        return str_return
