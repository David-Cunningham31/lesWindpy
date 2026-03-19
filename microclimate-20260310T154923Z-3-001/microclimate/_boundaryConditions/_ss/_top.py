# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 09:58:48 2023

@author: Viet.Le
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

class top:
    
    def __init__(self,name,flowDir,UtopofDomain=30,ztopofDomain=480,z0=0.03):
    
        
        # the name of the top
        self.name = name
        self.UtopofDomain = UtopofDomain
        
        # flow direction
        self.angle = np.radians(flowDir)

        # calculate shear stress at the top of the domain       
        kappa = 0.41
        rho = 1
        ustar = UtopofDomain*kappa / np.log((ztopofDomain+z0)/z0)
        self.tau = rho * ustar ** 2
        
    
    def write_U(self):
        taux = self.tau*-np.sin(self.angle)
        tauy = self.tau*-np.cos(self.angle)
        tauxy=f"({taux:.4f} {tauy:.4f} 0)"
        
#         str_return = f"\t{self.name}\n\
# \t{{\n\
# \t\ttype\t\t\tfixedShearStress;\n\
# \t\ttau\t\t\t{tauxy};\n\
# \t\tvalue\t\t\tuniform (0 0 0);\n\
# \t}}\n"             
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tslip;\n\
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
\t\ttype\t\t\tcalculated;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return
    
    def write_p(self):
        str_return = f"\t{self.name}\n\
\t{{\n\
\t\ttype\t\t\tslip;\n\
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
\t\ttype\t\t\tslip;\n\
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
\t\ttype\t\t\tslip;\n\
\t}}\n"         
        return str_return
