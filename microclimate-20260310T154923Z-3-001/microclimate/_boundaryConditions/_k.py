# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
import numpy as np
from ._general import header_template

logger = logging.getLogger(__name__)



#%%

class k:
    
    def __init__(self,flowDir,intakes=None,exhausts=None):
    
        self.flowDir = flowDir
        if not isinstance(flowDir, (list, tuple)) or len(flowDir) != 3:
            
            if not isinstance(flowDir, (int, float)):
                logger.error("Could not read wind direction")
                raise TypeError(f'Please provide wind direction {flowDir} as a 3-digit list or tuple or as a single number')
        self.intakes = intakes
        self.exhausts = exhausts
        


    def write_header(self):
        kstr = header_template('volScalarField','k','[0 2 -2 0 0 0 0]','$ambient_k')
        return kstr



    def write_inlet(self):
        str_return = f"\tside{self.flowDir:03}\n\
\t{{\n\
\t\tz0\t\t\tuniform $z0;\n\
\t\tzDir\t\t\t(0 0 1);\n\
\t\tvalue\t\t\tuniform 0;\n\
\t\tzGround\t\t\tuniform 0;\n\
\t\tUref\t\t\t$Uref;\n\
\t\tZref\t\t\t$Zref;\n\
\t\ttype\t\t\tatmBoundaryLayerInletK;\n\
\t\tflowDir\t\t\t$flowDir;\n\
\t\td\t\t\tuniform $d;\n\
\t\tCmu\t\t\t$Cmu;\n\
\t\tC1\t\t\t$C1;\n\
\t\tC2\t\t\t$C2;\n\
\t}}\n"          
        
        return str_return
    
    

    def write_outlet(self):
        str_return = f"\tside{(self.flowDir+180)%360:03}\n\
\t{{\n\
\t\ttype\t\t\tinletOutlet;\n\
\t\tvalue\t\t\t$internalField;\n\
\t\tinletValue\t\t\t$internalField;\n\
\t}}\n"   
    
        return str_return



    def write_sides(self):

        side_name = [f'side{(self.flowDir+90)%360:03}',f'side{(self.flowDir+270)%360:03}']
        str_return = f"\t{side_name[0]}\n\
\t{{\n\
\t\ttype\t\t\tslip;\n\
\t}}\n\
\t{side_name[1]}\n\
\t{{\n\
\t\ttype\t\t\tslip;\n\
\t}}\n"         
        return str_return



    def write_top(self):
        str_return = "\ttop\n\
\t{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}\n"         
        return str_return



    def write_ground(self):
        str_return = "\t\"(ground).*\"\n\
\t{\n\
\t\ttype\t\t\tkqRWallFunction;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}\n"         
        return str_return



    def write_walls(self):
        str_return = "\t\"(wall).*\"\n\
\t{\n\
\t\ttype\t\t\tkqRWallFunction;\n\
\t\tvalue\t\t\t$internalField;\n\
\t}\n"         
        return str_return



    def write_intakes(self):
        str_return = ""
        for intake, param in self.intakes.items():
            str_return += f"\n\t{intake}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return



    def write_exhausts(self):
        str_return = ""
        for exhaust, param in self.exhausts.items():
            str_return += f"\n\t{exhaust}\n\
\t{{\n\
\t\ttype\t\t\tzeroGradient;\n\
\t}}\n"         
        return str_return
    
    
    
    
    
    
    
    
    
    