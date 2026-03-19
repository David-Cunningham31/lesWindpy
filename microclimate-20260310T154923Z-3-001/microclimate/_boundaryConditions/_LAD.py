# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
from ._general import header_template

logger = logging.getLogger(__name__)



#%%

class LAD:
    
    def __init__(self,flowDir,
                 intakes=None,exhausts=None):
    
        # required parameters related to wind
        self.flowDir = flowDir
        if not isinstance(flowDir, (list, tuple)) or len(flowDir) != 3:
            
            if not isinstance(flowDir, (int, float)):
                logger.error("Could not read wind direction")
                raise TypeError(f'Please provide wind direction {flowDir} as a 3-digit list or tuple or as a single number')
        self.intakes = intakes
        self.exhausts = exhausts
        


    def write_header(self):
        LADstr = header_template('volScalarField','leafAreaDensity','[0 -1 0 0 0 0 0]','0')
        return LADstr



    def write_inlet(self):
        LADstr = f"\tside{self.flowDir:03}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"            
        return LADstr
    
    

    def write_outlet(self):
        str_return = f"\tside{(self.flowDir+180)%360:03}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
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
        str_return = f"\ttop\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"          
        return str_return



    def write_ground(self):
        str_return = f"\t\"(ground).*\"\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"          
        return str_return



    def write_walls(self):
        str_return = f"\t\"(wall).*\"\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"         
        return str_return



    def write_intakes(self):
        str_return = ""
        for intake, param in self.intakes.items():
            str_return += f"\n\t{intake}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"           
        return str_return



    def write_exhausts(self):
        str_return = ""
        for exhaust, param in self.exhausts.items():
            str_return += f"\n\t{exhaust}\n\
\t{{\n\
\t\ttype\t\t\tfixedValue;\n\
\t\tvalue\t\t\tuniform 0;\n\
\t}}\n"        
        return str_return
    
    
    
    
    
    
    
    
    
    