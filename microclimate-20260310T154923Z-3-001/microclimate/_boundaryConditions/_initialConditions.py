# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
from ._general import header_template

logger = logging.getLogger(__name__)



#%%

class initialConditions:
    
    def __init__(self,intakes=None,exhausts=None):
        self.intakes = intakes
        self.exhausts = exhausts
        
    def write_header(self):
        return "\
/*--------------------------------*- C++ -*----------------------------------*\n\
| =========                |                                                 |\n\
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n\
|  \    /   O peration     | Version:  v2006                                 |\n\
|   \  /    A nd           | Website:  www.openfoam.com                      |\n\
|    \/     M anipulation  |                                                 |\n\
\*---------------------------------------------------------------------------*/\n\
FoamFile\n\
{\n\
\tversion\t 4.0;\n\
\tformat\t ascii;\n\
\tclass\t dictionary;\n\
\tlocation\t \"0\";\n\
\tobject\t initialConditions;\n\
}\n\
        "
    
    def write_closer(self):
        str_return = "\n\
ambient_k\t\t0.1;\n\
ambient_omega\t\t0.001;\n\
"
        return str_return
        



    
    
    
    
    
    
    
    
    
    