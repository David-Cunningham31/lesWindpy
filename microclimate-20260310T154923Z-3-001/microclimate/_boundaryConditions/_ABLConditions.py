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

class ABLConditions:
    
    def __init__(self,Uref,Zref,k,omega,ustar,flowDir,z0,z0_ground,z0_water,
                 intakes=None,exhausts=None):
    
        # required parameters related to wind
        self.flowDir = flowDir
        if not isinstance(flowDir, (list, tuple)) or len(flowDir) != 3:
            
            if isinstance(flowDir, (int, float)):
                angle = np.radians(flowDir)
                self.xvel = -np.sin(angle)
                self.yvel = -np.cos(angle)    
            else:
                logger.error("Could not read wind direction")
                raise TypeError(f'Please provide wind direction {flowDir} as a 3-digit list or tuple or as a single number')
        else:
            self.xvel = Uref * np.radians(flowDir[0])
            self.yvel = Uref * np.radians(flowDir[1])
        self.zvel = 0
        self.z0 = z0
        self.z0_ground = z0_ground
        self.z0_water = z0_water
        self.Uref = Uref
        self.Zref = Zref
        self.k = k
        self.omega = omega
        self.ustar = ustar
        self.intakes = intakes
        self.exhausts = exhausts
        
        # calculate shear stress at the top of the domain       
        (self.taux, self.tauy, self.tauz) = (self.ustar**2*self.xvel, self.ustar**2*self.yvel, self.ustar**2*self.zvel)



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
\tobject\t ABLConditions;\n\
}\n\
        "
    
    def write_closer(self):
        str_return = f"\n\
d\t\t0;\n\
C2\t\t1;\n\
C1\t\t0;\n\
kappa\t\t0.41;\n\
Cmu\t\t0.09;\n\
z0\t\t{self.z0:.6f};\n\
z0_ground\t\t{self.z0_ground:.6f};\n\
z0_water\t\t{self.z0_water:.6f};\n\
abl_k\t\t{self.k:.3f};\n\
abl_omega\t\t{self.omega:.3f};\n\
Tau\t\t({self.taux:.3f} {self.tauy:.3f} {self.tauz:.3f});\n\
Uref\t\t{self.Uref:.3f};\n\
Zref\t\t{self.Zref:.3f};\n\
ustar\t\t{self.ustar:.3f};\n\
flowDir\t\t({self.xvel:.3f} {self.yvel:.3f} {self.zvel:.3f});\n\
"
        return str_return
        



    
    
    
    
    
    
    
    
    
    