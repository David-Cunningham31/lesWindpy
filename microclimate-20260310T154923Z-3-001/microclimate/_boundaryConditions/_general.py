# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 15:07:56 2023

@author: Viet.Le
"""

import logging
logger = logging.getLogger(__name__)



def header_template(file_class,file_name,file_dimensions,file_internalField):
        
        return f"\
/*--------------------------------*- C++ -*----------------------------------*\n\
| =========                |                                                 |\n\
| \      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |\n\
|  \    /   O peration     | Version:  v2006                                 |\n\
|   \  /    A nd           | Website:  www.openfoam.com                      |\n\
|    \/     M anipulation  |                                                 |\n\
\*---------------------------------------------------------------------------*/\n\
FoamFile\n\
{{\n\
\tversion\t 4.0;\n\
\tformat\t ascii;\n\
\tclass\t {file_class};\n\
\tlocation\t \"0\";\n\
\tobject\t {file_name};\n\
}}\n\
#include \"initialConditions\";\n\
#include \"ABLConditions\";\n\
dimensions\t\t {file_dimensions};\n\
internalField\t\t uniform {file_internalField};\n\
boundaryField\n\
{{\n\
"



def write_general_closer():
    return "\n}"


def write_str(obj):
    file_str = str()

    file_str += obj.write_header()
    
    methods = ["write_inlet","write_outlet","write_sides","write_top","write_ground","write_walls"]
    for method in methods:
        write_op = getattr(obj, method, None)
        if write_op:
            file_str += write_op()

    try:
        if obj.intakes or obj.exhausts:
            file_str += obj.write_intakes()
            file_str += obj.write_exhausts()
    except Exception as e:
        print(f"Did not write out intakes or exhausts because of error {e}")


    write_op = getattr(obj, "write_closer", None)
    if write_op:
        file_str += write_op()            
    else:
        file_str += write_general_closer()

    return file_str
    
    
    
    