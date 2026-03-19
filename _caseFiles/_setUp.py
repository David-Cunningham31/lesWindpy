# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 13:10:45 2026

@author: David Cunningham
"""

import re
import os
import logging
import numpy as np

#%%

def parse_setup_file(case_path):
    
    filepath = os.path.join(case_path, "setUp")
    
    variable_dict = {}

    with open(filepath, "r") as f:
        for line in f:
            # Remove inline comments
            line = line.split("//", 1)[0].strip()

            # Skip empty lines
            if not line:
                continue

            # Match either:
            # varName value;
            # varName = value;
            match = re.match(r"^([A-Za-z_]\w*)\s*(?:=\s*|\s+)([^;]+)\s*;\s*$", line)
            if match:
                key = match.group(1)
                value = match.group(2).strip()

                # Convert to int/float if possible
                if re.fullmatch(r"[+-]?\d+", value):
                    value = int(value)
                elif re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[Ee][+-]?\d+)?", value):
                    value = float(value)

                variable_dict[key] = value

    return variable_dict

