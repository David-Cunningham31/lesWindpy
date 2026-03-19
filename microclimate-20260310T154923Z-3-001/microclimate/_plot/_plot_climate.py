# -*- coding: utf-8 -*-
"""
Created on Fri Dec 9 11:29:20 2022

@author: Viet.Le
"""

import logging
import numpy as np
import pandas as pd
import cycler as cy
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)



#%% MPL settings using Arup-themed colors and fonts

def setUpMPL(FS=12):
    # Import Arup-brand colors
    colors = [
    '#E61E28', # arup red
    '#7D4196', # purple
    '#005AAA', # dark blue
    '#32A4A0', # teal
    '#C83C96', # pink
    '#4BA046', # green
    '#1E9BD7', # dark baby blue
    '#91967D', # tan
    '#E66E23', # orange
    '#50697D' # dark blue-gray
    ]
    linestyles = ['-', '--', '-.', ':']
    # set font
    mpl.rc("font", family="Times New Roman",size=FS)
    # mpl.rc("figure", figsize=(4,4), dpi=150)
    mpl.rc("axes", prop_cycle=cy.cycler(linestyle=linestyles) * cy.cycler(color=colors))



def plot_PofE(directions,PofE,leg_entry=None,ylabel=None,last=False,title='',current_ax=None):
    current_ax = plt.gca()


    plt.sca(current_ax)
    ax = plt.plot(directions,PofE,'-',label=f'{leg_entry} m/s')
    ax = plt.plot(directions,PofE,'*',color=ax[0]._color)
    plt.xlim([min(directions), max(directions)])
    # plt.ylim([0, 25])
    plt.xticks(directions)

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel(ylabel)
    if leg_entry is not None:
        plt.legend(loc='best')
    plt.title(f'{title}')

    plt.tight_layout()
    fig_PofE = plt.gcf()
    
    return fig_PofE, current_ax














    