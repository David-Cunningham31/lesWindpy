# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:29:20 2022

@author: Viet.Le
"""

import logging
import numpy as np
import pandas as pd
import cycler as cy
import seaborn as sbn
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



#%% Functions for CFD probe analysis

def gen_coords_str(coords):
    
    coords = coords[['x','y','z']].applymap(lambda x: str(round(x,2)))
    probe_ID_str = [str(value)+': ' for index, value in enumerate(coords.index)]
    coords_str = coords['x'] + ', ' + coords['y'] + ', ' + coords['z']
    coords_str = pd.Series(probe_ID_str, index=coords_str.index).add('').add(coords_str)
    
    return coords_str

def plot_boxplots_stability(data,coords,numIterback=10,
                            fig=None,plot_title=None,legloc='upper right'):
    
    """
    
    Plots boxplots of the desired variable for each probe point. 
    Use to gauge the stability of the CFD simulation at various probe locations
    
    Parameters
    ----------
    data : ARRAY/FRAME
        The wind speed data.
    coords : ARRAY
        The coordinates x, y, z for each point. 
    numIterback : INT, optional
        Number of iterations to sample back from the end for data to plot. 
        The default is 10.
    fig : MATPLOTLIB FIGURE, optional
        Figure to plot on. If None, a figure will be created and output.
    plot_title : STR, optional
        The plot title. The default is None.
    legloc : STR, optional
        The legend location. The default is 'upper right'.
        
    Returns
    -------
    fig : MATPLOTLIB FIGURE
        The output figure.

    """
    
    # setup the figure
    if fig is None:
        fig = plt.figure(figsize=(15,6))
        ax = fig.add_subplot(111)
        
    B = plt.boxplot(data,
                showmeans=True,
                meanline=True,
                boxprops={'color':'grey','linewidth':1,'alpha':0.7},
                meanprops={'color':'blue', 'linewidth': 1},
                medianprops={'color': 'red'},
                flierprops={'marker': '.', 'markersize': 8, 'markerfacecolor': 'fuchsia','markeredgewidth':0.5})
    
    """
    Make the legend
    """
    legend_elements = [Line2D([0], [0], color='r', lw=1, linestyle='-', label='Median'),
                       Line2D([0], [0], color='b', lw=1, linestyle='--', label='Mean'),
                       Patch(facecolor='white', edgecolor='gray', alpha=0.7,
                             label='IQR = Q3 - Q1'),
                       Line2D([0], [0], color='k', lw=1, linestyle='-', label='Q3+1.5IQR \nQ1-1.5IQR'),
                       Line2D([0], [0], marker='o', color='w', label='Outlier > Q3+1.5IQR \nOutlier < Q1-1.5IQR',
                          markerfacecolor='fuchsia', markersize=8),]
    ax.legend(handles=legend_elements, loc='best')
    
    
    """
    set the x-labels using the coordinates
    """
    plt.grid(linestyle=':',alpha=0.5)
    xtick_label = coords[:].tolist()
    plt.xticks(np.arange(0,data.shape[1],1)+1, xtick_label, rotation=90)

    """
    global plot details
    """
    plt.title(f"Variation in probe values \n{plot_title}")
    plt.ylabel('Wind Speed [m/s]')
    plt.xlabel('Probes')
    ax.set_ylim([0,20])
    
    plt.tight_layout()

    logger.debug("Boxplots made")
    
    return fig



def plot_relerror_U(data,coords,fig=None,
    plot_title=None,numIterback=10,
    legloc='lower right',
    ylim=20):

    """
    
    Plots two sided figure with markers to indicate the wind speed at
    probe locations averaged over select number of iterations and to indicate
    the percentage change between max and min wind speed over the same number of
    iterations
    
    Parameters
    ----------
    data : ARRAY/FRAME
        The wind speed data.
    coords : ARRAY
        The coordinates x, y, z for each point. 
    numIterback : INT, optional
        Number of iterations to sample back from the end for data to plot. 
        The default is 10.
    fig : MATPLOTLIB FIGURE, optional
        Figure to plot on. If None, a figure will be created and output.
    plot_title : STR, optional
        The plot title. The default is None.
    legloc : STR, optional
        The legend location. The default is 'upper right'.
        
    Returns
    -------
    fig : MATPLOTLIB FIGURE
        The output figure.

    """
    
    # setup the figure
    if fig is None:
        fig = plt.figure(figsize=(15,6))
        ax1 = fig.add_subplot(111)

    # automatically sets the x-coordinate as the index
    lns1 = ax1.plot(data['err'].tolist(),
                    'ro',
                    label = 'WS rel. to max in last ' + str(numIterback) + ' iterations')
    xtick_label = coords[:].tolist()
    plt.xticks(np.arange(0,data['err'].shape[0],1), xtick_label, rotation=90)
    plt.ylim(ymin=0.0, ymax=1)
    plt.ylabel('(max - min) / avg', color='red')
    plt.xlabel('Probes')

    ax1.tick_params(axis='y', colors='red')



    """
    Plot the mean wind speed of the last "timeStep" timesteps on RHS
    """

    # duplicate the axis to the RHS
    ax2 = ax1.twinx()

    # automatically sets the x-coordinate as the index
    lns2 = ax2.plot(data['U'].tolist(),
                    'bx',
                    label = 'WS mean in last ' + str(numIterback) + ' timesteps')
    plt.ylabel('Mean wind speed in last ' + str(numIterback)+ ' iter [m/s]', color='blue') 



    """
    global plot details
    """

    # title
    plt.title("Variation in probe values \n"+plot_title)

    # combined legend
    lns = lns1+lns2
    labs = [l.get_label() for l in lns1+lns2]
    ax1.legend(lns, labs, loc=legloc, bbox_to_anchor=(1, 1.07))
    ax2.tick_params(axis='y', colors='blue')
    ax2.set_ylim([0,ylim])

    # change fontsize of tick labels
    ax = fig.axes
    ax[0].tick_params(axis='x',labelsize=8)

    plt.tight_layout()

    return fig



def plot_vel_profile(data,fig=None,origin={'x': 0, 'y': 0},title=None,plot_label=None):

    """
    
    Plots velocity profiles from the CFD results
    
    Parameters
    ----------
    data : ARRAY/FRAME
        The wind speed data.
    fig : MATPLOTLIB FIGURE, optional
        Figure to plot on. If None, a figure will be created and output.
    origin : Dictionary with floats, optional
        The central location of the domain
    title : STR, optional
        The plot title. The default is None.
    plot_label : STR, optional
        The legend location. The default is 'upper right'.
        
    Returns
    -------
    fig : MATPLOTLIB FIGURE
        The output figure.

    """
    
    # setup the figure
    if fig is None:
        # subplot for velocity and gust velocity ratio profiles
        fig = plt.figure(figsize = (18,10))
        ax1a = fig.add_subplot(121)
        ax1b = fig.add_subplot(122)
    else:
        ax1a = fig.axes[0]
        ax1b = fig.axes[1] 

    
    if plot_label is None:
        dist_from_origin = np.sqrt((data['x'].iloc[0] - origin['x'])**2 + (data['y'].iloc[0] - origin['y'])**2)
        plot_label = f'{dist_from_origin:.2f}m from origin'

    # velocity profile
    ax1a.plot(data["U"], data["z"], 'o-',
                    label=plot_label)

    # gust velocity ratio profile
    ax1b.plot(data["GWS_ratio"], data["z"], 'o-')

    """
    global plot details
    """
    ax1a.legend(loc='best')
    ax1a.set_xlabel('Velocity [m/s]')
    ax1a.set_ylabel('Elevation [m]')
    ax1a.set_ylim((0, 300))
    ax1a.set_xlim((0, 30))

    ax1b.set_xlabel('Gust wind speed ratio')
    ax1b.set_ylabel('Elevation [m]')
    ax1b.set_ylim((0, 300))
    ax1b.set_xlim((0, 4))
    fig.suptitle(title)

    # tight figure layout that considers presence of suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # reduce extra white space below the suptitle
    fig.subplots_adjust(top=0.95)

    return fig



def plot_turb_profile(data,fig=None,origin={'x': 0, 'y': 0},title='',plot_label=None):

    """
    
    Plots turbulence profiles from the CFD results
    
    Parameters
    ----------
    data : ARRAY/FRAME
        The turbulence data.
    fig : MATPLOTLIB FIGURE, optional
        Figure to plot on. If None, a figure will be created and output.
    origin : Dictionary with floats, optional
        The central location of the domain
    title : STR, optional
        The plot title. The default is None.
    plot_label : STR, optional
        The legend location. The default is 'upper right'.
        
    Returns
    -------
    fig : MATPLOTLIB FIGURE
        The output figure.

    """
    
    # setup the figure
    if fig is None:
        # subplot for velocity and gust velocity ratio profiles
        fig = plt.figure(figsize = (18,10))
        ax1a = fig.add_subplot(121)
        ax1b = fig.add_subplot(122)
    else:
        ax1a = fig.axes[0]
        ax1b = fig.axes[1] 

    
    if plot_label is None:
        dist_from_origin = np.sqrt((data['x'].iloc[0] - origin['x'])**2 + (data['y'].iloc[0] - origin['y'])**2)
        plot_label = f'{dist_from_origin:.2f}m from origin'

    # TKE profile
    ax1a.plot(data["k"], data["z"], 'o-',
                    label= plot_label)

    # TI profile
    ax1b.plot(data["Iu"], data["z"],'o-',)
    
    """
    global plot details
    """
    ax1a.legend(loc='best')
    ax1a.set_xlabel('TKE [m^2/s^2]')
    ax1a.set_ylabel('Elevation [m]')
    ax1a.set_xlim((0, 20))
    ax1a.set_ylim((0, 300))

    ax1b.set_xlabel('Iu')
    ax1b.set_ylabel('Elevation [m]')
    ax1b.set_ylim((0, 300))
    ax1b.set_xlim((0, 0.5))
    fig.suptitle(title)

    # tight figure layout that considers presence of suptitle
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # reduce extra white space below the suptitle
    fig.subplots_adjust(top=0.95)
    
    return fig



#%% Barcharts

def stackBarSeries(df,
             colorSet=['#204772','#09b5f3','#0db25a','#e87a38','#c40404'],
             labelSet=['Cold', 'Cool', 'Comfortable', 'Warm', 'Hot'],
             xlabeltext='Month',
             title=''):
    
    """
    Plots stacked barchart, separated by the specified bins, for a SERIES of bars

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    colorSet : TYPE, optional
        DESCRIPTION. The default is ['#204772','#09b5f3','#0db25a','#e87a38','#c40404'].
    labelSet : TYPE, optional
        DESCRIPTION. The default is ['Cold', 'Cool', 'Comfortable', 'Warm', 'Hot'].
    xlabeltext : TYPE, optional
        DESCRIPTION. The default is 'Month'.
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    Returns
    -------
    fig_bar : TYPE
        DESCRIPTION.

    """
    
    r = np.arange(1,df.shape[0]+1)
    if xlabeltext == 'Month' and 'Annual' in df.index.values:
        r[-1] = 14
    numdiv = df.shape[1]

    # From raw value to percentage
    totals = []
    rBar = np.zeros([len(r),numdiv])
    for rr in range(0,len(r),1):
        totals.append(df.iloc[rr].sum())
        
        for nn in range(0,numdiv):
            rBar[rr,nn] = df.iloc[rr,nn] / totals[rr] * 100

    # Plot
    if len(r) == 1:
        barWidth = 0.15
        fig_bar = plt.figure(figsize=(5,5))
    else:
        barWidth = 0.85
        fig_bar = plt.figure(figsize=(10,5))
    barEdges='white'
    
    # Create the color bars
    tempbottom = np.zeros(len(r),)
    for nn in range(0,numdiv):
        plt.bar(r, rBar[:,nn], bottom=tempbottom, color=colorSet[nn], edgecolor=barEdges, width=barWidth, label=labelSet[nn])
        tempsum = np.sum(rBar[:,0:(nn+1)], axis=1)
        tempbottom = tempsum

    # Custom x axis
    if xlabeltext.capitalize() == 'Month':
        names = ('Jan', 'Feb', 'Mar', 'Apr', 'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Year')
        plt.xticks(r, names, fontsize=18)
    elif xlabeltext.capitalize() == 'Season':
        names = df.index.values
        plt.xticks(r, names, fontsize=18)
    else:
        plt.xticks([])
    plt.yticks(fontsize=18)
    plt.xlabel(xlabeltext,fontsize=18)
    plt.ylabel("Percentage of time (%)",fontsize=18)
    plt.ylim([0,100])
    
    plt.title(title,fontsize=18)

    # # Add a legend
    # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    
    # #TODO: reverse the legend
    # ax = plt.gca()
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    # tighten the layout
    plt.tight_layout()
    
    return fig_bar



#%% Heat matrix

def heat_map(df,flag_disc_cb=True,valmin=0,valmax=12,
             title='',cbar_label='',colmap='jet',flag_cbar=True,num_ticks=13):
    
    """
    Plots average data as a heat matrix against months of the year and hours 
    in the day

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    flag_disc_cb : TYPE, optional
        DESCRIPTION. The default is True.
    valmin : TYPE, optional
        DESCRIPTION. The default is 0.
    valmax : TYPE, optional
        DESCRIPTION. The default is 12.
    title : TYPE, optional
        DESCRIPTION. The default is ''.
    cbar_label : TYPE, optional
        DESCRIPTION. The default is ''.
    colmap : TYPE, optional
        DESCRIPTION. The default is 'jet'.
    flag_cbar : TYPE, optional
        DESCRIPTION. The default is True.
    num_ticks : TYPE, optional
        DESCRIPTION. The default is 13.

    Returns
    -------
    fig_heatmap : TYPE
        DESCRIPTION.

    """   
    
    # color bar properties
    if flag_disc_cb:
        ticks_cb = np.linspace(valmin,valmax,num_ticks)
        qrates = ["{:.0f}".format(x) for x in ticks_cb]
        qrates.reverse()
        norm = mpl.colors.BoundaryNorm(np.linspace(valmin, valmax, num_ticks), num_ticks-1)
        fmt = mpl.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])
        cb_dictionary = {"orientation": "horizontal",'ticks':ticks_cb,'format':fmt,'label':cbar_label}
        cmap_prop = plt.get_cmap(colmap, num_ticks-1)
    else:
        cb_dictionary = {"orientation": "horizontal"}
        cmap_prop = plt.get_cmap(colmap)
    
    # make the heatmap plot
    grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    if flag_cbar:
        fig_heatmap, (ax_heatmap, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
        ax_heatmap = sbn.heatmap(df, vmin=valmin, vmax=valmax, cmap=cmap_prop,
                         linewidths=0.30, yticklabels=True,
                         ax=ax_heatmap, cbar_ax=cbar_ax, cbar_kws=cb_dictionary)
    else:
        fig_heatmap = plt.figure()
        ax_heatmap = sbn.heatmap(df, vmin=valmin, vmax=valmax, cmap=cmap_prop,
                         linewidths=0.30, yticklabels=True, cbar=False)
    fig_heatmap.set_size_inches(11,6.8)
    
    """
    global plot details
    """
    ax_heatmap.set_title(title)
    ax_heatmap.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    ax_heatmap.set_ylabel('Hours of the day')
    ax_heatmap.set_xlabel('')
    yticks = np.arange(0,26,2)
    ax_heatmap.set_yticks(yticks)
    ax_heatmap.set_yticklabels(yticks.astype(str))
    
    # flip the y-axis so that 0hr starts at bottom
    ax_heatmap.invert_yaxis()

    return fig_heatmap