# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:29:20 2022

module with functions for calculations related to wind speed

@author: Viet.Le
"""

import logging
import numpy as np
from scipy.stats import weibull_min
from .._utilities._tools import rounder

logger = logging.getLogger(__name__)



#%%

def calc_WSR(Umean,TKE=0,refWS=10,gustadjust=1,
             mode='planes',gust_case='Mean'):
    """
    Reads the wind speed data and normalizes it wrt the reference wind speed
    used in the CFD simulations

    Parameters
    ----------
    Umean : numeric
        Mean wind speed loaded from OF
    TKE : numeric
        TKE loaded from OF, optional. Default is 0
    refWS : numeric
        reference wind speed at the reference height used in OF, default is 0
    gustadjust : numeric
        factors to scale up MWS to mean of gusts (rough equivalent to GEM)
    mode : string
        option to work with plane data from VTKs or point data
    gust_case : string
        option to indicate which type of gust we want to calculate

    Returns
    -------
    normWS : numeric
        OF wind speeds normalized with the input reference wind speed
    windSpeed : 
        OF wind speed adjusted for gust (if specified)

    """
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    
    numDir = Umean.shape[1]
    
    # use this to convert any input provided as a single numeric to a list
    if isinstance(gustadjust, (int, float)):
        gustadjust = [gustadjust]*numDir
    gustadjust = np.array(gustadjust)
        
    if isinstance(refWS, (int, float)):
        refWS = [refWS]*numDir
    refWS = np.array(refWS)
        
    # Adjust the MWS to gustadjusted MWS, if factor is provided
    Umean = Umean * gustadjust
    
    # calculate turbulence intensity
    if gust_case in ['EGV','GEM']:
        TI = calc_Iu(TKE,Umean,
                     flag_ratiocalc=False)
        logger.info('Turbulence intensity calculated')

    
    
    """
    method of calculating gust will change depending on which 
    representation of gust wind speed you choose
    """
            
    # gust equivalent mean
    if gust_case == 'GEM':
        
        # calculate gust wind speed
        GWS = calc_gustWS(TI,Umean)

        # calculate GEM and set equal to the windspeed for normalization
        GEM = calc_GEM(GWS)

        # check which is greater, GEM or gust adjusted MWS
        windSpeed = GEM
        windSpeed[Umean > GEM] = Umean[Umean > GEM]
        logger.info('Gust equivalent mean is used')
        
    # effective gust velocity
    elif gust_case == 'EGV':
        
        EGV = calc_EGV(TI, Umean)
        windSpeed = EGV
        logger.info('Equivalent gust velocity is used')
    
    # mean wind speed
    else:
        
        logger.info('Mean velocity is used')
        windSpeed = Umean
        
    # Normalize the wind speeds by the ref wind speed
    normWS = windSpeed / refWS
    
    # replace NaN with zeros
    normWS = np.nan_to_num(normWS)
    
    # convert NaN to zeroes
    windSpeed = np.nan_to_num(windSpeed)
    normWS = np.nan_to_num(normWS)
    
    np.seterr(invalid='print')
    np.seterr(divide='print')
    
    return normWS, windSpeed



def calc_percWS(freq, shape, scale, 
                WSR_per_direction, scaling_ratio=1, 
                target=0.05, start_max=50, min_speed=0.01, 
                epsilon=0.0001, max_iter=1000):
    """
    Calculate N-th percentile wind speeds using Weibull parameters
    (i.e. the wind speed that is exceed 100-N% of the time)

    Parameters
    ----------
    freq : numeric
        frequency for each valid wind direction, as array.
    shape : numeric
        Weibull shape factor for each valid wind direction, as array.
    scale : numeric
        Weibull scale factor for each valid wind direction, as array.
    WSR_per_direction : array
        matrix of wind speeds at each location for each valid wind direction, 
        normalized by reference wind speed (row = location, column = direction).
    scaling_ratio : TYPE, optional
        multiplier on output velocities. The default is 1.
    target : TYPE, optional
        fraction of wind speeds that may exceed output speed. 
        The default is 0.05 (i.e. 95th percentile wind speed)
    epsilon : TYPE, optional
        allowable error to reach target. The default is 0.001.
    start_max : TYPE, optional
        highest wind speed to consider. The default is 50.
    min_speed : TYPE, optional
        minimum wind speed. The default is 0.01.
    max_iter : TYPE, optional
        number of iterations to allow in fitting. The default is 1000.

    Returns
    -------
    ndarray
        N-th percentile wind speed multiplied by the scaling ratio 
        (usually 1 if the scaling was already performed with the ref WS).

    """
    np.seterr(divide='ignore')
    numDir = WSR_per_direction.shape[1]
    numPoints = WSR_per_direction.shape[0]
    
    # use this to convert any input provided as a single numeric to a list
    if isinstance(freq, (int, float)):
        freq = [freq]*numDir
    freq = np.array(freq)
    if isinstance(shape, (int, float)):
        shape = [shape]*numDir
    shape = np.array(shape)
    if isinstance(scale, (int, float)):
        scale = [scale]*numDir
    scale = np.array(scale)

    vmin = np.zeros(numPoints)
    vmax = np.full_like(vmin, start_max)
    velocity = vmin

    f = freq / freq.sum()
    print("Frequency of each wind direction rescaled to add to 100%")
    scale_local = np.multiply(scale, WSR_per_direction)
    scale_local[scale_local == 0] = min_speed

    # Iterate to calculate velocity
    count_unconverged = numPoints
    while count_unconverged and max_iter:

        survival = np.zeros_like(scale_local)
        
        # for each shape and scale function
        for w in range(len(shape)):
            
            # survival function sf = 1 - cdf
            survival[:,w] = weibull_min.sf(velocity, shape[w], 0, scale_local[:,w]) 
            
        # the error difference, note that the freq is incorporated in the dot product
        error = target - np.dot(survival, f)

        too_high = np.greater(error, epsilon)
        too_low = np.less(error, -epsilon)
        vmax = np.where(too_high, velocity, vmax)
        vmin = np.where(too_low, velocity, vmin)
        velocity = 0.5 * (vmax + vmin)

        count_unconverged = np.count_nonzero(too_high) + np.count_nonzero(too_low)
        
        # prevent infinite loop
        max_iter -= 1 

    if count_unconverged:
        logger.warn(f'Weibull CDF estimation did not converge for \
                      {count_unconverged} points')
    logger.info(f'{(1-target)*100:.4e}-th percentile wind speed calculated')

    np.seterr(divide='print')
    # Return a velocity for each point. Add in additional scale if provided 
    return velocity * scaling_ratio



def calc_Iu(k,meanWS,flag_ratiocalc=False,direction=0):

    """
    Turn on flag_ratiocalc if you want to calculate the turbulence intensity using the Iu, Iv, Iz results from the ESDU file
    TODO: this is still a work-in-progress. By default, I suggest not using it at the moment.
    """
    if flag_ratiocalc:
        
        # # load the ESDU mean wind speed, gust wind speed, Iu, and GWS_ratio profiles

        # # NOTE: You need the final ESDU run with the corrected weibulls
        # ESDU_file = 'E0108v11_OC_to_KevSt - 12dir  - withcorrRefWS.xls'
        # if 'withcorrRefWS' not in ESDU_file:
        #     raise Exception('Make sure you are using the ESDU with the corrected Weibulls as reference wind speed!')

        # # Load in the ESDU data that's been cleaned and tidied up
        # ESDU_filepath = os.path.join('02_CLIMATE',ESDU_file)
        # ESDU_Iu, ESDU_Lux, ESDU_Luy, ESDU_Luz = load_and_prep_ESDUturbprop(ESDU_filepath,
        #                                                sheetname_turb='u-component turbulence')
        # ESDU_Iv, ESDU_Lvx, ESDU_Lvy, ESDU_Lvz = load_and_prep_ESDUturbprop(ESDU_filepath,
        #                                                sheetname_turb='v-component turbulence')
        # ESDU_Iw, ESDU_Lwx, ESDU_Lwy, ESDU_Lwz = load_and_prep_ESDUturbprop(ESDU_filepath,
        #                                                sheetname_turb='w-component turbulence')

        # # get the ratio between TI of different directions at various elevation as predicted from the ESDU
        # ratio_IuIu = ESDU_Iu[direction][ESDU_Iu['z'] == 10].to_numpy() / \
        #     ESDU_Iu[direction][ESDU_Iu['z'] == 10].to_numpy()
        # ratio_IuIv = ESDU_Iu[direction][ESDU_Iv['z'] == 10].to_numpy() / \
        #     ESDU_Iu[direction][ESDU_Iu['z'] == 10].to_numpy()
        # ratio_IuIw = ESDU_Iu[direction][ESDU_Iw['z'] == 10].to_numpy() / \
        #     ESDU_Iu[direction][ESDU_Iu['z'] == 10].to_numpy()

        # # add ratios together
        # ratio_TI = ratio_IuIu + ratio_IuIv + ratio_IuIw

        # """
        # calculate turbulence intensity using elevation dependent ratios
        # between turbulence components
        # """
        # turbInt=(2/ratio_TI*k)**(0.5)/meanWS
        
        # TODO: implement ratios for ESDU 
        # turbInt=(2*k)**(0.5)/meanWS

        logger.warn("Note that TI components (u,v,w) are not assumed all the same \n")

    else:
        np.seterr(divide='ignore')
        turbInt=(2/3*k)**(0.5)/meanWS
        np.seterr(divide='print')
        

    return turbInt



def calc_EGV(Iu,meanWS):
    """
    calculates Effective Gust Velocity (BPDA)

    Parameters
    ----------
    Iu : ndarray
        turbulence intensity.
    meanWS : ndarray
        mean wind speed.

    Returns
    -------
    egv : ndarray
        effective gust velocity.

    """
    
    # standard deviation of the fluctuating wind speed (in this case, root-mean-square)
    rms_ws = Iu * meanWS 
    
    # effective gust velocity
    egv = meanWS + 1.5*rms_ws
        
    return egv



def calc_gustWS(Iu,meanWS):
    """
    Calculates the gust wind speed via approximation based on turbulence 
    intensity

    Parameters
    ----------
    Iu : ndarray
        turbulence intensity.
    meanWS : ndarray
        mean wind speed.

    Returns
    -------
    gws : ndarray
        gust wind speed.

    """
    gws=meanWS*(1+3*Iu)
    return gws



def calc_GEM(gustWS):
    """
    Calculates the gust effective mean

    Parameters
    ----------
    gustWS : ndarray
        gust wind speed.

    Returns
    -------
    ndarray
        gust effective mean.

    """
    return gustWS/1.85



def calc_rescale_WSR_EPW(wsr, ws, wd, directions=[0,30,60,90,120,150,180,210,240,270,300,330], 
    TF1=None, TF2=None, TF3=None):
    """
    TF1 is TF_airport_OC_10m
    TF2 is TF_OC_site_refheight
    TF3 is TF_1p5m_to_10m
    
    set TF3 to false for non-UTCI

    """
    
    def extend(T):
        if (T is None):
            T = 1
        if not hasattr(T, '__len__'):
            T = np.array(T)
        if np.array(1).size == 1:
            T = T*np.full(directions.shape, T)
        
        return T
    
    TF1 = extend(TF1)
    TF2 = extend(TF2)
    TF3 = extend(TF3)
    
    # round the wind direction of the EPW to the closest 12 directional sector
    b = np.array(directions)
    roundedwindDir = rounder(b)(wd)

    # determine the index of the direction each entry of roundedwindDir falls into
    direction_index = np.digitize(roundedwindDir, directions, right=True)

    # tranpose the EPW wind at airport to the site

    # AP to OC at zref = 10
    TF1 = np.tile(TF1,(wsr.shape[0],1)) 

    # OC to site at zref
    TF2 = np.tile(TF2,(wsr.shape[0],1))  

    # pedestrian level to 10 m
    TF3 = np.tile(TF3,(wsr.shape[0],1))  

    # broadcast the TFs so that the columns are extracted and are assigned to 
    # each hour via direction_index as a reference
    TF1_EPW = TF1[:,direction_index]
    TF2_EPW = TF2[:,direction_index]
    TF3_EPW = TF3[:,direction_index]



    """
    scale the WSR at each location (rows) by the appropriate wind speed and 
    direction for each hour.

    we now have the wind velocity at the site scaled appropriately by the wind 
    speed and direction at each hour based on the EPW file.

    ws * TF1_EPW * TF2_EPW gives us the wind speed at reference height 
    at the site. 
    This is used to rescale the WSR.

    TODO-NOTE: there is NO conversion from 1.5m to 10m 
    (the conversion is needed only for calculating UTCI and NOT for SET)
    """
    U_each_hr = wsr[:,direction_index] \
        * ws \
            * TF1_EPW \
                * TF2_EPW \
                    * TF3_EPW
    U_each_hr = np.nan_to_num(U_each_hr,0)
    
    return U_each_hr














