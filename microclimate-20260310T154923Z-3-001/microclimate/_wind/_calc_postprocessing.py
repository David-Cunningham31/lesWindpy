# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:29:20 2022

Module for postprocessing the wind calculations

@author: Viet.Le
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

logger = logging.getLogger(__name__)



#%% QA functions

def collect_for_QA(vtk_values,U50,U95,U99p975,num_hrs_excd,
                   coordinates,directions,
                   U_per_direction,dir_contribution):
    """
    Collects the most relevant information from the probe analysis for QA

    Parameters
    ----------
    vtk_values : ndarray
        Categorization of the wind speed via Lawson criteria.
    U50 : ndarray
        50th percentile wind.
    U95 : ndarray
        95th percentile wind.
    U99p975 : ndarray
        99.975th percentile wind.
    num_hrs_excd : ndarray
        number of hours in a year that a particularly velocity has been exceeded.
    coordinates : ndarray
        DESCRIPTION.
    directions : ndarray
        DESCRIPTION.
    U_per_direction : ndarray
        OF wind speeds.
    dir_contribution : ndarray
        contribution to percentile wind speed from each direction.

    Returns
    -------
    df_check : dataframe
        dataframe with all the relevant QA information.

    """
    
    
    data = np.column_stack((vtk_values,U50,U95,U99p975,num_hrs_excd))
    col_list = (['Rank','U50','U95','U99p975','hrs_distress'])

    """
    Collect direction contributions and percent check for 95th percentile
    """
    data_dirpercent95 = np.append(dir_contribution[95],
                  np.sum(dir_contribution[95],axis=1).reshape(-1,1),
                  axis=1)
    col_list.extend([f'{direction} comf' for direction in directions])
    col_list.append('Total - 95 [%]')
    


    """
    Collect direction contributions and percent check for 99.975th percentile
    """
    data_dirpercent99p975 = np.append(dir_contribution[99.975],
                  np.sum(dir_contribution[99.975],axis=1).reshape(-1,1),
                  axis=1)
    col_list.extend([f'{direction} dist' for direction in directions])
    col_list.append('Total - 99.975 [%]')
    
    
    
    """ 
    Collect velocities from each direction
    """
    data_U = U_per_direction
    col_list.extend(f'U{direction}' for direction in directions)



    """ 
    Collect all arrays together into one
    """
    data = np.hstack((data,data_dirpercent95,data_dirpercent99p975,data_U))

    # create dataframe of all QA checks
    df_check = pd.DataFrame(data, columns = col_list)
    df_check = df_check.join(coordinates.set_index(df_check.index))

    # rearrange dataframe so x,y,z comes first
    columns_xyz = (['x','y','z'])
    columns_xyz.extend(col_list)
    df_check = df_check[columns_xyz]
    
    logger.info("All relevant QA information collected together in dataframe")
    
    return df_check



#%% CFD postprocessing

def calc_prob_exceedance(U_target,wsr,weibull_param):
    """
    Calculates the probability of exceeding a target wind speed, using the
    input wind speed ratios and Weibull climate parameters

    Parameters
    ----------
    U_target : numeric, single value
        target wind speed for exceedance.
    wsr : numeric
        Array of wind speed ratios.
    weibull_param : tuple
        weibull parameters listed in order: frequency, scale, shape.

    Returns
    -------
    perc_total_target : numeric, array
        the probability that U_target is exceeded, calculated for each point.
    num_hrs_exceed : numeric, array
        number of hours in a year that are equal to or greater than U_target.

    """
    
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')

    # unpack the Weibull parameter tuple
    freq, scale, shape = weibull_param        

    # for each wind direction, what is the probability that wind speed 
    # exceeds U_target (typically 15 m/s)
    weibull_mod_c_factor = wsr * scale
    dir_contribute_target = freq / 100 * weibull_min.sf(
        U_target,shape,0,weibull_mod_c_factor)

    # convert any NaN to 0
    dir_contribute_target = np.nan_to_num(dir_contribute_target)

    # add together the probabilities from each wind direction to get
    # the probability that wind speed exceeds the target
    perc_total_target = np.sum(dir_contribute_target,axis=1)

    # number of hours in a year that are equal to or greater than U_target
    # round to the closest integer
    num_hrs_exceed = np.rint(perc_total_target * 8760)

    logger.info(f"Calculated the probability that U_target ({U_target} m/s) is exceeded")
    logger.info(f"Calculated the number of hours in a year that U_target ({U_target} m/s) is exceeded")
    
    np.seterr(invalid='print')
    np.seterr(divide='print')
    
    return perc_total_target, num_hrs_exceed



def calc_direction_contribution(wsr,U_percentile,weibull_param):
    """
    Evaluates the contribution that each direction provides to the percentile
    wind speed. Selects the highest contributing direction

    Parameters
    ----------
    wsr : numeric
        matrix of wind speed ratios.
    U_percentile : numeric
        the N-th percentile wind speed.
    weibull_param : tuple
        weibull parameters listed in order: frequency, scale, shape.

    Returns
    -------
    dir_contribute : numeric
        matrix of probability from each wind direction.
    check_stats : DataFrame
        statistics of the check for the accumulated percentage at each point.

    """
    
    np.seterr(invalid='ignore')
    np.seterr(divide='ignore')
    
    # unpack the Weibull parameter tuple
    freq, scale, shape = weibull_param
    
    # modified weibull factor 
    # (transferred from airport to each point of the site at ref height)
    weibull_mod_c_factor = wsr * scale

    # convert the vector to 2D array with one dimension
    U = U_percentile.reshape(-1,1)

    # evaluate the survival function using the scaled weibull factor and the wind speed
    dir_contribute = freq * weibull_min.sf(U,shape,0,weibull_mod_c_factor)
    dir_contribute = np.nan_to_num(dir_contribute)
    
    
    
    """
    check that all the wind speed calcs gives the target percentile.
    produces the statistics across all the points
    """
    perc_check = np.sum(dir_contribute,axis=1)
    check_stats = pd.DataFrame(perc_check[perc_check != 0]).describe()
    
    logger.info("Calculated the directional contribution to the percentile wind speed")
    logger.info("Determined the major contributing direction for each point")
    
    np.seterr(invalid='print')
    np.seterr(divide='print')
    
    dir_contribute = np.where(np.abs(dir_contribute) < 1e-10, 0, dir_contribute)
    
    return dir_contribute, check_stats



def calc_most_contrib_windDirs(contribution,directions,flag_dirlabels=False):
    """
    Determine the most and second most contributing wind directions for points
    

    Parameters
    ----------
    contribution : ndarray
        contribution to the percentile wind speed for each direction.
    directions : array
        wind directions.

    Returns
    -------
    U_1 : numeric
        Vector components for prevailing wind direction
    U_2 : numeric
        Vector components for 2nd most prevailing wind direction

    """

    # for each point, choose the index for the prevailing wind direction that
    # contributes the most to the N-th percentile wind speed. Then repeat for
    # the second most contributing direction
    ind_dir_prevailing = np.argmax(contribution,axis=1)
    ind_dir_prevailing_2 = np.argsort(contribution,axis=1)[:,-2]
    
    # use the index to determine most and 2nd most contributing wind direction
    dir_1 = np.array(directions)[ind_dir_prevailing]
    dir_2 = np.array(directions)[ind_dir_prevailing_2]
    
    # evaluate the vector components (prior to 2023-08-31, this calc was wrong)
    U_1x = np.cos(np.radians((dir_1+90) % 360))
    U_1y = np.sin(np.radians((dir_1-90) % 360))
    U_2x = np.cos(np.radians((dir_2+90) % 360))
    U_2y = np.sin(np.radians((dir_2-90) % 360))
    
    # put together the components
    U_1 = np.stack((U_1x, U_1y, np.zeros(U_1x.shape)), axis=1)
    U_2 = np.stack((U_2x, U_2y, np.zeros(U_2x.shape)), axis=1)

    # option to just get the name of the most contributing direction instead of the vector
    if flag_dirlabels:
        U_1 = np.take(directions, ind_dir_prevailing)
        U_2 = np.take(directions, ind_dir_prevailing_2)

    return U_1, U_2



def comfydistrank(Ucomf,Udist,activity='sitting'):
    """
    Ranks the comfort (typically 95th percentile) and
    distress (typically 99.975th percentile) wind speeds

    Parameters
    ----------
    Ucomf : numeric
        comfort wind speed.
    Udist : numeric
        distress wind speed.
    activity : activity, optional
        the desired function of the point or space. The default is 'sitting'.

    Returns
    -------
    ranking_for_comfort : array of integers
        ranking of the comfort wind speed.
    ranking_for_distress : array of integers
        ranking of the distress wind speed.

    """

    rank_distress = np.zeros(len(Udist),)
    rank_comfort = np.zeros(len(Ucomf),)



    """
    distress
    if the 99.975th percentile wind speed is greater than 15 m/s, then rank 4
    if the 99.975th percentile wind speed is greater than 20 m/s, then rank 5
    """
    rank_distress[np.where(Udist >= 20, True, False)] = 5
    rank_distress[np.where((Udist >= 15) & (Udist < 20), True, False)] = 4



    """
    comfort
    """
    # if the 95th percentile wind speed is greater than 15 m/s, then rank 4
    # if the 95th percentile wind speed is greater than 20 m/s, then rank 5
    if activity.lower() == 'sitting':
        limit = 4
    elif activity.lower() == 'standing':
        limit = 6
    elif activity.lower() in ['strolling','walking']:
        limit = 8
    elif activity.lower() == 'business walking':
        limit = 10

    rank_comfort[Ucomf >= 20] = 5
    rank_comfort[np.where((Ucomf >= 15) & (Ucomf < 20), True, False)] = 4
    rank_comfort[np.where((Ucomf >= limit+2) & (Ucomf < 15), True, False)] = 3
    rank_comfort[np.where((Ucomf >= limit) & (Ucomf < limit+2), True, False)] = 2
    rank_comfort[np.where(Ucomf < limit, True, False)] = 1
    
    # the vtk value will be same as comfort, unless there is a distress wind speed
    vtk_value = rank_comfort.astype(int)
    idx_distress = np.where(rank_distress > 0, True, False)
    vtk_value[idx_distress] = rank_distress[idx_distress].astype(int)
    
    logger.info('Evaluated comfort and distress rank for each point in VTK')
    
    return rank_comfort, rank_distress, vtk_value



#%%

def binLawson(windSpeed,
              thresholdSittingLong=2,
              thresholdSittingShort=4,
              thresholdStanding=6,
              thresholdStrolling=8,
              thresholdWalking=10,
              thresholdUnsafeFrail=15,
              thresholdUnsafeAll=20,
              ):
    
    """
    Percentage of annual hours within Lawson wind speed activity bands based on 
    95th percentile winds

    Parameters
    ----------
    windSpeed : ndarray
        Array of 95th percentile wind speeds values with one row per point and 
        one column per hour (for the full year).

    Returns
    -------
    sitlong, sitshort, stand, stroll, businesswalk, uncomfortable, frail, 
    unsafe : float
        percentage of hours within the year that 95th percentile wind is suitable
        for long periods of sitting, short periods of sitting, standing, walking
        around, faster-paced walking, uncomfortable, harmful to the frail
    """    
    
    windSpeed = np.ravel(windSpeed)
    a = windSpeed.ndim - 1
    hours = windSpeed.shape[a]

    """
    TODO: unsafe and frail should be using the distress wind speed
    (99.975th percentile) instead of comfort (95th percentile) ????
    """
    frail = np.count_nonzero((windSpeed > thresholdUnsafeFrail)) / hours
    unsafe = np.count_nonzero(windSpeed >= thresholdUnsafeAll) / hours
    
    # determine how often wind speeds are comfortable for each activity
    uncomfortable = np.count_nonzero(
        (windSpeed > thresholdWalking)) / hours    
    businesswalk = np.count_nonzero(
        (windSpeed > thresholdStrolling) & (windSpeed <= thresholdWalking)) / hours   
    stroll = np.count_nonzero(
        (windSpeed > thresholdStanding) & (windSpeed <= thresholdStrolling)) / hours    
    stand = np.count_nonzero(
        (windSpeed > thresholdSittingShort) & (windSpeed <= thresholdStanding)) / hours
    sitshort = np.count_nonzero(
        (windSpeed > thresholdSittingLong) & (windSpeed <= thresholdSittingShort)) / hours    
    sitlong = np.count_nonzero(
        (windSpeed < thresholdSittingLong)) / hours
    logger.info('Wind speed has been binned based on Lawson thresholds')
    
    return sitlong, sitshort, stand, stroll, businesswalk, uncomfortable, frail, unsafe



def binBPDA(windSpeed,
            thresholdSitting = 12,
            thresholdStanding = 15,
            thresholdWalking = 19,
            thresholdUncomfortable = 27,
            ):
    
    """
    Percentage of annual hours within BPDA wind speed activity bands based on 
    available wind speed (in mph)

    Parameters
    ----------
    windSpeed : ndarray
        MUST BE IN MPH
        Array of 99th percentile mean wind speeds with one row per point and 
        one column per hour (for the full year).

    Returns
    -------
    sitting, standing, walking, uncomfortable, dangerous : float
        percentage of hours within the year that EGV is categorized under BPDA bands
    """    
   
    windSpeed = np.ravel(windSpeed)
    a = windSpeed.ndim - 1
    hours = windSpeed.shape[a]
    
    # percentage of time that wind speeds are UNCOMFORTABLE or DANGEROUS
    dangerous = np.count_nonzero(
        (windSpeed >= thresholdUncomfortable)) / hours
    uncomfortable = np.count_nonzero(
        (windSpeed >= thresholdWalking) & (windSpeed < thresholdUncomfortable)) / hours

    # percentage of time that wind speeds are COMFORTABLE
    standing = np.count_nonzero(
        (windSpeed >= thresholdSitting) & (windSpeed < thresholdStanding)) / hours
    walking = np.count_nonzero(
        (windSpeed >= thresholdStanding) & (windSpeed < thresholdWalking)) / hours
    sitting = np.count_nonzero(
        (windSpeed < thresholdSitting)) / hours

    return sitting, standing, walking, uncomfortable, dangerous



def binSF(EWS90_7AM_6PM,EWS99p989_7AM_6PM):
    # TODO: fix this to be like the toher bads


    """
    Percentage of annual hours within Lawson wind speed activity bands based on 
    90th percentile mean wind speeds

    Parameters
    ----------
    EWS90_7AM_6PM : ndarray
        MUST BE IN MPH
        Array of 90th percentile equivalent wind speeds with one row per point and 
        one column per hour (for the full year).
        Equivalent wind speed (EWS) is an hourly mean that incorporates gust
        Must be between 7AM and 6PM
    EWS99p975_7AM_6PM : ndarray
        MUST BE IN MPH
        Array of 99.989th percentile (1 hour in a year) equivalent wind speeds with one row per point and 
        one column per hour (for the full year).
        Equivalent wind speed (EWS) is an hourly mean that incorporates gust
        Must be between 7AM and 6PM

    Returns
    -------
    sitting, walking, hazardous : float
        percentage of hours within the year that the 90th percentile mean wind 
        speeds are comfortable for sitting and walking 
        or
        99.989th percentile wind is hazardous
    """    

    # SF Criteria. any velocity below is considered comfortable
    thresholdSitting = 7 # mph
    thresholdWalking = 11 # mph
    
    # any velocity greater than this is considered hazaroud
    thresholdHazardous = 26 # mph
    
    a = windSpeed.ndim - 1
    hours = windSpeed.shape[a]
    
    # percentage of time that wind speeds are COMFORTABLE
    sitting = np.count_nonzero(EWS90_7AM_6PM < thresholdSitting, axis=a) / hours
    walking = np.count_nonzero(EWS90_7AM_6PM < thresholdWalking, axis=a) / hours
    
    # percentage of time that wind speeds are UNCOMFORTABLE or DANGEROUS
    hazardous = np.count_nonzero(EWS99p989_7AM_6PM > thresholdDangerous, axis=a) / hours

    return sitting, walking, hazardous






