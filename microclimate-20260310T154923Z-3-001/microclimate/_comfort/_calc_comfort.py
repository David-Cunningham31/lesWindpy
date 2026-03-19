# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:29:20 2022

Module for functions related to evaluating comfort and usability

@author: Viet.Le
"""

import logging
import numpy as np
import pandas as pd
from .comfort import hourIndex, seasons, calculateUTCI 

logger = logging.getLogger(__name__)



#%%

def calculateUTCI_TFcorr(dbt, mrt, rh, vel, TF_WS=1.4):


    print(f"Multipying wind speeds by {TF_WS} to tranpose from 1.5m to 10m height")
    vel *= TF_WS

    UTCI_approx = calculateUTCI(dbt, mrt, rh, vel)

    return UTCI_approx

def clean_epw(df,time_filt=None,index=None):

    df_new = pd.DataFrame(index=df.index)
    # Set columns and datatype
    df_new['T_a'] = np.array(df['Dry Bulb Temperature'], dtype=np.float64)
    df_new['windSpeed'] = np.array(df['Wind Speed'], dtype=np.float64)
    df_new['windDir'] = np.array(df['Wind Direction'], dtype=np.float64)
    df_new['humidity'] = np.array(df['Relative Humidity'], dtype=np.float64)
    df_new['hir'] = np.array(df['Horizontal Infrared Radiation Intensity'], dtype=np.float64)
    df_new['dnr'] = np.array(df['Direct Normal Radiation'], dtype=np.float64)

    # Clean data
    df_new['T_a'][df_new['T_a']==99.9] = np.nan
    df_new['windSpeed'][df_new['windSpeed']==999] = np.nan
    df_new['windDir'][df_new['windDir']==999] = np.nan
    df_new['humidity'][df_new['humidity']==999] = np.nan
    df_new['hir'][df_new['hir']==999] = np.nan
    df_new['dnr'][df_new['dnr']==999] = np.nan
    
    df_new['irradiance'] = df_new['dnr'].copy(deep=True)
    df_new['alt_rad'] = df['alt_rad'].copy(deep=True)

    # filter the times
    if time_filt is not None:
        df_new = df_new[time_filt]
        
    # add the index
    if index is not None:
        df_new.index = index

    return df_new



def calc_MRT_EPW(T_a, hir):

    # calculate sky temperature from Horizontal Infrared Radiation Intensity
    sigma = 5.6697e-8 # Stefan-Boltzmann constant
    epsilon_person = 0.95
    
    T_sky = (hir/sigma/epsilon_person) ** 0.25 - 273.15
    T_grnd = np.roll(T_a,1) # thermal lag by 1 hour
    
    # MRT is the average between T_sky and T_grnd
    MRT = (T_sky + T_grnd)/2

    return MRT



def calc_clo(Ta, model='Havineth', limit_bounds=True):

    clo = 1.374 - 0.013847*Ta - 0.00043804*Ta**2 - 0.0000238383*Ta**3
    
    # ensure that clo stays within range used for UTCI
    if limit_bounds:
        clo[clo < 0.4] = 0.4
        clo[clo > 2.6] = 2.6

    return clo



def usability(utci, u_wind=None, cold_threshold=0, hot_threshold=32, wind_threshold=4):
    """
    Calculate percentage of annual comfort hours based on temperature and wind speed.

    Parameters
    ----------
    utci : ndarray
        List UTCI values with one row per point and one column per hour.
    u_wind : ndarray, optional
        List wind speeds with one row per point and one column per hour.
        The default is None, in which case wind speed is ignored.
    cold_threshold : number, optional
        Minimum UTCI temperature for comfort. The default is 0 Celcius.
    hot_threshold : number, optional
        Maximum UTCI temperature for comfort. The default is 32 Celcius.
    wind_threshold : number, optional
        Maximum wind speed for comfort. The default is 4 m/s.

    Returns
    -------
    usable : ndarray or number
        Usable fraction of time at each point.

    """
    a = utci.ndim - 1
    hours = utci.shape[a]
    if u_wind is None:
        usable = np.count_nonzero((utci >= cold_threshold) & (utci < hot_threshold), axis=a) / hours
    else:
        usable = np.count_nonzero((utci >= cold_threshold) & (utci < hot_threshold) & (u_wind < wind_threshold), axis=a) / hours

    logger.info('Usability (percentage of comfort hours) evaluated')

    return usable



def usabilityLondon(utci, date_range, cold_threshold=0, hot_threshold=32, start_hour=8, end_hour=20):
    """
    Calculate usage category according to the Thermal Comfort Guidelines for
    Developments in the City of London.
    https://www.cityoflondon.gov.uk/assets/Services-Environment/thermal-comfort-guidelines-for-developments-in-the-city-of-london.pdf

    Parameters
    ----------
    utci : ndarray
        List UTCI values with one row per point and one column per hour (for the full year).
    date_range : array of DatetimeIndex
        List of hours in the full year year.
    cold_threshold : number, optional
        Minimum UTCI temperature for comfort. The default is 0 Celcius.
    hot_threshold : number, optional
        Maximum UTCI temperature for comfort. The default is 32 Celcius.
    start_hour : number, optional
        First hour of day for comfort testing. The default is 8.
    end_hour : number, optional
        Last hour of day for comfort testinga. The default is 20.

    Returns
    -------
    category : ndarray or number
        Usage category at each point.

    """

    # Calculate fraction of time usable in each season
    spring = usability(utci[...,hourIndex(date_range, month=seasons[0], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    summer = usability(utci[...,hourIndex(date_range, month=seasons[1], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    autumn = usability(utci[...,hourIndex(date_range, month=seasons[2], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    winter = usability(utci[...,hourIndex(date_range, month=seasons[3], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)

    """
    for each location (along the rows), assign a usage category based on the 
    London criteria for percentage of usage across the season. 
    
    This will dictate the usability of each location. End result is a vector
    with entries representing usage category for each point in the VTK
    
    4 = all season
    3 = seasonal
    2 = short-term
    1 = short-term seasonal
    0 = transient
    
    """
    ct = np.zeros(utci.shape[0])
    for cc, (sp, su, au, wi) in enumerate(zip(spring,summer,autumn,winter)):
        if all([all(np.array([sp, su, au]) >= 0.9), (wi >= 0.9)]):
            ct[cc] = 4
        elif all([all(np.array([sp, su, au]) >= 0.9), (wi >= 0.7)]):
            ct[cc] = 3
        elif all([all(np.array([sp, su, au]) >= 0.5), (wi >= 0.5)]):
            ct[cc] = 2
        elif all([all(np.array([sp, su, au]) >= 0.5), (wi >= 0.25)]):
            ct[cc] = 1
        elif any([any(np.array([sp, su, au]) < 0.5), (wi < 0.25)]):
            ct[cc] = 0
    category = ct

    logger.info(f'Usability categorization evaluated based on London criteria')

    return category



def usabilityProvidence(utci, date_range, cold_threshold=0, hot_threshold=32, start_hour=8, end_hour=20):
    """
    Calculate usage category according to the Thermal Comfort Guidelines for
    Developments in the City of London.
    https://www.cityoflondon.gov.uk/assets/Services-Environment/thermal-comfort-guidelines-for-developments-in-the-city-of-london.pdf

    NOTE: this has been modified for Providence, Rhode Island

    Parameters
    ----------
    utci : ndarray
        List UTCI values with one row per point and one column per hour.
    date_range : array of DatetimeIndex
        List of hours in the year.
    cold_threshold : number, optional
        Minimum UTCI temperature for comfort. The default is 0 Celcius.
    hot_threshold : number, optional
        Maximum UTCI temperature for comfort. The default is 32 Celcius.
    start_hour : number, optional
        First hour of day for comfort testing. The default is 8.
    end_hour : number, optional
        Last hour of day for comfort testinga. The default is 20.

    Returns
    -------
    category : ndarray or number
        Usage category at each point.

    """
    # Calculate fraction of time usable in each season
    spring = usability(utci[...,hourIndex(date_range, month=seasons[0], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    summer = usability(utci[...,hourIndex(date_range, month=seasons[1], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    autumn = usability(utci[...,hourIndex(date_range, month=seasons[2], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)
    winter = usability(utci[...,hourIndex(date_range, month=seasons[3], startHour=start_hour, endHour=end_hour)], cold_threshold=cold_threshold, hot_threshold=hot_threshold)

    # Calculate winter score
    # bins = [ 0.25, 0.5, 0.7, 0.9 ]
    bins = [ 0.25, 0.35, 0.55, 0.8 ]
    categoryWinter = np.searchsorted(bins, winter, side='right')

    # Calculate score for spring, summer, and fall
    # bins = [ 0.5, 0.9 ]
    bins = [ 0.5, 0.8 ]
    categoryNonWinter = np.searchsorted(bins, np.minimum(spring, np.minimum(summer, autumn)), side='right') * 2

    category = np.minimum(categoryWinter, categoryNonWinter)

    ct = np.zeros(category.shape)
    for cc, (sp, su, au, wi) in enumerate(zip(spring,summer,autumn,winter)):
        if all([all(np.array([sp, su, au]) >= 0.8), (wi >= 0.8)]):
            ct[cc] = 4
        elif all([all(np.array([sp, su, au]) >= 0.8), (wi >= 0.55)]):
            ct[cc] = 3
        elif all([all(np.array([sp, su, au]) >= 0.5), (wi >= 0.35)]):
            ct[cc] = 2
        elif all([all(np.array([sp, su, au]) >= 0.5), (wi >= 0.25)]):
            ct[cc] = 1
        elif any([any(np.array([sp, su, au]) < 0.5), (wi < 0.25)]):
            ct[cc] = 0
    category = ct

    logger.info(f'Usability categorization evaluated based on customized Providence criteria')

    return category

def binning(data,bins):

    # ensure that you are evaluating the data along the correct axis (the time-axis should be the last axis)
    a = data.ndim - 1
    hours = data.shape[a]

    perc_time_in_bin = {}
    for bin_name, bin_range in bins.items():
        perc_time_in_bin[bin_name] = np.count_nonzero((data > bin_range[0]) & (data <= bin_range[1]), axis=a) / hours

    # check that all percentages add up to 100%
    check = sum(perc_time_in_bin.values())
    if abs(check - 1) > 1e-6:
        logger.error(f'Binning does not add up to 100%')
        raise ValueError("Why is the check not equal to 100%?")    

    return perc_time_in_bin

def binUTCI(utci):
    
    """
    Percentage of hours within thermal sensation thresholds based on UTCI

    Parameters
    ----------
    utci : ndarray
        List UTCI values with one row per point and one column per hour (for the full year).

    Returns
    -------
    hot, warm, comfy, cool, cold : float
        percentage of hours within the year that utci is either hot, warm, comfy, cool, or cold
    """

    ## UTCI Temperature thresholds
    thresholdHot = 32 # deg C
    thresholdWarm = 26 # deg C
    thresholdCool = 9 # deg C
    thresholdCold = 0 # deg C

    a = utci.ndim - 1
    hours = utci.shape[a]
    
    # temperature exceedances (exceedance in bands are output in the return)
    hot = np.count_nonzero(utci >= thresholdHot, axis=a) / hours
    warm = np.count_nonzero(utci >= thresholdWarm, axis=a) / hours
    cool = np.count_nonzero(utci < thresholdCool, axis=a) / hours
    cold = np.count_nonzero(utci < thresholdCold, axis=a) / hours
    
    hotwarm = np.count_nonzero(utci >= thresholdWarm, axis=a) / hours
    coolcold = np.count_nonzero(utci < thresholdCool, axis=a) / hours

    logger.info('UTCI has been binned based on customized thresholds')
    
    # NOTE: the middle output, which is comfy, refers to only neutral UTCI temp range
    # IT IS NOT WARM+NEUTRAL+COOL
    return hot, warm - hot, 1 - (warm + cool), cool - cold, cold, hotwarm, coolcold



def binUTCI_standard(utci):

    """
    Percentage of hours within thermal comfort condition thresholds based on UTCI

    Parameters
    ----------
    utci : ndarray
        List UTCI values with one row per point and one column per hour (for the full year).

    Returns
    -------
    ExtremeHeatStress, VeryStrongHeatStress, StrongHeatStress, ModerateHeatStress,
    NoThermalStress, SlightColdStress, ModerateColdStress, VeryStrongColdStress,
    ExtremeColdStress: float
        percentage of hours within the year that utci is either within the various
        thermal stress thresholds
    """
    
    # standard UTCI Temperature thresholds
    thresholdExtremeHeatStress = 46 # deg C
    thresholdVeryStrongHeatStress = 38 # deg C
    thresholdStrongHeatStress = 32 # deg C
    thresholdModerateHeatStress = 26 # deg C
    thresholdNoThermalStress = 9 # deg C
    thresholdSlightColdStress = 0 # deg C
    thresholdModerateColdStress = -13 # deg C
    thresholdStrongColdStress = -27 # deg C
    thresholdVeryStrongColdStress = -40 # deg C
    thresholdExtremeColdStress = -40 # deg C

    # ensure that you are comparing the thermal stress along the correct axis (the time-axis)
    a = utci.ndim - 1
    hours = utci.shape[a]
    
    """
    bin the UTCI against the various thermal stress bands
    """
    ExtremeHeatStress = np.count_nonzero(utci > thresholdExtremeHeatStress, axis=a) / hours
    VeryStrongHeatStress = np.count_nonzero((utci > thresholdVeryStrongHeatStress) & 
                                                     (utci <= thresholdExtremeHeatStress), axis=a) / hours
    StrongHeatStress = np.count_nonzero((utci > thresholdStrongHeatStress) & 
                                                     (utci <= thresholdVeryStrongHeatStress), axis=a) / hours
    ModerateHeatStress = np.count_nonzero((utci > thresholdModerateHeatStress) & 
                                                     (utci <= thresholdStrongHeatStress), axis=a) / hours
    NoThermalStress = np.count_nonzero((utci > thresholdNoThermalStress) & 
                                                     (utci <= thresholdModerateHeatStress), axis=a) / hours
    SlightColdStress = np.count_nonzero((utci > thresholdSlightColdStress) & 
                                                     (utci <= thresholdNoThermalStress), axis=a) / hours
    ModerateColdStress = np.count_nonzero((utci > thresholdModerateColdStress) & 
                                                     (utci <= thresholdSlightColdStress), axis=a) / hours
    StrongColdStress = np.count_nonzero((utci > thresholdStrongColdStress) & 
                                                     (utci <= thresholdModerateColdStress), axis=a) / hours
    VeryStrongColdStress = np.count_nonzero((utci > thresholdVeryStrongColdStress) & 
                                                     (utci <= thresholdStrongColdStress), axis=a) / hours
    ExtremeColdStress = np.count_nonzero(utci <= thresholdVeryStrongColdStress, axis=a) / hours
                                            
    # check: should add up to 100%
    check = (ExtremeHeatStress + VeryStrongHeatStress + StrongHeatStress + ModerateHeatStress + 
        NoThermalStress + SlightColdStress + ModerateColdStress + StrongColdStress +
            VeryStrongColdStress + ExtremeColdStress)
    if all(check) != 1:
        logger.error(f'Binning does not add up to 100%')
        raise ValueError("Why is the check not equal to 100%?")    
    return ExtremeHeatStress, VeryStrongHeatStress, StrongHeatStress, \
        ModerateHeatStress, NoThermalStress, SlightColdStress, \
            ModerateColdStress, StrongColdStress, VeryStrongColdStress, ExtremeColdStress

    logger.info('UTCI has been binned based on standardized thresholds')
    


def binSET(setstar):

    """
    Percentage of hours within thermal comfort condition thresholds based on SET*

    Parameters
    ----------
    setstar : ndarray
        List SET* values with one row per point and one column per hour (for the full year).

    Returns
    -------
    ExtremeHeatStress, VeryStrongHeatStress, StrongHeatStress, ModerateHeatStress,
    NoThermalStress, SlightColdStress, ModerateColdStress, VeryStrongColdStress,
    ExtremeColdStress: float
        percentage of hours within the year that utci is either within the various
        thermal stress thresholds
    """
    
    # standard SET Temperature thresholds
    thresholdVeryStrongHeatStress = 37.5 # deg C
    thresholdStrongHeatStress = 34.5 # deg C
    thresholdModerateHeatStress = 30 # deg C
    thresholdSlightHeatStress = 25.6 # deg C
    thresholdNoThermalStress = 22 # deg C
    thresholdSlightColdStress = 17.5 # deg C
    thresholdModerateColdStress = 14.5 # deg C

    # ensure that you are comparing the thermal stress along the correct axis (the time-axis)
    a = setstar.ndim - 1
    hours = setstar.shape[a]
    
    
    
    """
    bin the SET* against the various thermal stress bands
    """
    VeryStrongHeatStress = np.count_nonzero(setstar > thresholdVeryStrongHeatStress, axis=a) / hours
    StrongHeatStress = np.count_nonzero((setstar > thresholdStrongHeatStress) & 
                                                     (setstar <= thresholdVeryStrongHeatStress), axis=a) / hours
    ModerateHeatStress = np.count_nonzero((setstar > thresholdModerateHeatStress) & 
                                                     (setstar <= thresholdStrongHeatStress), axis=a) / hours
    SlightHeatStress = np.count_nonzero((setstar > thresholdSlightHeatStress) & 
                                                     (setstar <= thresholdModerateHeatStress), axis=a) / hours
    NoThermalStress = np.count_nonzero((setstar > thresholdNoThermalStress) & 
                                                     (setstar <= thresholdSlightHeatStress), axis=a) / hours
    SlightColdStress = np.count_nonzero((setstar > thresholdSlightColdStress) & 
                                                     (setstar <= thresholdNoThermalStress), axis=a) / hours
    ModerateColdStress = np.count_nonzero((setstar > thresholdModerateColdStress) & 
                                                     (setstar <= thresholdSlightColdStress), axis=a) / hours
    StrongColdStress = np.count_nonzero(setstar <= thresholdModerateColdStress, axis=a) / hours
                                            
    # check: should add up to 100%
    check = VeryStrongHeatStress + StrongHeatStress + SlightHeatStress + ModerateHeatStress + \
        NoThermalStress + SlightColdStress + ModerateColdStress + StrongColdStress
    if all(check) != 1:
        logger.error(f'Binning does not add up to 100%')
        raise ValueError("Why is the check not equal to 100%?")
        
    logger.info('SET has been binned based on standardized thresholds')

    return VeryStrongHeatStress, StrongHeatStress, ModerateHeatStress, SlightHeatStress, \
        NoThermalStress, SlightColdStress, ModerateColdStress, StrongColdStress

def binPET(pet):

    """
    Percentage of hours within thermal comfort condition thresholds based on PET

    Parameters
    ----------
    setstar : ndarray
        List SET* values with one row per point and one column per hour (for the full year).

    Returns
    -------
    ExtremeHeatStress, VeryStrongHeatStress, StrongHeatStress, ModerateHeatStress,
    NoThermalStress, SlightColdStress, ModerateColdStress, VeryStrongColdStress,
    ExtremeColdStress: float
        percentage of hours within the year that utci is either within the various
        thermal stress thresholds
    """
    
    # standard SET Temperature thresholds
    thresholdExtremeHeatStress = 41 # deg C
    thresholdStrongHeatStress = 35 # deg C
    thresholdModerateHeatStress = 29 # deg C
    thresholdSlightHeatStress = 23 # deg C
    thresholdNoThermalStress = 18 # deg C
    thresholdSlightColdStress = 13 # deg C
    thresholdModerateColdStress = 8 # deg C
    thresholdStrongColdStress = 4 # deg C

    # ensure that you are comparing the thermal stress along the correct axis (the time-axis)
    a = pet.ndim - 1
    hours = pet.shape[a]
    
    
    
    """
    bin the SET* against the various thermal stress bands
    """
    ExtremeHeatStress = np.count_nonzero(pet > thresholdExtremeHeatStress, axis=a) / hours
    StrongHeatStress = np.count_nonzero((pet > thresholdStrongHeatStress) & 
                                                     (pet <= thresholdExtremeHeatStress), axis=a) / hours
    ModerateHeatStress = np.count_nonzero((pet > thresholdModerateHeatStress) & 
                                                     (pet <= thresholdStrongHeatStress), axis=a) / hours
    SlightHeatStress = np.count_nonzero((pet > thresholdSlightHeatStress) & 
                                                     (pet <= thresholdModerateHeatStress), axis=a) / hours
    NoThermalStress = np.count_nonzero((pet > thresholdNoThermalStress) & 
                                                     (pet  <= thresholdSlightHeatStress), axis=a) / hours
    SlightColdStress = np.count_nonzero((pet > thresholdSlightColdStress) & 
                                                     (pet  <= thresholdNoThermalStress), axis=a) / hours
    ModerateColdStress = np.count_nonzero((pet > thresholdModerateColdStress) & 
                                                     (pet  <= thresholdSlightColdStress), axis=a) / hours
    StrongColdStress = np.count_nonzero((pet > thresholdStrongColdStress) & 
                                                     (pet  <= thresholdModerateColdStress), axis=a) / hours
    ExtremeColdStress = np.count_nonzero((pet <= thresholdStrongColdStress), axis=a) / hours

                                      
    # check: should add up to 100%
    check = ExtremeHeatStress + StrongHeatStress + SlightHeatStress + ModerateHeatStress + \
        NoThermalStress + SlightColdStress + ModerateColdStress + StrongColdStress

    if all(check) != 1:
        raise ValueError("Why is the check not equal to 100%?")
        
    return ExtremeHeatStress, StrongHeatStress, ModerateHeatStress, SlightHeatStress, \
        NoThermalStress, SlightColdStress, ModerateColdStress, StrongColdStress, ExtremeColdStress
