# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:45:09 2022

Module with useful functions and tools

@author: Viet.Le
"""

import logging
import calendar
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# the jupyter notebook style for tables
my_style = """<style>
.dataframe     
    table {
      border: none;
      border-collapse: collapse;
      border-spacing: 0;
      color: black;
      font-size: 12px;
      table-layout: fixed;
    }
    thead {
      border-bottom: 1px solid black;
      vertical-align: bottom;
    }
    tr, th, td {
      text-align: right;
      vertical-align: middle;
      padding: 0.5em 0.5em;
      line-height: normal;
      white-space: normal;
      max-width: none;
      border: none;
    }
    th {
      font-weight: bold;
    }
    tbody tr:nth-child(odd) {
      background: #f5f5f5;
    }
    tbody tr:hover {
      background: rgba(66, 165, 245, 0.2);
    }
</style>\n"""



#%% Tools

# function to round values in an array to closest value in another array
# https://stackoverflow.com/questions/33450235/rounding-a-list-of-values-to-the-nearest-value-from-another-list-in-python
def rounder(values):
    '''
    The rounder function takes the values which we want to round to. 
    It creates a function which takes a scalar and returns the closest element 
    from the values array. We then transform this function to a broadcast-able 
    function using numpy.frompyfunc. This way you are not limited to using this 
    on 2d arrays, numpy automatically does broadcasting for you without any 
    loops.
    
    i.e. performs the function f(x) for each element in the numpy array

    Parameters
    ----------
    values : array
        list of reference values to round to.

    Returns
    -------
    array
        closest value to round to.

    '''
    def f(x):
        idx = np.argmin(np.abs(values - x))
        return values[idx]
    return np.frompyfunc(f, 1, 1)



def gen_inputs_html(name,**kwargs):
    """
    writes inputs to an HTML for easy viewing and QA check

    Parameters
    ----------
    name : str
        name of the HTML file.
    **kwargs : dict
        dictionary of arguments to write to HTML file.

    Returns
    -------
    None.

    """

    html = my_style
    list_of_var_sametable = []
    list_of_key_sametable = []
    logger_trigger = True
    for key, value in kwargs.items():
    
        """
        Generate html contents 
        """
        # if the value is some kind of variable with more than one entry
        if np.ndim(value) == 0:
            if key in ['refheight',]:
                extra_str = 'm'
            else:
                extra_str = ''
                
            html = f"<html> <head> </head> <h2> {key} = {value}{extra_str} </h2> <body> </body> </html>\n" + html
                

        # if the value is some kind of variable with multiple entries
        else:
            
            try:
                # if the variable is the same length as the number of directions,
                # collect everything in a table. 
                # Don't generate the html table just yet!
                if len(value) == len(kwargs['directions']):
                    list_of_var_sametable.append(value)
                    list_of_key_sametable.append(key)
                else:
                    
                    # generate table for each multi-entry variable
                    df = pd.DataFrame([value,], index=[key,])
                    html += df.to_html()
                    html += "<br>\n"
                    
            except KeyError:
                if logger_trigger == True:
                    logger.warn("No variable with wind direction written to HTML file")
                    logger_trigger = False
                    
            
            
    """
    Generate tables for inputs provided for each wind direction
    """
    try:
        df = pd.DataFrame(list_of_var_sametable, 
                          columns=kwargs['directions'], 
                          index=list_of_key_sametable)
        html += df.to_html()
        html += "<br>\n"
    except KeyError:
        None
        
        
        
    """
    write the HTML contents
    """
    with open(f'{name}', 'w') as html_content:
        html_content.write(html)
        
        
        
    logger.info("Inputs written out to HTML")
    
    return



#%%

def hourIndex(dates, month=None, startHour=0, endHour=24, flag_Ireland=False):
    """
    Get the indices of hours to include
    dates - 
    month - 
    startHour - 
    endHour - 

    Parameters
    ----------
    dates : list
        list of datetime objects, typically a whole year.
    month : integer, optional
        the month [1-12] or zero for all months.
    startHour : integer, optional
        first hour of day in the data (0 if data contains 24-hour day). The default is 0.
    endHour : integer, optional
        last hour of day in the data (24 if data contains 24-hour day). The default is 24.
    flag_Ireland : bool, optional
        flag for Irish seasons. The default is False.

    Raises
    ------
    Exception
        an incorrect month was provided.

    Returns
    -------
    array of booleans
        array that indicates which entries of the data fall through the filter.

    """

    seasons = [
            "Spring",
            "Summer",
            "Autumn",
            "Winter",
            "Shoulder"
            ]
    
    if startHour == endHour:
        occupied = (dates.hour == startHour)
    elif startHour < endHour:
        occupied = (dates.hour >= startHour) & (dates.hour < endHour)
    else:
        occupied = (dates.hour >= startHour) | (dates.hour < endHour)

    if month is None or month == 0:
        return occupied

    if month in range(1, len(calendar.month_name)):
        return (dates.month == month) & occupied

    if month == 13 or month == seasons[0]: # Spring
        if flag_Ireland:
            return (dates.month >= 2) & (dates.month < 5) & occupied
        else:
            return (dates.month >= 3) & (dates.month < 6) & occupied

    if month == 14 or month == seasons[1]: # Summer
        if flag_Ireland:
            return (dates.month >= 5) & (dates.month < 8) & occupied
        else:
            return (dates.month >= 6) & (dates.month < 9) & occupied

    if month == 15 or month == seasons[2]: # Autumn
        if flag_Ireland:
            return (dates.month >= 8) & (dates.month < 11) & occupied
        else:
            return (dates.month >= 9) & (dates.month < 12) & occupied

    if month == 16 or month == seasons[3]: # Winter
        if flag_Ireland:
            return ((dates.month == 11) | (dates.month == 12) | (dates.month == 1)) & occupied
        else:    
            return ((dates.month >= 12) | (dates.month < 3)) & occupied

    if month == 17 or month == seasons[4]: # Shoulder
        if flag_Ireland:
            return (((dates.month >= 2) & (dates.month < 5)) | ((dates.month >= 8) & (dates.month < 11))) & occupied
        else:    
            return (((dates.month >= 3) & (dates.month < 6)) | ((dates.month >= 9) & (dates.month < 12))) & occupied

    raise Exception('Unknown month %s' % str(month))