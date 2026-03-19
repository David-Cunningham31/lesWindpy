# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 11:07:21 2022

@author: Viet.Le
"""

import os
import logging
import numpy as np
import pandas as pd
from functools import partial, reduce
import pyvista as pyv
# from ._vtk import readPointVTK, readPointVTK_coord

# setup the logger
logger = logging.getLogger(__name__)



#%% utilities for loading files

def validFile(path):
    """
    Check if file exists with non-zero size

    Parameters
    ----------
    path :
        The file path

    Returns
    -------
    Flag to indicate that the path exists

    """
    return os.path.exists(path) and os.path.getsize(path) > 0



#%%

def get_data_paths(basepath,file_to_load,sub_directories=[],time=False,):
    """
    Get the paths to files with a specific name, given a base directory.
    Will search sub directories for this file

    Parameters
    ----------
    basepath : string
        base directory where the file to load or the folders containing it lie.
    file_to_load : string
        the file to load.
    sub_directories : TYPE, optional
        sub directories within basepath that may contain file to load.
        these are usually the wind directions
    time : string, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    OSError
        Could not find the data or folder at various points.

    Returns
    -------
    path_with_data : list
        list of paths with the wind speed data.

    """

    all_paths_with_file_to_load = list(basepath.rglob(f"**/{file_to_load}"))
    if not all_paths_with_file_to_load:
        logger.error(f"Could not find {file_to_load}")
        raise FileNotFoundError(f"Could not find {file_to_load} in {basepath}")
    
    if len(sub_directories) == 0:
        sub_directories = ["",]
        
    # filter out path_WSdata_all by direction
    path_with_data = []
    for dd, sub_directory in enumerate(sub_directories):
        
        # in the sub_directory for the provided folder path, extract any 
        # additional directories
        additional_sub_dir = []
        for fname in all_paths_with_file_to_load:
            if ("\\"+sub_directory in str(fname) or "/"+sub_directory in str(fname)) and \
                not ("_SS".lower() in str(fname).lower()) and \
                    fname.parts[-2] != 'constant':
                        additional_sub_dir.append(fname)
        if len(additional_sub_dir) == 0:
            logger.error(f"Could not find {sub_directory}")
            raise FileNotFoundError(f"Could not find {sub_directory} in {all_paths_with_file_to_load}")
            
        # remove additional sub directories that are not related to a timestep
        
                
        # Of these additional directories, extract the last one in the list. 
        # Usually, these additional directories are for time or iterations
        # If a specific time or iteration is specified, take that instead
        temp2 = []
        if time == "" or time == False:
            temp2 = additional_sub_dir[-1]
            if "ss" in str(temp2).lower():
                print(f"MAKE SURE that \n{temp2}\n which has been loaded in, is NOT an SS folder!")
        else:
            
            # if a time is provided for each sub_directory
            if hasattr(time, "__len__"):
                time_to_look_for = time[dd]
                
            # if only one time has been provided
            else:
                time_to_look_for = time
                
            # look through all the paths 
            for fname in additional_sub_dir:
                if time_to_look_for in str(fname):
                    temp2 = fname
        path_with_data.append(temp2)
                    
        if len(str(temp2)) == 0:
            logger.error(f"\n{sub_directory[:-1]} was not loaded. Please check from this list of paths: \n\n{*additional_sub_dir,}")
            raise FileNotFoundError(f"\n{sub_directory[:-1]} was not loaded. Please check from this list of paths: \n\n{*additional_sub_dir,}")

    # check that none of the paths are duplicated. if so, then you have an error
    newlist = []
    duplist = []
    for i in path_with_data:
        if i not in newlist:
            newlist.append(i)
        else:
            duplist.append(i)
    if len(duplist) != 0:
        raise Exception(f"You have duplicate paths: {duplist}")
            
    # check that the number of paths for WS matches the number of direction strings
    if len(path_with_data) != len(sub_directories):
        logger.error("Please check your folder structures! One or some of the directions were not loaded!")
        raise OSError("\nPlease check your folder structures! \n \
            One or some of the directions were not loaded!")
    
    logger.info("Got all selected files to load in {basepath}")
    
    return path_with_data



def loadOFdata(path_list,variable='U',
               mode='plane',
               pointmode='resampled',iteration=1,keeptime=False):
    """
    Loads in the variable from OF

    Parameters
    ----------
    pathWS : list
        list of paths where we need to load in the variable.
    variable : string, optional
        the variable to load in.
    mode : string, optional
        specifies what kind of data we are working with (point or plane).
    pointmode : string, optional
        if working with point data, then either one of two methods or points:
            resampled or runtime
    iteration : int, optional
        for point data, how many iterations back. The default is 1.
    keeptime : bool, optional
        flag to keep the iteration history of the CFD data.

    Raises
    ------
    OSError
        Could not load in the file based on the provided paths.

    Returns
    -------
    data : ndarray
        array of data loaded in from provided paths.
    coordinates : ndarray
        x, y, z coordinates for each point of the data.

    """
    
    
    
    coordinates = {}
    data = {}
    for pp, path in enumerate(path_list):
        if validFile(path):

            """
            Different modes for data types
            """
            if mode == 'plane':

                # grab the OF data. If vector or U, then calculate magnitude
                # otherwise, just load it in
                # data_temp = readPointVTK(path, variable) ### no longer needed
                pyv_reader = pyv.get_reader(path)
                data_temp = pyv_reader.read()[variable]
                
                # check that the loaded data is a vector or not
                flag_vector = False
                if data_temp.ndim > 1: 
                    flag_vector = True
                    
                # find the magnitude if the data is a vector
                if flag_vector:
                    data_temp = np.linalg.norm(data_temp, axis=1)
                
                # form the matrix that will hold all the data. grows with path
                data[path] = pd.DataFrame(data_temp, columns=[variable,])
                
                # if pp == 0:
                    # data = data_temp.reshape(-1,1)
                # else:
                    # data = np.concatenate((data, data_temp.reshape(-1,1)), 1)
                
                # load in the coordinates
                # coordinates = readPointVTK_coord(path)
                coordinates[path] = pd.DataFrame(pyv_reader.read().points, 
                                                 columns=['x','y','z'])
                
            elif mode == 'points':

                # get mean wind speed from probe data and convert to numpy array
                if pointmode == 'resampled':
                    data[path], coordinates[path] = ofProbeReSamp(path)
                    
                elif pointmode == 'runtime':
                    
                    if variable in ['U','UMean']:
                        data_temp_df = ofProbeVector(path)
                    else:
                        data_temp_df = ofProbeScalar(path)
                    if data_temp_df.empty:
                        logger.error(f"Dataframe is empty. Please check file path for: {path}")
                        raise OSError(f"Dataframe is empty. Please check file path for: {path}")
                    # data_temp = data_temp_df.to_numpy()
                    
                    # keep the previous iterations or average/take the last one
                    if keeptime:
                        
                        # this keeps all iterations
                        if isinstance(iteration,str) and iteration.lower() == 'all':
                            iteration = data_temp_df.shape[0]
                        data[path] = data_temp_df.iloc[-iteration::]
                        
                    else:
                        
                        # take the average of the last several iterations
                        data[path] = pd.DataFrame(data_temp_df.iloc[-iteration::].mean(axis=0))
                                
                    # load in the coordinates
                    coordinates[path] = ofProbeCoords(path)
                    
                # rename data index and name, if runtime iterations not kept
                if not keeptime:
                    data[path].index.name = 'Probe_ID'
                    data[path] = data[path].rename(columns={0: variable})
                    
                    data[path].index = data[path].index.astype(str)
                    coordinates[path].index = coordinates[path].index.astype(str)
                else:
                    data[path].index.name = 'Iteration'
                    
            # combine coordinates with data dataframe
            if not keeptime:
                data[path] = pd.concat([data[path], coordinates[path]],axis=1)
                    
            logger.info(f"Loaded {variable} data in {path} successfully")
            
        else:
            logger.error(f"Could not load {variable} data in {path}")
            raise OSError(f"Could not find file path: {path}")


    """
    check that all the x,y,z coordinates are all the same.
    If so, then collapse the coordinates dictionary so we return less data
    """
    first_coord_values = list(coordinates.values())[0]
    all_equal = all(first_coord_values.equals(value) for value in coordinates.values())
    
    if not keeptime:
        if all_equal:
            coordinates = first_coord_values
            logger.info("Coordinates all the same. Output only the first set")
            for key, df in data.items():
                temp = df[variable].to_numpy()
                data[key] = temp.ravel()
        else:
            logger.info("Coordinates not all the same. Output only of merged coordinates!")
            print("Coordinates not all the same. Output only of merged coordinates!")
            my_reduce = partial(pd.merge,how='inner',
                                left_on=['x','y','z'],
                                right_on=['x','y','z'])
            coordinate_merge = reduce(my_reduce, coordinates.values())
            avg_dflen = np.mean([len(df) for df in coordinates.values()])
            print(f"Coordinates reduced from avg length of {avg_dflen} to {len(coordinate_merge)}")
            if len(coordinate_merge) / avg_dflen < 0.5:
                raise Exception("You have reduced the length of the data via merging like-coordinates by a significant amount\nPlease check your VTKs!")
            
            coordinates = coordinate_merge
            for key, df in data.items():
                temp = df.loc[(df['x'].isin(coordinates['x'])) \
                              & (df['y'].isin(coordinates['y'])) \
                                  & (df['z'].isin(coordinates['z']))][variable]
                temp = temp.to_numpy()
                data[key] = temp.ravel()
    
        # convert dictionary to numpy array
        data = np.array(list(data.values())).T
    else:
        coordinates = first_coord_values

    return data, coordinates



def ofProbeReSamp(filename,flag_vector=False):
    """
    Extracts the magnitude and the coordinates of the data resampled from OF

    Parameters
    ----------
    filename : string
        path to the file you want to load in.
    flag_vector : boolean, optional
        indication of whether the data is a vector or not. The default is False.

    Returns
    -------
    OFdata : array
        the data loaded from OF
    df : dataframe
        coordinates for each point of the data

    """    
    
    data = np.genfromtxt(filename,
                         skip_header = 0)
    
    ndim = data.shape[1]

    # extract the data    
    if ndim == 6:
        OFdata = np.sqrt(data[:,3] ** 2 + data[:,4] ** 2 + data[:,5] ** 2)
    else:
        OFdata = data[:,3]

    # extract the coordinates
    df = pd.DataFrame(data = data[:,0:3], columns = ['x','y','z'])
    df.index.name = 'Probe_ID'

    return pd.DataFrame(OFdata).apply(pd.to_numeric), df



def ofProbeScalar(filename):
    """
    Load in the scalar data from OF runtime sampling

    Parameters
    ----------
    filename : string
        path to file to load in data.

    Returns
    -------
    dataframe
        dataframe with loaded data.

    """
    f = open(filename,'r')
    probeVal=[]
    for line in f:
        if(line[:1] != '#'):
            data = line.split()
            data[-1] = data[-1].replace('\n','')
            probeVal.append(data[1:])
    return pd.DataFrame(probeVal).apply(pd.to_numeric)



def ofProbeCoords(filename):
    """
    Load in the probe coordinates from OF runtime sampling

    Parameters
    ----------
    filename : string
        path to file to load in data.

    Returns
    -------
    dataframe
        dataframe with loaded coordinates.

    """
    f = open(filename,'r')
    probeList=[]
    for line in f:
        if(line[:8] == '# Probe '):
            probes = line.replace('('," ").replace(')',"").replace('\n','').split()
            probeList.append(probes[0:6]) 
            
    df = pd.DataFrame(probeList, columns =['#', 'Probe',"Probe_ID","x","y","z"])
    return df.set_index('Probe_ID').drop(['#',"Probe"],axis=1).apply(pd.to_numeric)



def ofProbeVector(filename):
    """
    Load in the magnitude of vector data from OF runtime sampling

    Parameters
    ----------
    filename : string
        path to file to load in data.

    Returns
    -------
    dataframe
        dataframe with loaded coordinates.

    """
    f = open(filename,'r')
    allData=[]
    for line in f:
        vectorList = []
        
        """
        This happens for each line. 
        Separate the Iteration Number with the velocity coords for each probe
        """
        if(line[:1] != '#'):
            data = line.split('  ')
            data = list(filter(None, data))
            data[0] = data[0].strip()
            data[-1] = data[-1].replace('\n','')
            for i in range(len(data)):
                cut = data[i].find('(',0,3)
                cut=cut+1
                if(i==0):
                    vectorList.append(data[i])
                else:
                    vectorData = data[i][cut:-1].split(' ')
                    try:
                        vectorMag = np.sqrt(float(vectorData[0])**2 + float(vectorData[1])**2 + float(vectorData[2])**2)
                        vectorList.append(vectorMag)
                    except(OverflowError):
                        vectorList.append('')
                        continue
            allData.append(vectorList[1:])
    return pd.DataFrame(allData).apply(pd.to_numeric)





