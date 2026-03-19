# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:45:09 2022

Module with functions to interact with VTKs

@author: Viet.Le
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)



#%%

def appendData(path, data, name, cell=True, first=False, outformat='%.8E'):
    """
    Appends data to a VTK.
    Works with either scalar or vector

    Parameters
    ----------
    path : string
        path to the VTK.
    data : ndarray
        the data to append (each row should be a point, each column is an x,y,z component).
    name : string
        name of the data to append.
    cell : bool, optional
        flag for cell data or not. The default is True.
    first : bool, optional
        flag for first entry to VTK. The default is False.
    outformat : string, optional
        format for output of VTK data. The default is '%.8E'.

    Returns
    -------
    None.

    """
    with open(path, 'a', newline='') as writer:
        if first:
            writer.write('%s %d\r\n' % ('CELL_DATA' if cell else 'POINT_DATA', data.size))
            
        # for scalar quantities
        if len(data.shape) == 1:
            writer.write(f"SCALARS {name.replace(' ', '_')} float LOOKUP_TABLE {data.size:d}\r\n")
            data.tofile(writer, sep='\r\n', format=outformat)
            
        # for vector quantities
        else:
            writer.write("FIELD attributes 1 \n")
            writer.write(f"{name.replace(' ', '_')} {data.shape[1]} {data.shape[0]} float \n")
            # data.tofile(writer, sep='\t\r\n', format=outformat)
            np.savetxt(writer, data, delimiter=' ', fmt=outformat)
        writer.write('\r\n')
    
    return
        

        
def clearVTK(path):
    """
    clears all data from a VTK except for coordinate information

    Parameters
    ----------
    path : string
        path to VTK file that is to be cleared.

    Returns
    -------
    None.

    TODO: fix POINT_DATA search (VL -> remember Dubai project)
    """
    path = str(path)
    with open(path, "r") as rf, open(path.replace('.vtk','-copy.vtk'), "w") as wf:
        for line in rf:
            wf.write(line)
            if 'POINT_DATA' in line:
                break
    os.remove(path)
    os.rename(path.replace('.vtk','-copy.vtk'),path)
    logger.info(f"{path} has been cleared of all data except for coordinates")
    
    return



def writetoVTK(name,coords_data,data,varname,mode='w',outformat='%.8E'):
    """
    writes data to a VTK. Can write from new or append data

    Parameters
    ----------
    name : string
        path to VTK file to write or append to.
    coords_data : dataframe
        x,y,z coordinates for each point.
    data : ndarray
        data to append to the VTK.
    varname : string
        name of the variable in the VTK.
    mode : string, optional
        Either 'a' for append or 'w' for write. The default is 'w'.
    outformat : string, optional
        number of decimal places to write data out to.

    Returns
    -------
    None.

    """
    
    points=str(len(data))

    # if writing data to VTK from scratch
    if mode == 'w':
        coords_data_str = coords_data['x'].astype(str) + ' ' + \
            coords_data['y'].astype(str) + ' ' + \
                coords_data['z'].astype(str)
        coords_data_str = "\n".join(coords_data_str.values)

        # check that amount coordinates and data inputs are the same
        if len(coords_data) != len(data):
            logger.error("The number of coordinates does not match the number of data points!")
            raise Exception("The number of coordinates does not match the number of data points!")

        with open(name, "wt") as writer:
            writer.write('# vtk DataFile Version 2.0 \ndata file - '\
                          + name + '\nASCII \nDATASET POLYDATA \nPOINTS   ' \
                              + points + '    FLOAT \n')
            writer.write(coords_data_str)

            if len(data.shape) == 1:
                writer.write('\nPOINT_DATA '+ points \
                              +f' \nSCALARS {varname} float LOOKUP_TABLE {points} \n')
            else:
                writer.write('\nPOINT_DATA '+ points \
                              +f' \nFIELD attributes 1\n' \
                              +f'{varname} 3 {points} float')
            np.savetxt(writer, data, delimiter=' ', fmt=outformat)

    # if appending data to an existing VTK
    elif mode == 'a':
        with open(name,"a+") as writer:

            # for scalar quantities
            if len(data.shape) == 1:
                writer.write('\nSCALARS ' + varname + \
                              f' float LOOKUP_TABLE {points} \n')
                np.savetxt(writer, data, delimiter=' ', fmt=outformat)

            # for vector quantities
            else:
                writer.write("FIELD attributes 1 \n")
                writer.write(f"{varname.replace(' ', '_')} {data.shape[1]} {data.shape[0]} float \n")
                # data.tofile(writer, sep='\t\r\n', format=outformat)
                np.savetxt(writer, data, delimiter=' ', fmt=outformat)

    logger.info(f'{varname} has been written to {name}')
    return




#%% _SS

def readPointVTK_coord(path):
    """
    Reads the coordinates of the VTK data

    Parameters
    ----------
    path : string
        path to the VTK file.

    Returns
    -------
    x, y, z : ndarray
        three separate arrays of x, y, z coordinates for each point of the VTK

    """
    with open(path, 'r') as reader:
        # Read the number of points
        line = reader.readline()
        
        # determine number of points in the VTK
        while line:
            if not line.startswith('POINTS'):
                line = reader.readline()
                continue
            elif line.startswith('POINTS'):
                num_pts = int(line.split(' ')[1])
                count = 1
                
                """
                start extracting the x, y, z coordinates 
                until the number of extractions = number of pts in the VTK
                """
                x = np.zeros(num_pts,)
                y = np.zeros(num_pts,)
                z = np.zeros(num_pts,)
                while count < num_pts:
                    line_xyz = reader.readline()
                    no_of_coords=int(len(line_xyz.split())/3)
                    for coord_no in range(no_of_coords):
                        x[count-1] = float(line_xyz.split()[0+(coord_no*3)])
                        y[count-1] = float(line_xyz.split()[1+(coord_no*3)])
                        z[count-1] = float(line_xyz.split()[2+(coord_no*3)])
                        count += 1
                break
    logger.info('Coordinates of VTK loaded')
    return x, y, z



def readPointVTK(path, field):
    """
    Read in point data

    Parameters
    ----------
    path : STR
        path to the VTK file.
    field : STR
        the variable to read.

    Returns
    -------
    values : ARRAY OF FLOAT
        Value of the field for each sampling point in the VTK.

    """
    ndims = 0
    npoints = 0
    with open(path, 'r') as reader:
        # Read the number of points
        line = reader.readline()
        
        """
        Grab the dimensions of the variable (i.e. scalar or not)
        
        And determine how many points
        """
        while line:
            if (not line.startswith(field)) and (not line.startswith(f'SCALARS {field}')):
                line = reader.readline()
                continue
            
            if line.startswith(field):
                ndims = int(line.split()[1])
                npoints = int(line.split()[2])
            elif line.startswith(f'SCALARS {field}'):
                ndims = 1
                npoints = int(line.split()[-1])
                
            break



        """
        If, for some reason, no points are found with this variable, 
        return an exception and alert user
        """
        if npoints == 0:
            logger.error(f'Variable {field} was not found. Please check {path}')
            raise Exception(f'Variable {field} was not found. Please check {path}')
            

        
        """
        Read each point until a non-float is return 
        (which indicates change of variables)
        """
        values = np.fromfile(reader, dtype=float, sep=' ').reshape(npoints, ndims)
        logger.info(f'Variable {field} was loaded')
        
    return values   
