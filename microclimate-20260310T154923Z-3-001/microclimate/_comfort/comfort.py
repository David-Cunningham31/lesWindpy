# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:39:46 2020

@author: Nathaniel.Jones
"""

import pandas as pd
import numpy as np
import sys
import os
import glob
import calendar
import datetime as dt
import argparse
#from shutil import copyfile

## Constants
C2K = 273.15
stefan_boltzmann = 5.67e-8
Wm2perMet = 58.15 # Watt/m^2 per MET from ASHRAE

"""
Arguments
"""
def getSettings(root='.', schemes=[], grids=[], directions=[], points='points.csv', grid_height=None, location='Climate', timezone='US/Eastern', occupancy=[8, 18], hours=[9, 12, 15], skydiv=4, uref=10, nprocs=1, binary=False):
    """
    Get settings specified in the command line arguments.

    Parameters
    ----------
    root : string, optional
        Root directory if no argument is given. The default is '.'.
    schemes : array of string, optional
        List of schemes if no arguments are given. The default is [].
    grids : array of string, optional
        List of grids if no arguments are given. The default is [], indicating
        all grids in the grid directory will be analyzed.
    directions : array of number, optional
        List of directions if now arguments are given. The default is [].
    points : string, optional
        Name of points database file if no argument is given. The default is
        'points.csv'.
    grid_height: number, optional
        Elevation of the grid plane from the ground if no arguments are given.
        The default is None.
    location : string, optional
        Name of climate used for sky file if no argument is given. The default
        is 'Climate'.
    timezone : string, optional
        Name of timezone for calculating daylight saving time to use if no
        argument is given. The default is 'US/Eastern'.
    occupancy : array of int, optional
        List of at least two values indicating the start and end hours of occupancy
        if no arguments are given. The default is [8, 18].
    hours : array of int, optional
        List of hours for point-in-time output if no arguments are given.
        The default is [9, 12, 15].
    skydiv : int, optional
        Level of Reinhart sky subdivision for use if no argument is given.
        The default is 4.
    uref : array_like, optional
        Reference wind speeds for each wind direction at 10 meter elevation
        if no arguments are given.
        The default is 10.
    nprocs : int, optional
        Number of processors to use for multiprocessing if no argument is given.
        The default is 1.
    binary : boolean, optional
        Flag to save binary VTK files as output. The default is False.

    Returns
    -------
    Namespace
        Attributes parsed from the command line arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', help='root directory for all files', required=False, type=str, default=root)
    parser.add_argument('--schemes', nargs='+', help='list of radiance schemes', required=False, type=str, default=schemes)
    parser.add_argument('--grids', nargs='+', help='list of radiance grids', required=False, type=str, default=grids)
    parser.add_argument('--directions', nargs='+', help='list of radiance view directions', required=False, type=int, default=directions)
    parser.add_argument('--points', help='path to points database file', required=False, type=str, default=points)
    parser.add_argument("--grid_height", help="elevation of the grid plane", type=float, required=False, default=grid_height)
    parser.add_argument("--weather", help="path to EPW file", type=str, required=False)
    parser.add_argument("--location", help="name of location", type=str, required=False, default=location)
    parser.add_argument("--timezone", help="name of timezone", type=str, required=False, default=timezone)
    parser.add_argument("--occupancy", nargs='+', help="start and end hour of occupancy", type=int, required=False, default=occupancy)
    parser.add_argument("--hours", nargs='+', help="hours for point-in-time output", type=int, required=False, default=hours)
    parser.add_argument("--skydiv", help="Reinhart sky division", type=int, required=False, default=skydiv)
    parser.add_argument('--uref', nargs='+', help='list of reference wind speeds', required=False, type=float, default=uref)
    parser.add_argument("--nprocs", help="number of parallel processes", type=int, required=False, default=nprocs)
    parser.add_argument("--binary", help="create binary output", action='store_false' if binary else 'store_true', required=False)

    return parser.parse_args()

"""
File Handling Routines
"""

## Check if file exists with non-zero size
def validFile(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

## Make a new directory or nested directory
def makeDir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

## Get file modified time
def dateModified(path):
    return os.path.getmtime(path)

def matchFiles(pattern, as_ints=False):
    """
    Get files that match the given pattern

    Parameters
    ----------
    pattern : string
        File pattern to match, with an asterisk (*) to find all matching.
    as_ints : boolean, optional
        Flag to read the wildcard in the pattern as integers. The default is False.

    Returns
    -------
    string or int
        The parts of the file paths that match the wildcard.
    string
        The full file paths.

    """
    pat = pattern.replace('/', os.sep).replace('\\', os.sep) # Use correct path separator
    parts = pat.split('*')
    paths = glob.glob(pat)
    if not paths:
        return [], []
    values = [path.replace(parts[0], '').replace(parts[-1], '') for path in paths]
    if as_ints:
        values = [int(numeric_string) for numeric_string in values]
    return (list(t) for t in zip(*sorted(zip(values, paths)))) # Sort both lists by order of values

def hasNumbers(inputString):
    """
    Test if a string contains any numeric characters

    Parameters
    ----------
    inputString : string
        Text to test for numeric characters.

    Returns
    -------
    boolean
        True if any numners were found.

    """
    return any(char.isdigit() for char in inputString)

def hasHeader(path):
    """
    Test if the file contains a header

    Parameters
    ----------
    path : string
        The file to check for a header.

    Returns
    -------
    boolean
        True if the first line of the file does not contain numbers.

    """
    with open(path) as f:
        first_line = f.readline()
        return not hasNumbers(first_line)

"""
Conversion Routines
"""

def f2c(f):
    return (f - 32) * 5/9

def c2f(c):
    return (c* 9/5) + 32

"""
Geometry Routines
"""

# Area of triangle
def area(a, b, c):
    return np.linalg.norm(np.cross(b - a, c - a)) / 2

# Return the barycenter (centroid) of a list of points
def barycenter(points):
    return np.mean(points, axis=0)

# Angle between two normalized vectors
def angle(a, b):
    axa = np.ndim(a) - 1
    axb = np.ndim(b) - 1
    if axa == 0 and axb == 0:
        return np.arccos(np.dot(a, b))
    return np.arccos(np.tensordot(a, b, axes=(axa, axb)))

# Tetrahedral solid angle
# Source: https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
def solidAngle(p1, p2, p3, o):
    # Vectors to each point from the origin
    a = p1 - o
    b = p2 - o
    c = p3 - o
    # Magnitude (length) of vectors
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    nc = np.linalg.norm(c, axis=1)

    tripleProduct = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = na * nb * nc + np.einsum("ij,ij->i", a, b) * nc + np.einsum("ij,ij->i", a, c) * nb + np.einsum("ij,ij->i", b, c) * na

    sigma = 2 * np.arctan(np.abs(tripleProduct) / denominator)
    sigma[sigma < 0] += np.pi

    return sigma

## Find the closest point with data to each queried point
#  df: dataframe of selected points for query, to be modified
#  points: points with known data, one point per row
#  column: name of column in dataframe for storing index of closest point
def findClosestPoints(df, points, column):
    for i in df.index:
        df.at[i, column] = np.argmin(np.linalg.norm(points - [ df.at[i, 'X'], df.at[i, 'Y'], df.at[i, 'Z'] ], axis=1))

def angleMidpoint(angles, radians=False):
    """
    Get the midpoints between successive pairs of angles

    Parameters
    ----------
    angles : array_like
        A list of angles.
    radians : boolean, optional
        Flag indicating whether the angles are in radians (True) or degrees
        (False). The default is False.

    Returns
    -------
    TYPE
        A list of the bisecting angles of each pair.

    """
    if np.size(angles) < 2:
        return angles
    modulo = 2 * np.pi if radians else 360
    return ((np.array(angles) + np.append(angles[1:], angles[0] + modulo)) / 2) % modulo

"""
Time Routines
"""

seasons = [
        "Spring",
        "Summer",
        "Autumn",
        "Winter",
        "Shoulder"
        ]

## Get the indices of hours to include
#  dates - list of datetime objects, typically a whole year
#  month - the month [1-12] or zero for all months
#  startHour - first hour of day in the data (0 if data contains 24-hour day)
#  endHour - last hour of day in the data (24 if data contains 24-hour day)
def hourIndex(dates, month=None, startHour=0, endHour=24):
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
        return (dates.month >= 3) & (dates.month < 6) & occupied
    if month == 14 or month == seasons[1]: # Summer
        return (dates.month >= 6) & (dates.month < 9) & occupied
    if month == 15 or month == seasons[2]: # Autumn
        return (dates.month >= 9) & (dates.month < 12) & occupied
    if month == 16 or month == seasons[3]: # Winter
        return ((dates.month >= 12) | (dates.month < 3)) & occupied
    if month == 17 or month == seasons[4]: # Shoulder
        return (((dates.month >= 3) & (dates.month < 6)) | ((dates.month >= 9) & (dates.month < 12))) & occupied
    raise Exception('Unknown month %s' % str(month))

## Get the indices of hours on solstice and equinox days
#  dates - list of datetime objects, typically a whole year
#  season - the name or index of the season starting with the solstice or equinox day
#  hour - hour on the day to query
#  days - number of days before or after the solstice or equinox to include
def hourIndexSolstice(dates, season, hour, days=0):
    if season == 4 or season == seasons[4]: # Shoulder
        return hourIndexSolstice(dates, 0, hour, days) | hourIndexSolstice(dates, 2, hour, days)
    if season == 0 or season == seasons[0]: # Spring
        doy = 80
    elif season == 1 or season == seasons[1]: # Summer
        doy = 172
    elif season == 2 or season == seasons[2]: # Autumn
        doy = 264
    elif season == 3 or season == seasons[3]: # Winter
        doy = 355
    if doy + days > 365:
        return ((dates.dayofyear >= (doy - days)) | (dates.dayofyear <= ((doy + days) % 365))) & (dates.hour == hour)
    return (dates.dayofyear >= (doy - days)) & (dates.dayofyear <= (doy + days)) & (dates.hour == hour)

## Get a string to represent a time on the solstice or equinox
#  season - the name or index of the season starting with the solstice or equinox day
#  hour - hour on the day to represent in the string
def solsticeString(season, hour):
    if season == 4 or season == seasons[4]: # Shoulder
        date = dt.datetime(2019, 9, 21, hour)
        return "17 Equinox %s" % date.strftime("%I%p")
    if season == 0 or season == seasons[0]: # Spring
        month = 3
    elif season == 1 or season == seasons[1]: # Summer
        month = 6
    elif season == 2 or season == seasons[2]: # Autumn
        month = 9
    elif season == 3 or season == seasons[3]: # Winter
        month = 12
    date = dt.datetime(2019, month, 21, hour)
    return date.strftime("%m %B %d %I%p")

def daysUntilMonth(month):
    """Return cumulative days until start of month.
    
    Months are indexed from 0 to 12, so month 0 returns 0, month 11
    returns 334, and month 12 (not a month) returns 365.

    In simple terms, if you're looking at times in January, use 0.
    """

    cumulative = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)

    return cumulative[month]


"""
Solar Position Routines
"""

## Calculate solar altitude for each time step, in degrees
'''
def solarAltitudes(epw, latitude, longitude, timezone):
    timesteps = epw.shape[0] # Number of rows in the dataframe
    tzlocal = tz.tzoffset(None, int(timezone * 3600)) # The timezone, defined by an offset from UTC in seconds
    alt = np.empty(timesteps)

    for timestep, row in epw.iterrows(): # For each hour of the year
        date = dt.datetime(row['Year'], row['Month'], row['Day'], row['Hour'] - 1, 30, 0, 0, tzlocal) # The local time, thirty minutes into the current hour
        alt[timestep] = sun.get_altitude(latitude, longitude, date) # The solar altitude in degrees, from the Pysolar package

    return alt
'''

def STAdj(DayOfYear, mer, lon):
    return 0.17 * np.sin((4 * np.pi / 373) * (DayOfYear - 80)) - 0.129 * np.sin((2 * np.pi / 355) * (DayOfYear - 8)) + 12 * ((mer - lon) * np.pi / 180) / np.pi

def soltimeadj(TimeOfDay, STAdj):
    return TimeOfDay + STAdj

def soldec(DayOfYear):
    return 0.4093 * np.sin((2 * np.pi / 368) * (DayOfYear - 81))

## 2. Calculate solar altitude for each hour in radians
def alt_clock_time(latitude, longitude, timezone, hourOfYear):
    return alt_clock_time_day_time(latitude, longitude, timezone, hourOfYear // 24, hourOfYear % 24)

def alt_clock_time_day_time(latitude, longitude, timezone, dayOfYear, hourOfDay):
    """Solar alt from day and time rather than hour of year.
    """

    lati = latitude * np.pi / 180
    mer = timezone * 15
    mySTAdj = STAdj(dayOfYear, mer, longitude)
    mySolDec = soldec(dayOfYear)
    mySolTimeAdj = soltimeadj(hourOfDay, mySTAdj)

    # Solar Altitude
    mySolAlt = np.arcsin(np.sin(lati) * np.sin(mySolDec) - np.cos(lati) * np.cos(mySolDec) * np.cos(mySolTimeAdj * (np.pi / 12)))
    return np.clip(mySolAlt, 0, np.pi/2)

"""
Input Routines
"""

## 1. Read in an EPW file to a Pandas dataframe
def readEPW(path):
    # Read the location
    df = pd.read_csv(path, nrows=1, header=None) # Read only the first row
    latitude = df.iloc[0, 6]
    longitude = df.iloc[0, 7]
    timezone = df.iloc[0, 8]

    # Read the hourly data
    df = pd.read_csv(path, skiprows=range(8), header=None) # Read the file, skip the 8 header rows
    # There are no column names in the file. We need to assign them.
    df.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Data Source and Uncertainty Flags',
                  'Dry Bulb Temperature', 'Dew Point Temperature', 'Relative Humidity', 'Atmospheric Station Pressure',
                  'Extraterrestrial Horizontal Radiation', 'Extraterrestrial Direct Normal Radiation', 'Horizontal Infrared Radiation Intensity',
                  'Global Horizontal Radiation', 'Direct Normal Radiation', 'Diffuse Horizontal Radiation',
                  'Global Horizontal Illuminance', 'Direct Normal Illuminance', 'Diffuse Horizontal Illuminance', 'Zenith Luminance',
                  'Wind Direction', 'Wind Speed', 'Total Sky Cover', 'Opaque Sky Cover', 'Visibility', 'Ceiling Height',
                  'Present Weather Observation', 'Present Weather Codes', 'Precipitable Water', 'Aerosol Optical Depth',
                  'Snow Depth', 'Days Since Last Snowfall', 'Albedo', 'Liquid Precipitation Depth', 'Liquid Precipitation Quantity']
    return df, latitude, longitude, timezone

## 2. Read in VTK and calc areas
#  path: file to read
#  df_select: list of points to query
def readAreas(path, df_select=None):
    npoints = 0
    nfaces = 0
    with open(path, 'r') as reader:
        # Read the number of points
        line = reader.readline()
        while line:
            if not line.startswith('POINTS'):
                line = reader.readline()
                continue
            npoints = int(line.split()[1])
            break

        # Read each point
        points = np.fromfile(reader, dtype=np.float32, sep=' ').reshape(npoints, 3)

        # Read number of polygons
        line = reader.readline()
        nfaces = int(line.split()[1])

        grid_areas = np.empty(nfaces)
        centers = np.empty((nfaces, 3))

        # Calculate the polygon areas
        for i in range(nfaces):
            line = reader.readline()
            entries = np.fromstring(line, dtype=int, sep=' ')
            grid_areas[i] = area(points[entries[1]], points[entries[2]], points[entries[3]])
            if entries[0] == 4:
                grid_areas[i] += area(points[entries[1]], points[entries[3]], points[entries[4]])
                centers[i] = barycenter([points[entries[1]], points[entries[2]], points[entries[3]], points[entries[4]]])
            else:
                centers[i] = barycenter([points[entries[1]], points[entries[2]], points[entries[3]]])

    # Find nearest points and set indices in df_select in place
    if df_select is not None:
        findClosestPoints(df_select, points, 'Vertex Index')
        findClosestPoints(df_select, centers, 'Cell Index')

    return grid_areas, npoints, nfaces

def readSelectPoints(path):
    """
    Read in a dataframe of points of interest, if available

    Parameters
    ----------
    path : string
        Path to the file of selected points of interest.

    Returns
    -------
    dataframe
        Contents of the file with added storage for closest vertex and cell indices.

    """
    headings = ['Name', 'Grid', 'X', 'Y', 'Z']
    if validFile(path):
        df_points = pd.read_csv(path, names=headings, header=0 if hasHeader(path) else None) # Read the file
    else:
        df_points = pd.DataFrame(columns=headings) # Empty
    df_points['Vertex Index'] = np.full(len(df_points.index), -1, dtype=np.int64)
    df_points['Cell Index'] = np.full(len(df_points.index), -1, dtype=np.int64)
    return df_points.set_index('Name')

## Read radiance points file
def readViews(path):
    views = np.loadtxt(path)
    points = views[:,0:3]
    directions = views[:,3:6]
    return points, directions

def readRadiance(path):
    """
    Read a Radiance output file, either in ASCII or binary format

    Parameters
    ----------
    path : string
        Path to the Radiance output file.

    Raises
    ------
    Exception
        A problem reading the file.

    Returns
    -------
    data : ndarray
        Data contained in the Radiance file.

    """
    nrows = 0
    ncols = 1
    ncomp = 3
    formt = 'ascii'
    with open(path, 'rb') as reader:
        # Read the file header
        for line in reader:
            decoded = line.decode('ascii').strip()
            #print(decoded)

            if len(decoded) == 0: # end of header
                break

            entries = decoded.split('=')
            if entries[0] == 'NROWS':
                if len(entries) < 2 or not entries[1].isdigit():
                    raise Exception('File %s has malformed %s' % (path, decoded))
                nrows = int(entries[1])
            elif entries[0] == 'NCOLS':
                if len(entries) < 2 or not entries[1].isdigit():
                    raise Exception('File %s has malformed %s' % (path, decoded))
                ncols = int(entries[1])
            elif entries[0] == 'NCOMP':
                if len(entries) < 2 or not entries[1].isdigit():
                    raise Exception('File %s has malformed %s' % (path, decoded))
                ncomp = int(entries[1])
                if ncomp != 1 and ncomp != 3:
                    raise Exception('File %s has bad %s. Must be 1 or 3.' % (path, decoded))
            elif entries[0] == 'FORMAT':
                if len(entries) < 2:
                    raise Exception('File %s has malformed %s' % (path, decoded))
                formt = entries[1].lower()

        if formt[0] == 'a': # ascii
            '''
            for line in reader:
                decoded = line.decode('ascii').strip()
                entries = decoded.split()
                for col in range(ncols):
                    data[row, col] = float(entries[col])
                row += 1
                break
            '''
            data = np.fromfile(reader, dtype=float, sep=' ')
        elif formt[0] == 'f': # float
            data = np.fromfile(reader, dtype=np.float32)
        elif formt[0] == 'd': # double
            data = np.fromfile(reader, dtype=np.float64)

        # Count rows if the information was not provided
        if nrows == 0:
            nrows = data.size // (ncols * ncomp)

        # Reshape input to matrix
        if ncomp == 1:
            data = data.reshape(nrows, ncols)
        else:
            data = data.reshape(nrows, ncols, ncomp)

    return data

def bright(red, green, blue, solar=False):
    """
    Calculate the combined brightness of Radiance color chanels

    Parameters
    ----------
    red : array_like
        Red chanel value(s).
    green : array_like
        Green chanel value(s).
    blue : array_like
        Blue chanel value(s).
    solar : boolean, optional
        Flag indicating whether full solar spectrum data is provided in the
        chanels. The default is False.

    Returns
    -------
    array_like
        The combined brighness for each set of input RGB values.

    """
    if solar:
        return (red + green + blue) / 3
    return 0.265 * red + 0.76 * green + 0.065 * blue

def calculateIrradiance(dc, smx, solar=False):
    """
    Calculate irraidance from daylight coefficients and sky matrix

    Parameters
    ----------
    dc : TYPE
        Daylight coefficient matrix.
    smx : TYPE
        Sky matrix.
    solar : boolean, optional
        Flag indicating whether full solar spectrum data is provided in smx.
        The default is False.

    Raises
    ------
    Exception
        Mismatched daylight coefficient and sky matrices.

    Returns
    -------
    ndarray
        Irradiance at each point (rows) and for each time (columns).

    """
    if dc.shape[1] != smx.shape[0]:
        raise Exception('Mismatched daylight coefficient and sky matrices (dc(%d, %d) s(%d, %d))' % (dc.shape[0], dc.shape[1], smx.shape[0], smx.shape[1]))

    if dc.ndim == 2:
        if smx.ndim == 2:
            return np.dot(dc, smx)
        else:
            red = np.dot(dc, smx[:,:,0])
            grn = np.dot(dc, smx[:,:,1])
            blu = np.dot(dc, smx[:,:,2])
    else:
        if smx.ndim == 2:
            red = np.dot(dc[:,:,0], smx)
            grn = np.dot(dc[:,:,1], smx)
            blu = np.dot(dc[:,:,2], smx)
        else:
            red = np.dot(dc[:,:,0], smx[:,:,0])
            grn = np.dot(dc[:,:,1], smx[:,:,1])
            blu = np.dot(dc[:,:,2], smx[:,:,2])

    return bright(red, grn, blu, solar)

"""
Sky Subdivision Routines
"""

class ReinhartSky:
    def __init__(self, divisions):
        tnaz = [ 30, 30, 24, 24, 18, 12, 6 ] # Number of patches per row
        divCounts = [ 2, 146, 578, 1298, 2306, 3602, 5186, 7058, 9218, 11666, 14402, 17426, 20738 ]
    
        # Check sky partitions
        mf = divCounts.index(divisions)
    
        # Calculate patches per row
        self.rings = len(tnaz) * mf + 1
        self.patchesPerRow = np.empty(self.rings, dtype=np.int64)
        self.firstPatchIndex = np.empty(self.rings, dtype=np.int64)
        count = 1 # The below horizon patch
        for i in range(len(tnaz)):
            for j in range(mf):
                self.firstPatchIndex[i * mf + j] = count
                self.patchesPerRow[i * mf + j] = tnaz[i] * mf
                count += self.patchesPerRow[i * mf + j]
        self.firstPatchIndex[-1] = count
        self.patchesPerRow[-1] = 1
    
        # Calculate solid angle of patches in each row
        self.solidAngle = np.empty(self.rings)
        self.ringElevationAngle = np.pi / (2 * self.rings - 1)
        self.solidAngle[0] = 2 * np.pi
        remaining = np.pi / 2
        for i in range(1, self.rings):
            remaining -= self.ringElevationAngle
            self.solidAngle[i] = 2 * np.pi * (1 - np.cos(remaining)) # solid angle of cap
            self.solidAngle[i - 1] -= self.solidAngle[i]
            self.solidAngle[i - 1] /= self.patchesPerRow[i - 1]

    def get_patch_direction(self, patch):
        row = np.digitize(patch, self.firstPatchIndex) - 1

        alt = np.pi / 2 - self.ringElevationAngle * (row - (self.rings - 1))
        azi = 2 * np.pi * (patch - self.firstPatchIndex[row]) / self.patchesPerRow[row]
        cos_alt = np.cos(alt)
        x = np.where(row < 0, 0, cos_alt * -np.sin(azi))
        y = np.where(row < 0, 0, cos_alt * -np.cos(azi))
        z = np.where(row < 0, -1, np.sin(alt))
        
        return np.column_stack((x, y, z))

    def get_patch_solid_angle(self, patch, cos_theta):
        row = np.digitize(patch, self.firstPatchIndex) - 1

        # Solid angle overlap between visible hemisphere and ground hemisphere
        return np.where(row < 0, 2 * (np.pi - 2 * np.arccos(cos_theta)), self.solidAngle[row])

"""
Comfort Routines
"""

## 4. Calculate MRT
## Calculate temperature of inside of glazing in degC
def inside_glazing_temperature(dbt_out, dbt_in, u_glaz, h_out, h_in):
    q = (dbt_out - dbt_in) / (1 / h_out + 1 / u_glaz + 1 / h_in) # q = U*deltaT
    t_inside_of_glazing = q / h_in + dbt_in
    return t_inside_of_glazing

## 5. Calculate MRT*
#  T_ra: mean radiant temperature
#  alt: solar altitude in radians
#  irrad: solar irradiance in W/m2
#  shgc: of glass, defaults to 1
#  absorptivity: of person
#  emissivity: of person
def calculateMRTStar(T_ra, alt, irrad, shgc=1.0, absorptivity=0.7, emissivity=0.9):
    fp = np.cos(alt) / np.pi # for a standing person (contains 8760 values)
    T_ras = np.power(np.power(T_ra + C2K, 4) + (absorptivity * shgc / (emissivity * stefan_boltzmann)) * fp * irrad, 0.25) - C2K
    return T_ras

# Saturation vapor pressure in Pascals
#  tk in Kelvin, can be array of values
def Pws(tk):
    pws = np.power(10, 12.5380997 - 2663.91 / tk) # for values below freezing
    np.putmask(pws, tk > C2K, np.power(10, 33.59051 - 8.2 * np.log10(tk) + 0.0024804 * tk - 3142.31 / tk)) # above freezing
    return pws

# Vapor pressure in Pascals
#  dbt in Kelvin, can be array of values
#  rh as percent, can be array of values
def Pa(dbt, rh):
    return rh / 100 * Pws(dbt)

## 7. Calculate Predicted Mean Vote
#  dbt in Kelvin, can be array of values
#  mrt in Kelvin, can be array of values
#  rh as percent, can be array of values
#  vel in m/s, can be array of values
#  clo in CLO units
#  met in Watts
#  w in Watts, usually zero
def calculatePMV(dbt, mrt, rh, vel, clo, met, w=0):
    k = 39.6e-9

    # calculate vapour pressure
    pa = Pa(dbt, rh)

    # body heat equations
    eres = 1.72e-5 * met * (5867 - pa)
    cres = 1.4e-3 * met * (34 + C2K - dbt)
    Ec = 3.05e-3 * (5733 - 6.99 * (met - w) - pa) + 0.42 * (met - w - 58.15)

    # Skin temperature
    tsk = 35.7 + C2K - 0.028 * (met - w)

    # determine effective clothing value
    icl = clo * 0.155
    fcl = 1.05 + 0.645 * icl
    np.putmask(fcl, icl < 0.078, 1 + 1.29 * icl)

    # Initilse clothing Temperature to Skin Temperature
    tcl = 0.5 * tsk + 0.25 * (dbt + mrt)

    hc_vel = 12.1 * np.sqrt(vel)

    while True:
        hc_nat = np.zeros(tcl.shape)
        np.putmask(hc_nat, tcl >= dbt, 2.38 * np.power(tcl - dbt, 0.25))
        hc = np.maximum(hc_nat, hc_vel)

        # Iterative process to determine tcl, for now assume skin T
        tcl_new = tsk - fcl * icl * k * (np.power(tcl, 4) - np.power(mrt, 4)) - fcl * icl * hc * (tcl - dbt)

        tcl_err = tcl_new - tcl
        if np.absolute(tcl_err).max() > 0.2:
            tcl = tcl + 0.4 * tcl_err
        else:
            tcl = tcl_new
            break

    # Heat loss from person
    H = k * fcl * (np.power(tcl, 4) - np.power(mrt, 4)) + fcl * hc * (tcl - dbt)

    # PMV
    pmv = (0.303 * np.exp(-0.036 * met) + 0.028) * ((met - w) - H - Ec - cres - eres)
    return pmv

## Calculate Predicted Percentage Dissatisfied
def calculatePPD(pmv):
    ppd = 100.0 - 95.0 * np.exp(-(0.03353 * np.power(pmv, 4) + 0.2179 * np.power(pmv, 2)))
    return ppd

def rhFromDewPoint(db_temp, dewpoint_temp):
    """
    Calculate RH from dewpoint and DBT.

    Accepts either single values or vectorized arrays.

    Parameters
    ----------
    db_temp : float
        Dry bulb temperature in deg C
    dewpoint_temp : float
        Dewpoint in deg C

    Returns
    -------
    relative_humidity : float
        Relative humidity bounded [0, 100]

    """

    c_b = 17.625
    c_c = 243.04

    relative_humidity = 100*np.exp((c_b*c_c*(dewpoint_temp - db_temp)) \
                            /((c_c + db_temp)*(c_c + dewpoint_temp)))

    return relative_humidity

def globalHorizontalZHM(solar_alt, cloud_cover, delta_temp, relative_humidity, wind_speed, irrad_extra=1355):
    """
    Approximate global horizontal radiation from hourly weather data.

    Accepts either single values or vectorized arrays.
    
    Parameters
    ----------
    solar_alt : float or array_like
        Solar altitude in degrees
    cloud_cover : float or ndarray
        Cloud cover fraction bounded [0, 1]
    delta_temp : float or ndarray
        Temperature difference between current and previous hours
        in degrees Kelvin.
    relative_humidity : float or ndarray
        Relative humidity bounded [0, 100]
    wind_speed : float or ndarray
        Wind speed in m/s
    irrad_extra : float or array_like, optional
        Extraterrestrial irradiance in W/m2. The default is 1355.

    Returns
    -------
    global_horizontal : float or ndarray
        Global horizontal radiation in W/m^2

    """

    c_0 = 0.5598
    c_1 = 0.4982
    c_2 = -0.6762
    c_3 = 0.02842
    c_4 = -0.00317
    c_5 = 0.014
    d = -17.853
    k = 0.843

    global_horizontal = (irrad_extra*np.sin(np.radians(solar_alt))*c_0 \
                        + c_1*cloud_cover + c_2*cloud_cover**2 \
                        + c_3*delta_temp + c_4*relative_humidity/100.0 \
                        + c_5*wind_speed + d)/k

    # Set values less than zero to zero.
    if np.isscalar(global_horizontal):
        if global_horizontal < 0:
            global_horizontal = 0
    else:
        global_horizontal[global_horizontal < 0] = 0

    return global_horizontal

def findDeltaTemp(temperature):
    """Find difference between 1d array elements.

    Formula is `dT[n] = T[n] - T[n-1]` with first element doubled up to
    maintain the same shape.

    Parameters
    ----------
    temperature : array
        1d array
    
    Returns
    -------
    dt : array
        1d array of the same shape as input
    """

    dt = np.ediff1d(temperature)

    # double up on first value to maintain shape
    dt = np.concatenate(([dt[0]], dt))

    return dt

## Calculate Universal Thermal Climate Index
#  dbt in degrees C, can be array of values
#  mrt in degrees C, can be array of values
#  rh as percent, can be array of values
#  vel in m/s, can be array of values
def calculateUTCI_old(dbt, mrt, rh, vel):
    delta_mrt = mrt - dbt
    pa = Pa(dbt + C2K, rh) / 1000

    UTCI_approx = dbt + \
                (6.07562052e-01) + \
                (-2.27712343e-02) * dbt + \
                (8.06470249e-04) * dbt * dbt + \
                (-1.54271372e-04) * dbt * dbt * dbt + \
                (-3.24651735e-06) * dbt * dbt * dbt * dbt + \
                (7.32602852e-08) * dbt * dbt * dbt * dbt * dbt + \
                (1.35959073e-09) * dbt * dbt * dbt * dbt * dbt * dbt + \
                (-2.25836520e+00) * vel + \
                (8.80326035e-02) * dbt * vel + \
                (2.16844454e-03) * dbt * dbt * vel + \
                (-1.53347087e-05) * dbt * dbt * dbt * vel + \
                (-5.72983704e-07) * dbt * dbt * dbt * dbt * vel + \
                (-2.55090145e-09) * dbt * dbt * dbt * dbt * dbt * vel + \
                (-7.51269505e-01) * vel * vel + \
                (-4.08350271e-03) * dbt * vel * vel + \
                (-5.21670675e-05) * dbt * dbt * vel * vel + \
                (1.94544667e-06) * dbt * dbt * dbt * vel * vel + \
                (1.14099531e-08) * dbt * dbt * dbt * dbt * vel * vel + \
                (1.58137256e-01) * vel * vel * vel + \
                (-6.57263143e-05) * dbt * vel * vel * vel + \
                (2.22697524e-07) * dbt * dbt * vel * vel * vel + \
                (-4.16117031e-08) * dbt * dbt * dbt * vel * vel * vel + \
                (-1.27762753e-02) * vel * vel * vel * vel + \
                (9.66891875e-06) * dbt * vel * vel * vel * vel + \
                (2.52785852e-09) * dbt * dbt * vel * vel * vel * vel + \
                (4.56306672e-04) * vel * vel * vel * vel * vel + \
                (-1.74202546e-07) * dbt * vel * vel * vel * vel * vel + \
                (-5.91491269e-06) * vel * vel * vel * vel * vel * vel + \
                (3.98374029e-01) * delta_mrt + \
                (1.83945314e-04) * dbt * delta_mrt + \
                (-1.73754510e-04) * dbt * dbt * delta_mrt + \
                (-7.60781159e-07) * dbt * dbt * dbt * delta_mrt + \
                (3.77830287e-08) * dbt * dbt * dbt * dbt * delta_mrt + \
                (5.43079673e-10) * dbt * dbt * dbt * dbt * dbt * delta_mrt + \
                (-2.00518269e-02) * vel * delta_mrt + \
                (8.92859837e-04) * dbt * vel * delta_mrt + \
                (3.45433048e-06) * dbt * dbt * vel * delta_mrt + \
                (-3.77925774e-07) * dbt * dbt * dbt * vel * delta_mrt + \
                (-1.69699377e-09) * dbt * dbt * dbt * dbt * vel * delta_mrt + \
                (1.69992415e-04) * vel * vel * delta_mrt + \
                (-4.99204314e-05) * dbt * vel * vel * delta_mrt + \
                (2.47417178e-07) * dbt * dbt * vel * vel * delta_mrt + \
                (1.07596466e-08) * dbt * dbt * dbt * vel * vel * delta_mrt + \
                (8.49242932e-05) * vel * vel * vel * delta_mrt + \
                (1.35191328e-06) * dbt * vel * vel * vel * delta_mrt + \
                (-6.21531254e-09) * dbt * dbt * vel * vel * vel * delta_mrt + \
                (-4.99410301e-06) * vel * vel * vel * vel * delta_mrt + \
                (-1.89489258e-08) * dbt * vel * vel * vel * vel * delta_mrt + \
                (8.15300114e-08) * vel * vel * vel * vel * vel * delta_mrt + \
                (7.55043090e-04) * delta_mrt * delta_mrt + \
                (-5.65095215e-05) * dbt * delta_mrt * delta_mrt + \
                (-4.52166564e-07) * dbt * dbt * delta_mrt * delta_mrt + \
                (2.46688878e-08) * dbt * dbt * dbt * delta_mrt * delta_mrt + \
                (2.42674348e-10) * dbt * dbt * dbt * dbt * delta_mrt * delta_mrt + \
                (1.54547250e-04) * vel * delta_mrt * delta_mrt + \
                (5.24110970e-06) * dbt * vel * delta_mrt * delta_mrt + \
                (-8.75874982e-08) * dbt * dbt * vel * delta_mrt * delta_mrt + \
                (-1.50743064e-09) * dbt * dbt * dbt * vel * delta_mrt * delta_mrt + \
                (-1.56236307e-05) * vel * vel * delta_mrt * delta_mrt + \
                (-1.33895614e-07) * dbt * vel * vel * delta_mrt * delta_mrt + \
                (2.49709824e-09) * dbt * dbt * vel * vel * delta_mrt * delta_mrt + \
                (6.51711721e-07) * vel * vel * vel * delta_mrt * delta_mrt + \
                (1.94960053e-09) * dbt * vel * vel * vel * delta_mrt * delta_mrt + \
                (-1.00361113e-08) * vel * vel * vel * vel * delta_mrt * delta_mrt + \
                (-1.21206673e-05) * delta_mrt * delta_mrt * delta_mrt + \
                (-2.18203660e-07) * dbt * delta_mrt * delta_mrt * delta_mrt + \
                (7.51269482e-09) * dbt * dbt * delta_mrt * delta_mrt * delta_mrt + \
                (9.79063848e-11) * dbt * dbt * dbt * delta_mrt * delta_mrt * delta_mrt + \
                (1.25006734e-06) * vel * delta_mrt * delta_mrt * delta_mrt + \
                (-1.81584736e-09) * dbt * vel * delta_mrt * delta_mrt * delta_mrt + \
                (-3.52197671e-10) * dbt * dbt * vel * delta_mrt * delta_mrt * delta_mrt + \
                (-3.36514630e-08) * vel * vel * delta_mrt * delta_mrt * delta_mrt + \
                (1.35908359e-10) * dbt * vel * vel * delta_mrt * delta_mrt * delta_mrt + \
                (4.17032620e-10) * vel * vel * vel * delta_mrt * delta_mrt * delta_mrt + \
                (-1.30369025e-09) * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (4.13908461e-10) * dbt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (9.22652254e-12) * dbt * dbt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (-5.08220384e-09) * vel * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (-2.24730961e-11) * dbt * vel * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (1.17139133e-10) * vel * vel * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (6.62154879e-10) * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (4.03863260e-13) * dbt * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (1.95087203e-12) * vel * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (-4.73602469e-12) * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt + \
                (5.12733497e+00) * pa + \
                (-3.12788561e-01) * dbt * pa + \
                (-1.96701861e-02) * dbt * dbt * pa + \
                (9.99690870e-04) * dbt * dbt * dbt * pa + \
                (9.51738512e-06) * dbt * dbt * dbt * dbt * pa + \
                (-4.66426341e-07) * dbt * dbt * dbt * dbt * dbt * pa + \
                (5.48050612e-01) * vel * pa + \
                (-3.30552823e-03) * dbt * vel * pa + \
                (-1.64119440e-03) * dbt * dbt * vel * pa + \
                (-5.16670694e-06) * dbt * dbt * dbt * vel * pa + \
                (9.52692432e-07) * dbt * dbt * dbt * dbt * vel * pa + \
                (-4.29223622e-02) * vel * vel * pa + \
                (5.00845667e-03) * dbt * vel * vel * pa + \
                (1.00601257e-06) * dbt * dbt * vel * vel * pa + \
                (-1.81748644e-06) * dbt * dbt * dbt * vel * vel * pa + \
                (-1.25813502e-03) * vel * vel * vel * pa + \
                (-1.79330391e-04) * dbt * vel * vel * vel * pa + \
                (2.34994441e-06) * dbt * dbt * vel * vel * vel * pa + \
                (1.29735808e-04) * vel * vel * vel * vel * pa + \
                (1.29064870e-06) * dbt * vel * vel * vel * vel * pa + \
                (-2.28558686e-06) * vel * vel * vel * vel * vel * pa + \
                (-3.69476348e-02) * delta_mrt * pa + \
                (1.62325322e-03) * dbt * delta_mrt * pa + \
                (-3.14279680e-05) * dbt * dbt * delta_mrt * pa + \
                (2.59835559e-06) * dbt * dbt * dbt * delta_mrt * pa + \
                (-4.77136523e-08) * dbt * dbt * dbt * dbt * delta_mrt * pa + \
                (8.64203390e-03) * vel * delta_mrt * pa + \
                (-6.87405181e-04) * dbt * vel * delta_mrt * pa + \
                (-9.13863872e-06) * dbt * dbt * vel * delta_mrt * pa + \
                (5.15916806e-07) * dbt * dbt * dbt * vel * delta_mrt * pa + \
                (-3.59217476e-05) * vel * vel * delta_mrt * pa + \
                (3.28696511e-05) * dbt * vel * vel * delta_mrt * pa + \
                (-7.10542454e-07) * dbt * dbt * vel * vel * delta_mrt * pa + \
                (-1.24382300e-05) * vel * vel * vel * delta_mrt * pa + \
                (-7.38584400e-09) * dbt * vel * vel * vel * delta_mrt * pa + \
                (2.20609296e-07) * vel * vel * vel * vel * delta_mrt * pa + \
                (-7.32469180e-04) * delta_mrt * delta_mrt * pa + \
                (-1.87381964e-05) * dbt * delta_mrt * delta_mrt * pa + \
                (4.80925239e-06) * dbt * dbt * delta_mrt * delta_mrt * pa + \
                (-8.75492040e-08) * dbt * dbt * dbt * delta_mrt * delta_mrt * pa + \
                (2.77862930e-05) * vel * delta_mrt * delta_mrt * pa + \
                (-5.06004592e-06) * dbt * vel * delta_mrt * delta_mrt * pa + \
                (1.14325367e-07) * dbt * dbt * vel * delta_mrt * delta_mrt * pa + \
                (2.53016723e-06) * vel * vel * delta_mrt * delta_mrt * pa + \
                (-1.72857035e-08) * dbt * vel * vel * delta_mrt * delta_mrt * pa + \
                (-3.95079398e-08) * vel * vel * vel * delta_mrt * delta_mrt * pa + \
                (-3.59413173e-07) * delta_mrt * delta_mrt * delta_mrt * pa + \
                (7.04388046e-07) * dbt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (-1.89309167e-08) * dbt * dbt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (-4.79768731e-07) * vel * delta_mrt * delta_mrt * delta_mrt * pa + \
                (7.96079978e-09) * dbt * vel * delta_mrt * delta_mrt * delta_mrt * pa + \
                (1.62897058e-09) * vel * vel * delta_mrt * delta_mrt * delta_mrt * pa + \
                (3.94367674e-08) * delta_mrt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (-1.18566247e-09) * dbt * delta_mrt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (3.34678041e-10) * vel * delta_mrt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (-1.15606447e-10) * delta_mrt * delta_mrt * delta_mrt * delta_mrt * delta_mrt * pa + \
                (-2.80626406e+00) * pa * pa + \
                (5.48712484e-01) * dbt * pa * pa + \
                (-3.99428410e-03) * dbt * dbt * pa * pa + \
                (-9.54009191e-04) * dbt * dbt * dbt * pa * pa + \
                (1.93090978e-05) * dbt * dbt * dbt * dbt * pa * pa + \
                (-3.08806365e-01) * vel * pa * pa + \
                (1.16952364e-02) * dbt * vel * pa * pa + \
                (4.95271903e-04) * dbt * dbt * vel * pa * pa + \
                (-1.90710882e-05) * dbt * dbt * dbt * vel * pa * pa + \
                (2.10787756e-03) * vel * vel * pa * pa + \
                (-6.98445738e-04) * dbt * vel * vel * pa * pa + \
                (2.30109073e-05) * dbt * dbt * vel * vel * pa * pa + \
                (4.17856590e-04) * vel * vel * vel * pa * pa + \
                (-1.27043871e-05) * dbt * vel * vel * vel * pa * pa + \
                (-3.04620472e-06) * vel * vel * vel * vel * pa * pa + \
                (5.14507424e-02) * delta_mrt * pa * pa + \
                (-4.32510997e-03) * dbt * delta_mrt * pa * pa + \
                (8.99281156e-05) * dbt * dbt * delta_mrt * pa * pa + \
                (-7.14663943e-07) * dbt * dbt * dbt * delta_mrt * pa * pa + \
                (-2.66016305e-04) * vel * delta_mrt * pa * pa + \
                (2.63789586e-04) * dbt * vel * delta_mrt * pa * pa + \
                (-7.01199003e-06) * dbt * dbt * vel * delta_mrt * pa * pa + \
                (-1.06823306e-04) * vel * vel * delta_mrt * pa * pa + \
                (3.61341136e-06) * dbt * vel * vel * delta_mrt * pa * pa + \
                (2.29748967e-07) * vel * vel * vel * delta_mrt * pa * pa + \
                (3.04788893e-04) * delta_mrt * delta_mrt * pa * pa + \
                (-6.42070836e-05) * dbt * delta_mrt * delta_mrt * pa * pa + \
                (1.16257971e-06) * dbt * dbt * delta_mrt * delta_mrt * pa * pa + \
                (7.68023384e-06) * vel * delta_mrt * delta_mrt * pa * pa + \
                (-5.47446896e-07) * dbt * vel * delta_mrt * delta_mrt * pa * pa + \
                (-3.59937910e-08) * vel * vel * delta_mrt * delta_mrt * pa * pa + \
                (-4.36497725e-06) * delta_mrt * delta_mrt * delta_mrt * pa * pa + \
                (1.68737969e-07) * dbt * delta_mrt * delta_mrt * delta_mrt * pa * pa + \
                (2.67489271e-08) * vel * delta_mrt * delta_mrt * delta_mrt * pa * pa + \
                (3.23926897e-09) * delta_mrt * delta_mrt * delta_mrt * delta_mrt * pa * pa + \
                (-3.53874123e-02) * pa * pa * pa + \
                (-2.21201190e-01) * dbt * pa * pa * pa + \
                (1.55126038e-02) * dbt * dbt * pa * pa * pa + \
                (-2.63917279e-04) * dbt * dbt * dbt * pa * pa * pa + \
                (4.53433455e-02) * vel * pa * pa * pa + \
                (-4.32943862e-03) * dbt * vel * pa * pa * pa + \
                (1.45389826e-04) * dbt * dbt * vel * pa * pa * pa + \
                (2.17508610e-04) * vel * vel * pa * pa * pa + \
                (-6.66724702e-05) * dbt * vel * vel * pa * pa * pa + \
                (3.33217140e-05) * vel * vel * vel * pa * pa * pa + \
                (-2.26921615e-03) * delta_mrt * pa * pa * pa + \
                (3.80261982e-04) * dbt * delta_mrt * pa * pa * pa + \
                (-5.45314314e-09) * dbt * dbt * delta_mrt * pa * pa * pa + \
                (-7.96355448e-04) * vel * delta_mrt * pa * pa * pa + \
                (2.53458034e-05) * dbt * vel * delta_mrt * pa * pa * pa + \
                (-6.31223658e-06) * vel * vel * delta_mrt * pa * pa * pa + \
                (3.02122035e-04) * delta_mrt * delta_mrt * pa * pa * pa + \
                (-4.77403547e-06) * dbt * delta_mrt * delta_mrt * pa * pa * pa + \
                (1.73825715e-06) * vel * delta_mrt * delta_mrt * pa * pa * pa + \
                (-4.09087898e-07) * delta_mrt * delta_mrt * delta_mrt * pa * pa * pa + \
                (6.14155345e-01) * pa * pa * pa * pa + \
                (-6.16755931e-02) * dbt * pa * pa * pa * pa + \
                (1.33374846e-03) * dbt * dbt * pa * pa * pa * pa + \
                (3.55375387e-03) * vel * pa * pa * pa * pa + \
                (-5.13027851e-04) * dbt * vel * pa * pa * pa * pa + \
                (1.02449757e-04) * vel * vel * pa * pa * pa * pa + \
                (-1.48526421e-03) * delta_mrt * pa * pa * pa * pa + \
                (-4.11469183e-05) * dbt * delta_mrt * pa * pa * pa * pa + \
                (-6.80434415e-06) * vel * delta_mrt * pa * pa * pa * pa + \
                (-9.77675906e-06) * delta_mrt * delta_mrt * pa * pa * pa * pa + \
                (8.82773108e-02) * pa * pa * pa * pa * pa + \
                (-3.01859306e-03) * dbt * pa * pa * pa * pa * pa + \
                (1.04452989e-03) * vel * pa * pa * pa * pa * pa + \
                (2.47090539e-04) * delta_mrt * pa * pa * pa * pa * pa + \
                (1.48348065e-03) * pa * pa * pa * pa * pa * pa
    return UTCI_approx

## Calculate Universal Thermal Climate Index
#  dbt in degrees C, can be array of values
#  mrt in degrees C, can be array of values
#  rh as percent, can be array of values
#  vel in m/s, can be array of values
def calculateUTCI(dbt, mrt, rh, vel):
    delta_mrt = mrt - dbt
    pa = Pa(dbt + C2K, rh) / 1000

    dbt2 = dbt * dbt
    dbt3 = dbt2 * dbt
    dbt4 = dbt3 * dbt
    dbt5 = dbt4 * dbt

    vel2 = vel * vel
    vel3 = vel2 * vel
    vel4 = vel3 * vel
    vel5 = vel4 * vel

    delta_mrt2 = delta_mrt * delta_mrt
    delta_mrt3 = delta_mrt2 * delta_mrt
    delta_mrt4 = delta_mrt3 * delta_mrt
    delta_mrt5 = delta_mrt4 * delta_mrt

    pa2 = pa * pa
    pa3 = pa2 * pa
    pa4 = pa3 * pa
    pa5 = pa4 * pa

    UTCI_approx = dbt + \
                (6.07562052e-01) + \
                (-2.27712343e-02) * dbt + \
                (8.06470249e-04) * dbt2 + \
                (-1.54271372e-04) * dbt3 + \
                (-3.24651735e-06) * dbt4 + \
                (7.32602852e-08) * dbt5 + \
                (1.35959073e-09) * dbt5 * dbt + \
                (-2.25836520e+00) * vel + \
                (8.80326035e-02) * dbt * vel + \
                (2.16844454e-03) * dbt2 * vel + \
                (-1.53347087e-05) * dbt3 * vel + \
                (-5.72983704e-07) * dbt4 * vel + \
                (-2.55090145e-09) * dbt5 * vel + \
                (-7.51269505e-01) * vel2 + \
                (-4.08350271e-03) * dbt * vel2 + \
                (-5.21670675e-05) * dbt2 * vel2 + \
                (1.94544667e-06) * dbt3 * vel2 + \
                (1.14099531e-08) * dbt4 * vel2 + \
                (1.58137256e-01) * vel3 + \
                (-6.57263143e-05) * dbt * vel3 + \
                (2.22697524e-07) * dbt2 * vel3 + \
                (-4.16117031e-08) * dbt3 * vel3 + \
                (-1.27762753e-02) * vel4 + \
                (9.66891875e-06) * dbt * vel4 + \
                (2.52785852e-09) * dbt2 * vel4 + \
                (4.56306672e-04) * vel5 + \
                (-1.74202546e-07) * dbt * vel5 + \
                (-5.91491269e-06) * vel5 * vel + \
                (3.98374029e-01) * delta_mrt + \
                (1.83945314e-04) * dbt * delta_mrt + \
                (-1.73754510e-04) * dbt2 * delta_mrt + \
                (-7.60781159e-07) * dbt3 * delta_mrt + \
                (3.77830287e-08) * dbt4 * delta_mrt + \
                (5.43079673e-10) * dbt5 * delta_mrt + \
                (-2.00518269e-02) * vel * delta_mrt + \
                (8.92859837e-04) * dbt * vel * delta_mrt + \
                (3.45433048e-06) * dbt2 * vel * delta_mrt + \
                (-3.77925774e-07) * dbt3 * vel * delta_mrt + \
                (-1.69699377e-09) * dbt4 * vel * delta_mrt + \
                (1.69992415e-04) * vel2 * delta_mrt + \
                (-4.99204314e-05) * dbt * vel2 * delta_mrt + \
                (2.47417178e-07) * dbt2 * vel2 * delta_mrt + \
                (1.07596466e-08) * dbt3 * vel2 * delta_mrt + \
                (8.49242932e-05) * vel3 * delta_mrt + \
                (1.35191328e-06) * dbt * vel3 * delta_mrt + \
                (-6.21531254e-09) * dbt2 * vel3 * delta_mrt + \
                (-4.99410301e-06) * vel4 * delta_mrt + \
                (-1.89489258e-08) * dbt * vel4 * delta_mrt + \
                (8.15300114e-08) * vel5 * delta_mrt + \
                (7.55043090e-04) * delta_mrt2 + \
                (-5.65095215e-05) * dbt * delta_mrt2 + \
                (-4.52166564e-07) * dbt2 * delta_mrt2 + \
                (2.46688878e-08) * dbt3 * delta_mrt2 + \
                (2.42674348e-10) * dbt4 * delta_mrt2 + \
                (1.54547250e-04) * vel * delta_mrt2 + \
                (5.24110970e-06) * dbt * vel * delta_mrt2 + \
                (-8.75874982e-08) * dbt2 * vel * delta_mrt2 + \
                (-1.50743064e-09) * dbt3 * vel * delta_mrt2 + \
                (-1.56236307e-05) * vel2 * delta_mrt2 + \
                (-1.33895614e-07) * dbt * vel2 * delta_mrt2 + \
                (2.49709824e-09) * dbt2 * vel2 * delta_mrt2 + \
                (6.51711721e-07) * vel3 * delta_mrt2 + \
                (1.94960053e-09) * dbt * vel3 * delta_mrt2 + \
                (-1.00361113e-08) * vel4 * delta_mrt2 + \
                (-1.21206673e-05) * delta_mrt3 + \
                (-2.18203660e-07) * dbt * delta_mrt3 + \
                (7.51269482e-09) * dbt2 * delta_mrt3 + \
                (9.79063848e-11) * dbt3 * delta_mrt3 + \
                (1.25006734e-06) * vel * delta_mrt3 + \
                (-1.81584736e-09) * dbt * vel * delta_mrt3 + \
                (-3.52197671e-10) * dbt2 * vel * delta_mrt3 + \
                (-3.36514630e-08) * vel2 * delta_mrt3 + \
                (1.35908359e-10) * dbt * vel2 * delta_mrt3 + \
                (4.17032620e-10) * vel3 * delta_mrt3 + \
                (-1.30369025e-09) * delta_mrt4 + \
                (4.13908461e-10) * dbt * delta_mrt4 + \
                (9.22652254e-12) * dbt2 * delta_mrt4 + \
                (-5.08220384e-09) * vel * delta_mrt4 + \
                (-2.24730961e-11) * dbt * vel * delta_mrt4 + \
                (1.17139133e-10) * vel2 * delta_mrt4 + \
                (6.62154879e-10) * delta_mrt5 + \
                (4.03863260e-13) * dbt * delta_mrt5 + \
                (1.95087203e-12) * vel * delta_mrt5 + \
                (-4.73602469e-12) * delta_mrt5 * delta_mrt + \
                (5.12733497e+00) * pa + \
                (-3.12788561e-01) * dbt * pa + \
                (-1.96701861e-02) * dbt2 * pa + \
                (9.99690870e-04) * dbt3 * pa + \
                (9.51738512e-06) * dbt4 * pa + \
                (-4.66426341e-07) * dbt5 * pa + \
                (5.48050612e-01) * vel * pa + \
                (-3.30552823e-03) * dbt * vel * pa + \
                (-1.64119440e-03) * dbt2 * vel * pa + \
                (-5.16670694e-06) * dbt3 * vel * pa + \
                (9.52692432e-07) * dbt4 * vel * pa + \
                (-4.29223622e-02) * vel2 * pa + \
                (5.00845667e-03) * dbt * vel2 * pa + \
                (1.00601257e-06) * dbt2 * vel2 * pa + \
                (-1.81748644e-06) * dbt3 * vel2 * pa + \
                (-1.25813502e-03) * vel3 * pa + \
                (-1.79330391e-04) * dbt * vel3 * pa + \
                (2.34994441e-06) * dbt2 * vel3 * pa + \
                (1.29735808e-04) * vel4 * pa + \
                (1.29064870e-06) * dbt * vel4 * pa + \
                (-2.28558686e-06) * vel5 * pa + \
                (-3.69476348e-02) * delta_mrt * pa + \
                (1.62325322e-03) * dbt * delta_mrt * pa + \
                (-3.14279680e-05) * dbt2 * delta_mrt * pa + \
                (2.59835559e-06) * dbt3 * delta_mrt * pa + \
                (-4.77136523e-08) * dbt4 * delta_mrt * pa + \
                (8.64203390e-03) * vel * delta_mrt * pa + \
                (-6.87405181e-04) * dbt * vel * delta_mrt * pa + \
                (-9.13863872e-06) * dbt2 * vel * delta_mrt * pa + \
                (5.15916806e-07) * dbt3 * vel * delta_mrt * pa + \
                (-3.59217476e-05) * vel2 * delta_mrt * pa + \
                (3.28696511e-05) * dbt * vel2 * delta_mrt * pa + \
                (-7.10542454e-07) * dbt2 * vel2 * delta_mrt * pa + \
                (-1.24382300e-05) * vel3 * delta_mrt * pa + \
                (-7.38584400e-09) * dbt * vel3 * delta_mrt * pa + \
                (2.20609296e-07) * vel4 * delta_mrt * pa + \
                (-7.32469180e-04) * delta_mrt2 * pa + \
                (-1.87381964e-05) * dbt * delta_mrt2 * pa + \
                (4.80925239e-06) * dbt2 * delta_mrt2 * pa + \
                (-8.75492040e-08) * dbt3 * delta_mrt2 * pa + \
                (2.77862930e-05) * vel * delta_mrt2 * pa + \
                (-5.06004592e-06) * dbt * vel * delta_mrt2 * pa + \
                (1.14325367e-07) * dbt2 * vel * delta_mrt2 * pa + \
                (2.53016723e-06) * vel2 * delta_mrt2 * pa + \
                (-1.72857035e-08) * dbt * vel2 * delta_mrt2 * pa + \
                (-3.95079398e-08) * vel3 * delta_mrt2 * pa + \
                (-3.59413173e-07) * delta_mrt3 * pa + \
                (7.04388046e-07) * dbt * delta_mrt3 * pa + \
                (-1.89309167e-08) * dbt2 * delta_mrt3 * pa + \
                (-4.79768731e-07) * vel * delta_mrt3 * pa + \
                (7.96079978e-09) * dbt * vel * delta_mrt3 * pa + \
                (1.62897058e-09) * vel2 * delta_mrt3 * pa + \
                (3.94367674e-08) * delta_mrt4 * pa + \
                (-1.18566247e-09) * dbt * delta_mrt4 * pa + \
                (3.34678041e-10) * vel * delta_mrt4 * pa + \
                (-1.15606447e-10) * delta_mrt5 * pa + \
                (-2.80626406e+00) * pa2 + \
                (5.48712484e-01) * dbt * pa2 + \
                (-3.99428410e-03) * dbt2 * pa2 + \
                (-9.54009191e-04) * dbt3 * pa2 + \
                (1.93090978e-05) * dbt4 * pa2 + \
                (-3.08806365e-01) * vel * pa2 + \
                (1.16952364e-02) * dbt * vel * pa2 + \
                (4.95271903e-04) * dbt2 * vel * pa2 + \
                (-1.90710882e-05) * dbt3 * vel * pa2 + \
                (2.10787756e-03) * vel2 * pa2 + \
                (-6.98445738e-04) * dbt * vel2 * pa2 + \
                (2.30109073e-05) * dbt2 * vel2 * pa2 + \
                (4.17856590e-04) * vel3 * pa2 + \
                (-1.27043871e-05) * dbt * vel3 * pa2 + \
                (-3.04620472e-06) * vel4 * pa2 + \
                (5.14507424e-02) * delta_mrt * pa2 + \
                (-4.32510997e-03) * dbt * delta_mrt * pa2 + \
                (8.99281156e-05) * dbt2 * delta_mrt * pa2 + \
                (-7.14663943e-07) * dbt3 * delta_mrt * pa2 + \
                (-2.66016305e-04) * vel * delta_mrt * pa2 + \
                (2.63789586e-04) * dbt * vel * delta_mrt * pa2 + \
                (-7.01199003e-06) * dbt2 * vel * delta_mrt * pa2 + \
                (-1.06823306e-04) * vel2 * delta_mrt * pa2 + \
                (3.61341136e-06) * dbt * vel2 * delta_mrt * pa2 + \
                (2.29748967e-07) * vel3 * delta_mrt * pa2 + \
                (3.04788893e-04) * delta_mrt2 * pa2 + \
                (-6.42070836e-05) * dbt * delta_mrt2 * pa2 + \
                (1.16257971e-06) * dbt2 * delta_mrt2 * pa2 + \
                (7.68023384e-06) * vel * delta_mrt2 * pa2 + \
                (-5.47446896e-07) * dbt * vel * delta_mrt2 * pa2 + \
                (-3.59937910e-08) * vel2 * delta_mrt2 * pa2 + \
                (-4.36497725e-06) * delta_mrt3 * pa2 + \
                (1.68737969e-07) * dbt * delta_mrt3 * pa2 + \
                (2.67489271e-08) * vel * delta_mrt3 * pa2 + \
                (3.23926897e-09) * delta_mrt4 * pa2 + \
                (-3.53874123e-02) * pa3 + \
                (-2.21201190e-01) * dbt * pa3 + \
                (1.55126038e-02) * dbt2 * pa3 + \
                (-2.63917279e-04) * dbt3 * pa3 + \
                (4.53433455e-02) * vel * pa3 + \
                (-4.32943862e-03) * dbt * vel * pa3 + \
                (1.45389826e-04) * dbt2 * vel * pa3 + \
                (2.17508610e-04) * vel2 * pa3 + \
                (-6.66724702e-05) * dbt * vel2 * pa3 + \
                (3.33217140e-05) * vel3 * pa3 + \
                (-2.26921615e-03) * delta_mrt * pa3 + \
                (3.80261982e-04) * dbt * delta_mrt * pa3 + \
                (-5.45314314e-09) * dbt2 * delta_mrt * pa3 + \
                (-7.96355448e-04) * vel * delta_mrt * pa3 + \
                (2.53458034e-05) * dbt * vel * delta_mrt * pa3 + \
                (-6.31223658e-06) * vel2 * delta_mrt * pa3 + \
                (3.02122035e-04) * delta_mrt2 * pa3 + \
                (-4.77403547e-06) * dbt * delta_mrt2 * pa3 + \
                (1.73825715e-06) * vel * delta_mrt2 * pa3 + \
                (-4.09087898e-07) * delta_mrt3 * pa3 + \
                (6.14155345e-01) * pa4 + \
                (-6.16755931e-02) * dbt * pa4 + \
                (1.33374846e-03) * dbt2 * pa4 + \
                (3.55375387e-03) * vel * pa4 + \
                (-5.13027851e-04) * dbt * vel * pa4 + \
                (1.02449757e-04) * vel2 * pa4 + \
                (-1.48526421e-03) * delta_mrt * pa4 + \
                (-4.11469183e-05) * dbt * delta_mrt * pa4 + \
                (-6.80434415e-06) * vel * delta_mrt * pa4 + \
                (-9.77675906e-06) * delta_mrt2 * pa4 + \
                (8.82773108e-02) * pa5 + \
                (-3.01859306e-03) * dbt * pa5 + \
                (1.04452989e-03) * vel * pa5 + \
                (2.47090539e-04) * delta_mrt * pa5 + \
                (1.48348065e-03) * pa5 * pa
    return UTCI_approx

"""
Output Routines
"""

def writeScalarCFX(path, pts, scalar, scalar_name="Irrad [W/m^2]"):
    """Write a CFX input file for a single scalar

    Parameters
    ----------
    path : str
        Relative or absolute path of file to write.
    pts : array of float
        x, y, z point coordinates in shape [number of points, 3]
    scalar : list or array
        Scalar value at each point. 1d array with shape [number of 
        points]
    """

    cfx_header = """[Name]
Points
[Spatial Fields]
x,y,z
[Data]
"""

    column_headers = "x [m], y [m], z [m], " + scalar_name + '\n'

    with open(path, "w") as f:

        f.write(cfx_header)
        f.write(column_headers)

        for i in range(pts.shape[0]):

            row_text = str(pts[i, 0]) + ", " + \
                        str(pts[i, 1]) + ", " + \
                        str(pts[i, 2]) + ", " + \
                        str(scalar[i]) + "\n"

            f.write(row_text)

class VTK:
    def __init__(self, path, name=None, points=None, num_points=0, vindex=None, num_cells=0, areas=None, area=0, binary=False, store=False):
        """
        Initialize a new VTK file.

        Parameters
        ----------
        path : string
            Path to new VTK file, which is created by calling this constructor.
        name : string, optional
            Name of VTK file. The default is None.
        points : ndarray, optional
            n x 3 matrix of vertex (x, y, z) coordinates. The default is None.
        num_points : int, optional
            Number of points in the VTK file. The default is 0.
        vindex : ndarray, optional
            List of integers describing the number of vertices per face and
            the inices of the points for each face. The default is None.
        num_cells : int, optional
            Number of faces in the VTK file. The default is 0.
        areas : ndarray, optional
            List of areas of each face. The default is None.
        area : TYPE, optional
            Total area of all faces. The default is 0.
        binary : boolean, optional
            Flag to indicate saved VTK should be binary. The default is False.
        store : boolean, optional
            Flag to indicate that saved results should be stored internally.
            The default is False.

        Returns
        -------
        None.

        """
        self.path = path
        self.name = name
        self.points = points
        self.npoints = points.shape[0] if points else num_points
        self.vindex = vindex                
        if self.vindex:
            self.nfaces = 0
            j = 0
            for i in range(len(self.vindex)):
                self.nfaces += self.vindex[j]
                j += 1 + self.vindex[j]
        else:
            self.nfaces = num_cells
        self.areas = areas
        self.area = areas.sum() if areas else area
        self.binary = binary
        self.store = store
        self.storage = None
        self.variables = None
        self.cell = None

    def readGeometry(self):
        """
        Read the geometry from an existing VTK file.

        Returns
        -------
        None.

        """
        self.npoints = 0
        self.nfaces = 0
        with open(self.path, 'r') as reader:
            # Read the dataset name
            line = reader.readline()
            while line:
                if line.startswith('#'):
                    line = reader.readline()
                    continue
                self.name = line
                break

            # Read the dataset name
            line = reader.readline()
            while line:
                if line.upper().startswith('ASCII'):
                    self.binary = False
                    break
                if line.upper().startswith('BINARY'):
                    self.binary = True
                    break
                line = reader.readline()

            # Read the number of points
            line = reader.readline()
            while line:
                if line.upper().startswith('POINTS'):
                    self.npoints = int(line.split()[1])
                    break
                line = reader.readline()

            # Read each point
            separator = '' if self.binary else ' '
            self.points = np.fromfile(reader, dtype=np.float32, count=3*self.npoints, sep=separator).reshape(self.npoints, 3)

            # Read number of polygons
            line = reader.readline()
            while line:
                if line.upper().startswith('POLYGONS'):
                    self.nfaces = int(line.split()[1])
                    nfacedata = int(line.split()[2])
                    break
                line = reader.readline()

            # Read faces
            self.vindex = np.fromfile(reader, dtype=np.int32, count=nfacedata, sep=separator)

            # Swap endianness #TODO not sure if this is necessary
            if self.binary:
                self.points.byteswap(inplace=True)
                self.vindex.byteswap(inplace=True)

    def calculateAreas(self, df_select=None):
        """
        Calculate and store the areas of each face and the total area of the
        geometry in this VTK object.

        Parameters
        ----------
        df_select : dataframe, optional
            Dataframe containing 'X', 'Y', 'Z', 'Vertex Index', and 'Cell Index'
            columns, so that the closest vertex index and face index can be
            recorded for each point listed in the dataframe. The default is None.

        Returns
        -------
        float
            Total area of the geometry in this VTK.

        """
        self.areas = np.empty(self.nfaces)

        # Calculate the polygon areas
        j = 0
        for i in range(self.nfaces):
            vindex = self.vindex[j + 1:j + 1 + self.vindex[j]]
            self.areas[i] = area(self.points[vindex[0]], self.points[vindex[1]], self.points[vindex[2]])
            if self.vindex[j] == 4:
                self.areas[i] += area(self.points[vindex[0]], self.points[vindex[2]], self.points[vindex[3]])
            j += 1 + self.vindex[j]
        self.area = self.areas.sum()

        # Find nearest points and set indices in df_select in place
        if df_select is not None and not df_select.empty:
            findClosestPoints(df_select, self.points, 'Vertex Index')
            findClosestPoints(df_select, self.faceCenters(), 'Cell Index')

        return self.area

    def faceCenters(self):
        """
        Calculate the barycenter of each face

        Returns
        -------
        centers : ndarray
            Face centers as (x,y,z) coordinates.

        """
        centers = np.empty((self.nfaces, 3), dtype=np.float32)
        j = 0
        for i in range(self.nfaces):
            vindex = self.vindex[j + 1:j + 1 + self.vindex[j]]
            centers[i] = barycenter(self.points[vindex])
            j += 1 + self.vindex[j]
        return centers

    def writeGeometry(self):
        """
        Write the geometry to the VTK file.

        Returns
        -------
        None.

        """
        # Make directory if it does not exist
        makeDir(os.path.dirname(self.path))
        with open(self.path, 'w', newline='') as writer:
            # Write header and points
            writer.write('# vtk DataFile Version 2.0\r\n%s\r\n%s\r\nDATASET POLYDATA\r\nPOINTS %d %s\r\n' %
                         (self.name if self.name else self.path, 'BINARY' if self.binary else 'ASCII', self.npoints, 'float' if self.points.itemsize == 4 else 'double'))
            if self.binary:
                self.points.byteswap().tofile(writer)
                writer.write('\r\n')
            else:
                for i in range(self.npoints):
                    self.points[i].tofile(writer, sep=' ', format='%.6f')
                    writer.write('\r\n')

            # Write vertex indices
            writer.write('POLYGONS %d %d\r\n' % (self.nfaces, self.vindex.size))
            if self.binary:
                self.vindex.byteswap().tofile(writer)
                writer.write('\r\n')
            else:
                j = 0
                for i in range(self.nfaces):
                    self.vindex[j:j + 1 + self.vindex[j]].tofile(writer, sep=' ')
                    writer.write('\r\n')
                    j += 1 + self.vindex[j]
    
    def copyGeometry(self, template):
        """
        Create a new VTK file by copying geometry from a template file

        Parameters
        ----------
        template : VTK
            VTK file path to copy geometry from.

        Returns
        -------
        None.

        """
        self.npoints = template.npoints
        self.nfaces = template.nfaces
        self.points = template.points
        self.areas = template.areas
        self.area = template.area
        self.vindex = template.vindex

        # Make directory if it does not exist
        #makeDir(os.path.dirname(self.path))

        # Copy parent file
        #copyfile(template, self.path)

    def append(self, data, name, single=True):
        """
        Write the given data to this VTK file.

        Parameters
        ----------
        data : ndarray
            Array of data to write to the VTK file.
        name : string
            Nave of VTK scalar or vector variable.
        single : boolean, optional
            Flag to convert double precision to single precision, for binary
            files only. The default is True.

        Raises
        ------
        Exception
            Length of data does not equal number of points or cells in VTK file.

        Returns
        -------
        None.

        """
        if data.shape[0] == self.nfaces:
            cell = True
        elif data.shape[0] == self.npoints:
            cell = False
        else:
            raise Exception('Size of VTK (%d vertices, %d faces) and input data (%d rows) do not match' % (self.nfaces, self.npoints, data.shape[0]))
        if data.shape[0] == data.size:
            scalar = True
        elif data.ndim == 2 and data.shape[1] == 3:
            scalar = False
        else:
            raise Exception('Input data has unhandled dimensions %s' % data.shape)

        with open(self.path, 'a', newline='') as writer:
            if self.cell != cell:
                self.cell = cell
                writer.write('%s %d\r\n' % ('CELL_DATA' if cell else 'POINT_DATA', data.size))
            writer.write('%s %s %s\r\n' % ('SCALARS' if scalar else 'VECTORS', name.replace(' ', '_'), 'float' if single or data.itemsize == 4 else 'double'))
            if scalar:
                writer.write('LOOKUP_TABLE default\r\n')
            if self.binary:
                if single and data.itemsize != 4:
                    data = data.astype(np.float32)
                data.byteswap().tofile(writer)
            else:
                data.tofile(writer, sep='\r\n', format='%.8E')
            writer.write('\r\n')
            #for value in data:
            #    writer.write('%.8E\r\n' % value)

        # Store data
        if self.store:
            if self.storage is None:
                self.storage = [ data ]
                self.variables = [ name ]
            else:
                self.storage = np.append(self.storage, [ data ], axis=0)
                self.variables = np.append(self.variables, name)

    def areaWeightedAverage(self, data):
        """
        Calculate the area-weighted average of the data corresponding to the
        faces or vertices of this dataset.

        Parameters
        ----------
        data : array_like
            Data corresponding to each face or vertex.

        Returns
        -------
        float
            Area weighted average.

        """
        if len(data) == self.nfaces:
            return np.dot(data, self.areas) / self.area
        elif len(data) == self.npoints:
            j = 0
            faceData = np.empty_like(data)
            for i in range(self.nfaces):
                vindex = self.vindex[j + 1:j + 1 + self.vindex[j]]
                faceData[i] = data[vindex].mean()
                j += 1 + self.vindex[j]
            return np.dot(faceData, self.areas) / self.area
        else:
            raise Exception('Size of VTK (%d vertices, %d faces) and input data (%d) do not match' % (self.nfaces, self.npoints, len(data)))

"""
User Interface Routines
"""

class ProgressBar:
    def __init__(self, steps, bar_length=20):
        """
        Create a progress bar to track completion of multi-step operation

        Parameters
        ----------
        steps : int
            Number of steps to track.
        bar_length : int, optional
            Number of characters to draw in progress bar. The default is 20.

        Returns
        -------
        None.

        """
        self.steps = steps
        self.current = 0
        self.bar_length = bar_length
        self.draw()

    def increment(self, steps=1):
        """
        Increment and draw the progress bar

        Parameters
        ----------
        steps : int, optional
            Number of steps to advance. The default is 1.

        Returns
        -------
        None.

        """
        self.current += steps
        self.draw()

        if self.current >= self.steps:
            sys.stdout.write('\n') # Done with progress bar
            self.current = 0 # Start again from zero

    def draw(self):
        """
        Draw the progress bar

        Returns
        -------
        None.

        """
        fraction = self.current / self.steps if self.steps > 0 else 1
        progress = ""
        for i in range(self.bar_length):
            if i < int(self.bar_length * fraction):
                progress += "="
            else:
                progress += " "
        sys.stdout.write("\r[ %s ] %.2f%%" % (progress, fraction * 100)) # Start with carriage return to overwrite line
        sys.stdout.flush()
