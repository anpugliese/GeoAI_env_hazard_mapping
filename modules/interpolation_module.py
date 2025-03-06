#Defining libraries
from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from scipy.interpolate import griddata, interpn

from scipy.interpolate import Rbf
from scipy.interpolate import RBFInterpolator
import pykrige
import rioxarray


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

#save a data frame with columns [y,x,value] to a raster using rioxarray
def save_as_raster(df, filepath, crs="4326"):
    df_xarray = df.set_index(['y', 'x']).value.to_xarray()
    df_xarray.rio.write_crs(crs, inplace=True)
    df_xarray.rio.to_raster(filepath)
    return df_xarray

#Parameters
#Start_date: start date to filter from ARPA lombardia data
#End_date: end date to filter from ARPA Lombardia data
#data: add a custom DF containg lat, lng, station_id, and values
#Variable: the dataframe column to be used for interpolation - from ARPA station names (Precipitazione, Umidità Relativa, Radiazione Globale, Temperatura)
#province: the province to filter the data of ARPA Lombardia (e.g., MI for Milan)
#method: interpolation methods: Nearest Neighbour (NN), Inverse Distance Weighted (IDW), and Kriging (K)
#You can set the variable to NN, IDW or K
#Shapefile: area of interest .shp (path)
#xdelta: x size of the grid in meters, by default 1000 meters (1km)
#ydelta: y size of the grid in meters, by default 1000 meters (1km)
def interpolate_with_locations(value, method, grid, data, visual_output=True):        
    gdf_grid = grid
    df = data
    if method == 'NN':
        if visual_output: print('Creating an interpolation with the Nearest Neighbour method')
        NN(gdf_grid, df, value)
    elif method == 'IDW':
        if visual_output: print('Creating an interpolation with the Inverse Distance Weighted method')
        IDW(gdf_grid, df, value)
    elif method == 'IDW_new':
        if visual_output: print('Creating an interpolation with the Inverse Distance Weighted method (library update)')
        IDW_new(gdf_grid, df, value)
    elif method == 'K':
        if visual_output: print('Creating an interpolation with the Kriging method')
        K(gdf_grid, df, value)

        
            
    #print('Storing the results...')    
    return gdf_grid, df

#Parameters
#Start_date: start date to filter from ARPA lombardia data
#End_date: end date to filter from ARPA Lombardia data
#data: add a custom DF containg lat, lng, station_id, and values
#Variable: the dataframe column to be used for interpolation - from ARPA station names (Precipitazione, Umidità Relativa, Radiazione Globale, Temperatura)
#province: the province to filter the data of ARPA Lombardia (e.g., MI for Milan)
#method: interpolation methods: Nearest Neighbour (NN), Inverse Distance Weighted (IDW), and Kriging (K)
#You can set the variable to NN, IDW or K
#Shapefile: area of interest .shp (path)
#xdelta: x size of the grid in meters, by default 1000 meters (1km)
#ydelta: y size of the grid in meters, by default 1000 meters (1km)
def interpolate(value, method, shapefile, data, xdelta=1000, ydelta=1000, visual_output=True, epsg_utm=None, plot_min=None, plot_max=None):        
        
    gdf_grid, df, bnd = create_grid(shapefile, data, xdelta, ydelta, epsg_utm=epsg_utm)
    if method == 'NN':
        if visual_output: print('Creating an interpolation with the Nearest Neighbour method')
        NN(gdf_grid, df, value)
    elif method == 'IDW':
        if visual_output: print('Creating an interpolation with the Inverse Distance Weighted method')
        IDW(gdf_grid, df, value)
    elif method == 'IDW_new':
        if visual_output: print('Creating an interpolation with the Inverse Distance Weighted method (library update)')
        IDW_new(gdf_grid, df, value)
    elif method == 'K':
        if visual_output: print('Creating an interpolation with the Kriging method')
        K(gdf_grid, df, value)
    
    if visual_output:
        print('Plotting the results...')
        if method != 'K':
            plot(gdf_grid,bnd,df,method, value,method,min=plot_min, max=plot_max)
        else:
            plot_kriging(gdf_grid,bnd,df, value)
            
    #print('Storing the results...')    
    return gdf_grid, df

#This function creates a grid from a shapefile of the area of interest.
#Parameters
#Shapefile: area of interest .shp
#xdelta: x size of the grid in meters, by default 1000 meters (1km)
#ydelta: y size of the grid in meters, by default 1000 meters (1km)
def create_grid_from_shapefile(shapefile, xdelta=1000, ydelta=1000, shapefile_epsg=32632):
    # Read in data
    #print('Reading shapefile ' + shapefile)
    bnd = gpd.read_file(shapefile)
    #print(bnd.crs)
    # Define coordinate systems
    epsg_wgs = 4326 # World Geodetic System 1984

    bnd = bnd.to_crs(epsg=shapefile_epsg) #in target epsg

    # Create grid to interpolate over
    xmin, ymin, xmax, ymax = bnd.total_bounds #Represents field in terms of meters

    # Create an empty array to save the grid
    grid = np.array([])

    for x in np.arange(xmin, xmax, xdelta): #min, max step
        for y in np.arange(ymin, ymax, ydelta): #min, max step
            cell = box(x,y, x+xdelta, y+ydelta)
            grid = np.append(grid, cell)

    gdf_grid = gpd.GeoDataFrame(grid, columns=['geometry'], crs=shapefile_epsg)
    gdf_grid['centroids'] = gdf_grid['geometry'].centroid
    gdf_grid.head()
    
    # Clip cells to shapefile boundary while on shapefile epsg
    gdf_grid = gpd.clip(gdf_grid, bnd['geometry'].iloc[0])

    # Convert CRS back to Lat/Long
    gdf_grid['original_centroids'] = gdf_grid['centroids']
    gdf_grid['centroids'] = gdf_grid['centroids'].to_crs(crs=epsg_wgs)
    gdf_grid['lat'] = gdf_grid['centroids'].y
    gdf_grid['lng'] = gdf_grid['centroids'].x

    #drop unused columns
    gdf_grid = gdf_grid.drop(['geometry'], axis=1)

    gdf_grid.reset_index(inplace=True, drop=True)
    print('The grid of the shapefile')
    return gdf_grid

#This function creates a grid from a shapefile of the area of interest. Also, creates a geodataframe of the point data
#Parameters
#Shapefile: area of interest .shp
#data: pandas dataframe containing the point data (station id, lat, lng, value) - the dataframe column names should be lat, lng
#xdelta: x size of the grid in meters, by default 1000 meters (1km)
#ydelta: y size of the grid in meters, by default 1000 meters (1km)

#Returns a geodataframe of point data and a grid of the area of interest to be used for interpolation
def create_grid(shapefile, data, xdelta=1000, ydelta=1000, epsg_utm=None):
    if epsg_utm is None: epsg_utm = 32632
    # Read in data
    #print('Reading shapefile ' + shapefile)
    bnd = gpd.read_file(shapefile)
    #print(bnd.crs)

    # Convert DataFrame to GeoDataFrame
    #print('Converting data to geodataframe')
    df = gpd.GeoDataFrame(data)
    df.head()

    # Define coordinate systems
    epsg_wgs = 4326 # World Geodetic System 1984
    epsg_utm = epsg_utm # UTM Zone 32

    # Convert latitude and longitude to point data
    df['geometry'] = gpd.points_from_xy(df['lng'], 
                                        df['lat'],
                                        crs = epsg_wgs)

    df.head()

    # Reproject shapefile boundary
    bnd  = bnd.to_crs(epsg_wgs)

    # Check CRS for both datasets
    #print(bnd.crs)
    #print(df.crs)

    # Plotting of map boundary and sampled points
    #fig, ax = plt.subplots(1,1, figsize=(8,8))
    #ax.ticklabel_format(useOffset=False)
    #bnd.plot(ax=ax, facecolor='w', edgecolor='k')
    #df.plot(ax=ax, marker='x', facecolor='k')
    #plt.show()
    
    # Find and delete points outside of the field boundaries
    for k,row in df.iterrows(): #Prints row without indexed rows
        point = row['geometry']
        if not point.within(bnd['geometry'].iloc[0]):
            df.drop(k, inplace=True) #Drops the Kth row

    df.reset_index(inplace=True, drop=True)

    # Plot again after deleting the points outside of the boundary
    # Plotting of map boundary and sampled points
    #fig, ax = plt.subplots(1,1, figsize=(8,8))
    #ax.ticklabel_format(useOffset=False)
    #bnd.plot(ax=ax, facecolor='w', edgecolor='k')
    #df.plot(ax=ax, marker='x', facecolor='k')
    #plt.show()
    
    # Create grid to interpolate over
    xmin, ymin, xmax, ymax = bnd.to_crs(epsg=epsg_utm).total_bounds #Represents field in terms of meters

    # Create an empty array to save the grid
    grid = np.array([])

    for x in np.arange(xmin, xmax, xdelta): #min, max step
        for y in np.arange(ymin, ymax, ydelta): #min, max step
            cell = box(x,y, x+xdelta, y+ydelta)
            grid = np.append(grid, cell)

    gdf_grid = gpd.GeoDataFrame(grid, columns=['geometry'], crs=epsg_utm)
    gdf_grid['centroids'] = gdf_grid['geometry'].centroid
    gdf_grid.head()

    # Convert CRS back to Lat/Long
    gdf_grid['original_centroids'] = gdf_grid['centroids']
    gdf_grid['geometry'] = gdf_grid['geometry'].to_crs(crs=epsg_wgs)
    gdf_grid['centroids'] = gdf_grid['centroids'].to_crs(crs=epsg_wgs)

    # Clip cells to shapefile boundary
    gdf_grid = gpd.clip(gdf_grid, bnd['geometry'].iloc[0])
    gdf_grid.reset_index(inplace=True, drop=True)

    #Plot
    #fig, ax = plt.subplots(1,1, figsize=(8,8))
    #ax.ticklabel_format(useOffset=False)
    #bnd.plot(ax=ax, facecolor='w', edgecolor='r')
    #df.plot(ax=ax, marker='x', facecolor='k')
    #gdf_grid.plot(ax=ax, facecolor = 'None', edgecolor='k', linewidth=0.1)
    #plt.show()
    
    #print('The grid of the shapefile was created and the station points have been georeferrenced')
    return gdf_grid, df, bnd
    
    
def IDW(gdf_grid, df,value):
    # Interpolate the values of observed values to each centroid
    x = df['lng']
    y = df['lat']
    z = df[value]

    xq = gdf_grid['centroids'].x
    yq = gdf_grid['centroids'].y

    # Arrange variables in griddata input format
    points = (x, y)
    values = z
    xi = (xq, yq)

    # Define the IDW function
    rbf = Rbf(x, y, z, function='inverse')

    # Interpolate the data at the grid points
    z_interp = rbf(xq,yq)

    #Save interpolated data values into the geodataframe
    gdf_grid['IDW'] = z_interp


def IDW_new(gdf_grid, df, value):
    # Interpolate the values of observed values to each centroid
    x = df['lng']
    y = df['lat']
    z = df[value]

    xq = gdf_grid['centroids'].x
    yq = gdf_grid['centroids'].y

    # Arrange variables in griddata input format
    points = (x, y)
    values = z
    xi = (xq, yq)

    # Define the IDW function
    rbf = RBFInterpolator(np.transpose(points), values, kernel='inverse_multiquadric',epsilon=20)

    # Interpolate the data at the grid points
    grid_array = np.column_stack((xq.ravel(),yq.ravel()))
    z_interp = rbf(grid_array)

    #Save interpolated data values into the geodataframe
    gdf_grid['IDW_new'] = z_interp


def NN(gdf_grid, df, value):
    # Interpolate the values of observed values to each centroid
    x = df['lng']
    y = df['lat']
    z = df[value]

    xq = gdf_grid['centroids'].x
    yq = gdf_grid['centroids'].y

    # Arrange variables in griddata input format
    points = (x, y)
    values = z
    xi = (xq, yq)

    interpolated = griddata(points,
                        values,
                        xi,
                        method='nearest',
                        )

    #Save interpolated data values into the geodataframe
    gdf_grid['NN'] = interpolated

    


#zvalues (ndarray, shape (M, N) or (N, 1)) – Z-values of specified grid or at the specified set of points.
#If style was specified as ‘masked’, zvalues will be a numpy masked array.

#sigmasq (ndarray, shape (M, N) or (N, 1)) – Variance at specified grid points or at the specified set of points. 
#If style was specified as ‘masked’, sigmasq will be a numpy masked array.

def K(gdf_grid, df, value):
    # Interpolate the values of observed values to each centroid
    x = df['lng']
    y = df['lat']
    z = df[value]

    xq = gdf_grid['centroids'].x
    yq = gdf_grid['centroids'].y

    # Arrange variables in griddata input format
    points = (x, y)
    values = z
    xi = (xq, yq)
    
    krig = pykrige.uk.UniversalKriging(x, y, z, variogram_model ='gaussian')
    z_interp, sigma = krig.execute('points', xq, yq)
    
    #Save interpolated data values into the geodataframe
    gdf_grid['K'] = z_interp
    gdf_grid['K_sigma'] = sigma



#Interpolation methods for single points
def single_IDW_new(df_all, value, date_range):
    #start_date = date(2016,1,1)
    #end_date = date(2022,12,31)
    start_date = date_range[0]
    end_date = date_range[1]
    print(start_date, end_date)
    
    day_count = (end_date - start_date).days + 1
    for single_date in (start_date + timedelta(n) for n in range(day_count)):
        try:
            df = df_all.loc[df_all['date'] == single_date.strftime("%Y-%m-%d")]
            # Interpolate the values of observed values to each centroid
            x = df[df[value].notna()]['lng']
            y = df[df[value].notna()]['lat']
            z = df[df[value].notna()][value]

            xq = df[df[value].isnull()]['lng']
            yq = df[df[value].isnull()]['lat']

            # Arrange variables in griddata input format
            points = (x, y)
            values = z
            xi = (xq, yq)

            # Define the IDW function
            rbf = RBFInterpolator(np.transpose(points), values, kernel='inverse_quadratic',epsilon=20)

            # Interpolate the data at the grid points
            grid_array = np.column_stack((xq.ravel(),yq.ravel()))
            z_interp = rbf(grid_array)

            #Save interpolated data values into the geodataframe
            df_all.loc[(df_all[value].isnull())&(df_all['date'] == single_date.strftime("%Y-%m-%d")), value] = z_interp
        except Exception as e:
            print(f'error in {single_date}')


def single_IDW(df_all,value, date_range):
    #start_date = date(2016,1,1)
    #end_date = date(2022,12,31)
    start_date = date_range[0]
    end_date = date_range[1]
    print(start_date, end_date)

    day_count = (end_date - start_date).days + 1
    for single_date in (start_date + timedelta(n) for n in range(day_count)):
        try:
            df = df_all.loc[df_all['date'] == single_date.strftime("%Y-%m-%d")]
            # Interpolate the values of observed values to each centroid
            x = df[df[value].notna()]['lng']
            y = df[df[value].notna()]['lat']
            z = df[df[value].notna()][value]

            xq = df[df[value].isnull()]['lng']
            yq = df[df[value].isnull()]['lat']

            # Arrange variables in griddata input format
            points = (x, y)
            values = z
            xi = (xq, yq)

            # Define the IDW function
            rbf = Rbf(x, y, z, function='inverse')

            # Interpolate the data at the grid points
            z_interp = rbf(xq,yq)

            #Save interpolated data values into the geodataframe
            df_all.loc[(df_all[value].isnull())&(df_all['date'] == single_date.strftime("%Y-%m-%d")), value] = z_interp
        except Exception as e:
            print(f'error in {single_date}')


def single_NN(df_all, value, date_range):

    start_date = date_range[0]
    end_date = date_range[1]
    print(start_date, end_date)

    day_count = (end_date - start_date).days + 1
    for single_date in (start_date + timedelta(n) for n in range(day_count)):
        try:
            df = df_all.loc[df_all['date'] == single_date.strftime("%Y-%m-%d")]
            # Interpolate the values of observed values to each centroid
            x = df[df[value].notna()]['lng']
            y = df[df[value].notna()]['lat']
            z = df[df[value].notna()][value]

            xq = df[df[value].isnull()]['lng']
            yq = df[df[value].isnull()]['lat']

            points = (x, y)
            values = z
            xi = (xq, yq)

            # Define the NN function
            interpolated = griddata(points,
                values,
                xi,
                method='nearest',
            )

            #Save interpolated data values into the geodataframe
            df_all.loc[(df_all[value].isnull())&(df_all['date'] == single_date.strftime("%Y-%m-%d")), value] = interpolated
        
        except Exception as e:
            print(f'error in {single_date}')

    
def plot(gdf_grid,bnd,df,param,value,method,min=None,max=None):
    max_value = 100 if max is None else max
    min_value = 0 if min is None else min
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    ax.ticklabel_format(useOffset=False)
    bnd.plot(ax=ax, facecolor='w', edgecolor='r')
    gdf_grid.plot(ax=ax, column=param, edgecolor='k', cmap='Reds', linewidth=0.1, vmin=min_value, vmax=max_value)
    df.plot(ax=ax, column=value, edgecolor='k', marker='o', cmap='Reds', vmin=min_value, vmax=max_value, alpha=0.05)

    #Legend interpolation
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, label='Grid values')


    #Legend points
    sm_points = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm_points._A = []
    cbar = fig.colorbar(sm_points, ax=ax, orientation='vertical', shrink=0.5, label='Station values')

    plt.show()

def plot_grid(gdf_grid,bnd,param,method,min=None,max=None):
    max_value = 100#max(df[value].max(),gdf_grid[method].max())
    min_value = 0#min(df[value].min(),gdf_grid[method].min())
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    ax.ticklabel_format(useOffset=False)
    bnd.plot(ax=ax, facecolor='w', edgecolor='r')
    gdf_grid.plot(ax=ax, column=param, edgecolor='k', cmap='Reds', linewidth=0.1, vmin=min_value, vmax=max_value)

    #Legend interpolation
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, label='Grid values')

    plt.show()


def plot_kriging(gdf_grid,bnd,df,value):
    max_value = 100#max(df[value].max(),gdf_grid['K'].max())
    min_value = 0#min(df[value].min(),gdf_grid['K'].min())
    
    fig, ax = plt.subplots(1,1, figsize=(15,15))
    ax.ticklabel_format(useOffset=False)
    bnd.plot(ax=ax, facecolor='w', edgecolor='r')
    gdf_grid.plot(ax=ax, column='K', edgecolor='k', cmap='Reds', linewidth=0.1,  vmin=min_value, vmax=max_value)
    df.plot(ax=ax, column=value, edgecolor='k', marker='o', cmap='Reds', vmin=min_value, vmax=max_value)

    #Legend interpolation
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, label='Grid values')


    #Legend points
    sm_points = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=min_value, vmax=max_value))
    sm_points._A = []
    cbar = fig.colorbar(sm_points, ax=ax, orientation='vertical', shrink=0.5, label='Station values')

    plt.show()
    
    #ERROR PLOT

    fig, ax = plt.subplots(1,1, figsize=(15,15))
    ax.ticklabel_format(useOffset=False)
    bnd.plot(ax=ax, facecolor='w', edgecolor='r')
    gdf_grid.plot(ax=ax, column='K_sigma', edgecolor='k', cmap='Reds', linewidth=0.1)
    df.plot(ax=ax, column=value, edgecolor='k', marker='o', cmap='Reds')

    #Legend interpolation
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=gdf_grid['K_sigma'].min(), vmax=gdf_grid['K_sigma'].max()))
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.5, label='Sigma')


    #Legend points
    sm_points = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=df[value].min(), vmax=df[value].max()))
    sm_points._A = []
    cbar = fig.colorbar(sm_points, ax=ax, orientation='vertical', shrink=0.5, label='Station values')

    plt.show()


