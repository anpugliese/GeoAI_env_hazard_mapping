# Processing pipeline module

#Defining libraries
import math
from datetime import date, timedelta
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box
from scipy.interpolate import griddata, interpn
import datacube
from copy import deepcopy
import statsmodels.api as sm

from scipy.interpolate import Rbf
from scipy.interpolate import RBFInterpolator
import pykrige
from sklearn.preprocessing import OneHotEncoder


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from modules import interpolation_module as interp
from modules import processing_utils as utils


class HarmoniaProcessor:
    #global class attributes
    date_format = '%Y-%m-%d'
    
    FREQ_HOUR = "hour"
    FREQ_DAY = "day"
    FREQ_MONTH = "month"
    FREQ_YEAR = "year"
    
    AGG_AVG = "mean"
    AGG_SUM = "sum"
    AGG_MEDIAN = "median"
    AGG_COUNT = "count"

    """
    options object:
    options: {
        "columns": [str],
        "type": "odc" | "timeseries",
        "value_columns": str | [str] | None,
        "date_format": str | None,
        "date_column": str | None,
        "start_date": date | None,
        "end_date": date | None,
        "nodata": any,
        "frequency": "hour" | "day" | "month" | "year",
        "aggregation_function": "mean" | "sum" | "count" | function,
        "remove_outliers": boolean,
        "remove_outliers_window": None | int
        "interpolation": "NN" 
    }
    """

    #constructors
    def __init__(self):
        #init instance variables
        self.dataset_data = {}
        self.dataset_names = []
        self.processor_data = None
        self.processor_data_value_columns = []
        self.dc = None
    
    #methods
    #dataset methods
    def add_odc_dataset(self, product_name, options, config="/home/user/ODC_harmonia/datacube.conf", app_name="my_app"):
        self.dc = datacube.Datacube(app = app_name, config = config)
        if product_name not in self.dataset_names:
            print(f"adding {product_name}")
            try:
                datasets = self.dc.find_datasets(product=product_name)
            except Exception as ex:
                datasets = None

            if datasets is not None:
                #dataset found in ODC
                self.dataset_names.append(product_name)
                self.dataset_data[product_name] = {}
                self.dataset_data[product_name]['df'] = None
                self.dataset_data[product_name]['options'] = deepcopy(options)
                print(f"Added {product_name}")
            else:
                print(f"Error adding {product_name}")
        else:
            print(f"Dataset with name {product_name} already exists")
            

    def add_winds_dataset(self, name, wind_velocity_df, wind_direction_df, wind_velocity_value_column, wind_direction_value_column, wind_options):
        print(f"Building wind sectors")
        #fix columns and drop unnecessary
        options = deepcopy(wind_options)
        wind_velocity_columns = options['columns'] + ['wind_velocity']
        wind_velocity_merge = wind_velocity_df.copy()
        wind_velocity_merge = wind_velocity_merge.rename(columns = {wind_velocity_value_column: 'wind_velocity'})
        wind_velocity_merge = wind_velocity_merge[wind_velocity_columns]
        
        wind_direction_columns = options['columns'] + ['wind_direction']
        wind_direction_merge = wind_direction_df.copy()
        wind_direction_merge = wind_direction_merge.rename(columns = {wind_direction_value_column: 'wind_direction'})
        wind_direction_merge = wind_direction_merge[wind_direction_columns]

        if 'remove_outliers' in options:
            remove_outliers = options['remove_outliers']
            if remove_outliers:
                wind_velocity_merge = self.__filter_outliers(wind_velocity_merge, 'wind_velocity')
                wind_direction_merge = self.__filter_outliers(wind_direction_merge, 'wind_direction')
                         
        
        #filter dates
        #date format if specified in the dataset options
        if 'date_format' in options:
            self.date_format = options['date_format']

        #convert the date column to datetime type
        if 'date_column' in options:
            wind_velocity_merge = wind_velocity_merge.rename(columns={options['date_column']: "date"})
            wind_direction_merge = wind_direction_merge.rename(columns={options['date_column']: "date"})
        wind_velocity_merge['date'] = pd.to_datetime(wind_velocity_merge['date'], format=self.date_format)
        wind_direction_merge['date'] = pd.to_datetime(wind_direction_merge['date'], format=self.date_format)
        
        if 'start_date' in options and 'end_date' in options: 
            wind_velocity_merge
            wind_velocity_merge = wind_velocity_merge.loc[
                (wind_velocity_merge['date'].dt.date >= options['start_date']) &
                (wind_velocity_merge['date'].dt.date <= options['end_date'])
            ]
            
            wind_direction_merge = wind_direction_merge.loc[
                (wind_direction_merge['date'].dt.date >= options['start_date']) &
                (wind_direction_merge['date'].dt.date <= options['end_date'])
            ]
        
        #round latitude and longitude to 6 decimal places -> cm level accuracy is enough
        wind_velocity_merge = wind_velocity_merge.round({'lat': 6, 'lng': 6})
        wind_direction_merge = wind_direction_merge.round({'lat': 6, 'lng': 6})
        #remove nodata for the velocity dataset
        #remove nodata
        if 'nodata' in options:
            nodata = options['nodata']
            wind_velocity_merge = wind_velocity_merge.loc[wind_velocity_merge['wind_velocity'] != nodata]
            wind_direction_merge = wind_direction_merge.loc[wind_direction_merge['wind_direction'] != nodata]
            
        wind_vel_group_cols = list(wind_velocity_merge.columns)
        wind_vel_group_cols.remove('wind_velocity')
        
        wind_dir_group_cols = list(wind_direction_merge.columns)
        wind_dir_group_cols.remove('wind_direction')

        #use the mean for the velocities
        wind_velocity_merge = wind_velocity_merge.groupby(wind_vel_group_cols).mean(numeric_only=True)
        wind_velocity_merge = wind_velocity_merge.reset_index()
        #use a mean of degrees for the directions
        wind_direction_merge = wind_direction_merge.groupby(wind_dir_group_cols)
        wind_direction_merge = wind_direction_merge.agg(utils.degree_average)
        wind_direction_merge = wind_direction_merge.reset_index()
        
        wind_velocity_merge = wind_velocity_merge.round({'lat': 6, 'lng': 6})
        wind_direction_merge = wind_direction_merge.round({'lat': 6, 'lng': 6})
        #merge wind direction and velocity in a single DF        
        wind_vel_dir = pd.merge(
            wind_velocity_merge[
                wind_velocity_columns
            ], 
            wind_direction_merge[
                wind_direction_columns
            ], 
            how='inner', 
            on=['date', 'lat', 'lng']
        )
        wind_vel_dir = wind_vel_dir.sort_values(by='date')
        
        #remove outliers
        window = 24 #24 hours
        wind_vel_dir = self.__filter_outliers(
            wind_vel_dir, 
            'wind_velocity', 
            #window=window
        )

        #determine the wind sector for each record
        wind_vel_dir['wind_sector'] = wind_vel_dir.apply(
            lambda row: utils.wind_sectors(row['wind_direction']), 
            axis=1
        )

        #pivot the wind to decompose the wind per day in each of the directions
        wind_vel_dir = wind_vel_dir.groupby([wind_vel_dir['date'].dt.date,'lat','lng','wind_sector']).mean(numeric_only=True)
        wind_vel_dir = wind_vel_dir.reset_index()
        
        wind_vel_dir_columns = options['columns'].copy() + ['wind_sector', 'wind_velocity']
        pivot_columns = options['columns'].copy()
        new_wind = wind_vel_dir[wind_vel_dir_columns]
        pivoted_wind = new_wind.pivot(index=pivot_columns, columns='wind_sector', values=['wind_velocity']).reset_index()
        pivoted_wind.columns = options['columns'].copy() + ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        pivoted_wind = pivoted_wind.fillna(0)
        #round again the latitude and longitude to avoid duplicates due to dobule-precision issues
        wind_vel_dir = wind_vel_dir.round({'lat': 6, 'lng': 6})
        wind_stations = wind_vel_dir[['lat', 'lng']].drop_duplicates()
        #merge the station location with the pivoted wind
        wind = pd.merge(pivoted_wind, wind_stations, how='left', on=['lat','lng'])
        #wind
        
        print(f"Adding winds dataset")
        options['value_columns'] = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        options['columns'] = options['columns'].copy() + ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        self.add_dataset(name, wind, options)
        
    #add a dataset to the processor
    def add_dataset(self, name, df, options, process=True):
        print(name)
        #build the dfs with the options
        if name not in self.dataset_names:
            
            #if the dataset to be added does not have to be processed, just add it to the list and return
            if not process:
                self.dataset_names.append(name)
                self.dataset_data[name] = {}
                self.dataset_data[name]['df'] = df.copy()
                self.dataset_data[name]['options'] = deepcopy(options)
                return
            
            dataset = df.copy()
            dataset_options = deepcopy(options)
            #extract specific columns, if specified in the options
            if 'columns' in dataset_options:
                columns = dataset_options['columns']
                dataset = dataset[columns] 

            #date format if specified in the dataset options
            if 'date_format' in dataset_options:
                self.date_format = dataset_options['date_format']

            #convert the date column to datetime type
            dataset['date'] = pd.to_datetime(dataset['date'], format=self.date_format)
            
            #filter by start date and end date, if specified
            if 'start_date' in dataset_options and 'end_date' in dataset_options: 
                dataset = dataset.loc[
                    (dataset['date'].dt.date >= dataset_options['start_date']) &
                    (dataset['date'].dt.date <= dataset_options['end_date'])
                ]
            
            #rename the value column to the name of the dataset
            if 'value_columns' in dataset_options:
                if len(dataset_options['value_columns']) == 1:
                    #change the name of the variable if it is only one
                    old_name = dataset_options['value_columns'][0]
                    dataset = dataset.rename(columns={old_name: name})
                    dataset_options['columns'][ dataset_options['columns'].index(old_name) ] = name
                    dataset_options['value_columns'][0] = name
            
            #remove nodata
            if 'nodata' in dataset_options:
                nodata = dataset_options['nodata']
                for column in dataset_options['value_columns']:
                    dataset = dataset.loc[dataset['column'] != nodata]
            
            #convert to the desired frequency by grouping all columns except the value column and 
            # applying an aggregation function defined by the "aggregation_function" option
            if 'frequency' in dataset_options:
                frequency = dataset_options['frequency']
                freq_group_columns = list(dataset.columns)
                for column in dataset_options['value_columns']:
                    freq_group_columns.remove(column)
                    
                freq_group_columns.remove('date')
                if frequency == self.FREQ_HOUR:
                    group_columns = [pd.Grouper(key="date", freq="H")]
                    print("Aggregated hourly.")
                elif frequency == self.FREQ_DAY:
                    group_columns = [pd.Grouper(key="date", freq="D")]
                    print("Aggregated daily.")
                elif frequency == self.FREQ_MONTH:
                    group_columns = [pd.Grouper(key="date", freq="MS")]
                    print("Aggregated monthly.")
                elif frequency == self.FREQ_YEAR:
                    group_columns = [pd.Grouper(key="date", freq="YS")]
                    print("Aggregated yearly")
                else:
                    print("Unknown frequency. Skipping aggregation.")
                
                group_columns = group_columns + freq_group_columns
                dataset = dataset.groupby(group_columns)
                
                #apply custom aggregation function. If no aggregation is provided, mean is applied
                if 'aggregation_function' in dataset_options:
                    agg_function = dataset_options['aggregation_function']
                    if agg_function == self.AGG_AVG:
                        dataset = dataset.mean()
                    elif agg_function == self.AGG_SUM:
                        dataset = dataset.sum()
                    elif agg_function == self.AGG_MEDIAN:
                        dataset = dataset.median()
                    elif agg_function == self.AGG_COUNT:
                        dataset = dataset.count()
                    elif hasattr(agg_function, '__call__'): #if the aggregation function is a custom function
                        dataset = dataset.agg(agg_function)
                    else:
                        dataset = dataset.mean()
                        
                    #reset index after aggregation    
                    dataset = dataset.reset_index()
                        
                else:    
                    dataset = dataset.mean().reset_index()
                    
            if 'remove_outliers' in dataset_options:
                remove_outliers = dataset_options['remove_outliers']
                if remove_outliers:
                    for value_col in dataset_options['value_columns']:
                        if 'remove_outliers_window' in dataset_options:
                            dataset = self.__filter_outliers(dataset, value_col, window=dataset_options['remove_outliers_window'])
                        else:
                            dataset = self.__filter_outliers(dataset, value_col)       
            
            dataset = dataset.sort_values(by='date')    
            dataset = dataset.round({'lat': 6, 'lng': 6})

            self.dataset_data[name] = {
                "df": dataset.copy(),
                "options": deepcopy(dataset_options)
            }
            self.dataset_names.append(name)
            
            print(f"Dataset {name} added to processor")

        else:
            print(f"Dataset with name {name} already exists")
        
    def remove_dataset(self, name):
        if name in self.dataset_names:
            del self.dataset_data[name]
            self.dataset_names.remove(name)
            print(f"Dataset {name} removed from processor")
        else:
            print(f"Dataset with name {name} does not exists")

    def add_merged_dataset(self, df, value_columns):
        self.processor_data = df.copy()

        #only use the columns that exist in the dataframe
        included_columns = []
        df_columns = df.columns
        for varia in value_columns:
            if varia in df_columns: included_columns.append(varia)

        self.processor_data_value_columns = included_columns.copy()

    #general methods
    #Helper function for plotting DataFrames
    def show_plot(self, df, x_col, y_col, name="Unnamed Plot", additional_traces=[], scatter=False):
        mode = 'markers' if scatter else 'lines+markers' 

        if type(df) != list:
            df = [df]
            x_col = [x_col]
            y_col = [y_col]
            name = [name]
        
        fig = go.Figure()
        for i, single_df in enumerate(df):
            x = single_df[x_col[i]]
            y = single_df[y_col[i]]
            fig_name = name[i]
            fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=fig_name))
        
        if len(additional_traces) > 0:
            for trace in additional_traces:
                fig.add_trace(trace)
        fig.show()
        
    def import_df(self, path, date_format="%Y-%m-%dT%H:%M:%S", date_column="date"):
        df = pd.read_csv(path, index_col = 0)
        if date_format is not None and date_column is not None:
            df[date_column] = pd.to_datetime(df[date_column],  format=date_format)
        return df

    """
    Save one of the datasets added to the processor as csv
    """
    def save_dataset(self, name, path):
        if name in self.dataset_names:
            df_to_save = self.dataset_data[name]['df']
            df_to_save.to_csv(path)
            print(f"{name} saved to {path} as CSV")
        else:
            print(f"No dataset with name {name}")
    
    """
    Save the merged dataset as a csv file in a given path.
    """
    def save_merged_dataset(self, path):
        if self.processor_data is not None:
            df_to_save = self.processor_data
            df_to_save.to_csv(path)
            print(f"merged dataset saved to {path} as CSV")
        else:
            print(f"No merge dataset available")


    def __filter_outliers(self, input_df, value_column, window=None):
        input_df['location'] = input_df['lat'].astype(str) + input_df['lng'].astype(str)
        location_list = list(input_df.location.unique())
        filtered_df = pd.DataFrame()
        for location in location_list:
            df = input_df.loc[input_df['location'] == location].copy()
            if window is not None:
                #iterate all the df with a rolling window for assigning mean and std
                df['mean']= df[value_column].rolling(window, center=True, step=1, min_periods=1).mean()
                df['std'] = df[value_column].rolling(window, center=True, step=1, min_periods=1).std()
            else:
                #assing the mean and std column for each row based on the global data
                df['mean']= df[value_column].mean()
                df['std'] = df[value_column].std()
            #filter setup
            df = df[
                (df[value_column] <= df['mean']+3*df['std']) & 
                (df[value_column] >= df['mean']-3*df['std'])
            ]
            filtered_df = pd.concat([filtered_df, df])

        #del df
        if 'mean' in list(filtered_df.columns):
            filtered_df = filtered_df.drop(["mean"], axis=1)
        if 'std' in list(filtered_df.columns):
            filtered_df = filtered_df.drop(["std"], axis=1)
        if 'location' in list(filtered_df.columns):
            filtered_df = filtered_df.drop(["location"], axis=1)
            
        filtered_df = filtered_df.sort_values(by='date')
        return filtered_df

    #individual pipelines
    def merge_datasets(self, interpolate=False, subset=None, locations=None):
        datasets_to_merge = []        
        if subset is not None and type(subset) == list:
            datasets_to_merge = subset
        else:
            datasets_to_merge = self.dataset_names.copy()
            
        if len(datasets_to_merge) > 0:
            merged_df = None
            self.processor_data_value_columns = []
            
            #list of ODC datasets
            odc_datasets = list(filter(
                lambda x: (self.dataset_data[x]['options']['type'] == 'odc'), 
                datasets_to_merge
            ))
            #List of timeseries datasets
            timeseries_datasets = list(filter(
                lambda x: (self.dataset_data[x]['options']['type'] == 'timeseries'), 
                datasets_to_merge
            ))
            
            for df_name in timeseries_datasets:
                #timeseries datasets to be merged
                df_value_columns = self.dataset_data[df_name]['options']['value_columns']
                df_to_merge = self.dataset_data[df_name]['df']

                if merged_df is None:
                    merged_df = df_to_merge.copy()
                    self.processor_data_value_columns = self.processor_data_value_columns + df_value_columns
                else:
                    df_to_merge_columns = list(df_to_merge.columns)
                    merge_columns = ['lat', 'lng']
                    if 'date' in df_to_merge_columns:
                        merge_columns.append('date')

                    self.processor_data_value_columns = self.processor_data_value_columns + df_value_columns
                    print(f"merging {df_name}")
                    merged_df = pd.merge(
                        merged_df, 
                        df_to_merge, 
                        how='outer', 
                        on=merge_columns
                    )
            
            #check if the merged dataset exists, so we can have the sampling locations
            if merged_df is None:
                print("Cannot merge ODC datasets only. It is necesary to have stations from timeseries data.")
                return
            
            merged_df = merged_df.round({'lat': 6, 'lng': 6})
            
            if locations is not None:
                print("processing for specified locations")
                #if custom locations are provided, interpolate the meteo values to the locations and then sample
                # the odc datasets in those locations
                loc_interp_columns = []
                loc_interp_methods = []
                for value_column in timeseries_datasets:
                    loc_interp_columns = loc_interp_columns + self.dataset_data[value_column]['options']['value_columns']
                    for counter in self.dataset_data[value_column]['options']['value_columns']:
                        loc_interp_methods = loc_interp_methods + [ self.dataset_data[value_column]['options']['interpolation'] ]

                gridded_df = pd.DataFrame()
                df_dates = list(merged_df.date.unique())
                print(f"Interpolating for {len(df_dates)} dates")
                date_count = 0
                for single_date in df_dates:
                    date_count += 1
                    if date_count % 100 == 0:
                        print(f"Date #{date_count}")

                    temp_df = pd.DataFrame()
                    for i in range(len(loc_interp_columns)):
                        value_column = loc_interp_columns[i]
                        value_interpolation = loc_interp_methods[i]
                        df_to_interpolate = merged_df.loc[
                            (merged_df['date'] == single_date) &
                            (merged_df[value_column].notnull()) 
                        ]
                        
                        try:
                            interpolated, original_df = interp.interpolate_with_locations(
                                value_column, 
                                value_interpolation, 
                                locations, 
                                df_to_interpolate,
                                visual_output=False
                            )

                            #the resulting column has the name of the interpolation method
                            temp_df[value_column] = interpolated[
                                value_interpolation
                            ]
                            
                            if 'lat' not in list(temp_df.columns):
                                temp_df['lat'] = interpolated['centroids'].y
                            if 'lng' not in list(temp_df.columns):
                                temp_df['lng'] = interpolated['centroids'].x

                        except Exception as ex:
                            print(ex)
                            print(f"Error for date {single_date} for {value_column}")
                    
                    temp_df['date'] = single_date
                    gridded_df = pd.concat([gridded_df, temp_df])
                        
                #assign the dataframe with the intepolated columns to the dataframe to return
                merged_df = gridded_df.copy()
                
            all_stations = merged_df[['lat','lng']].drop_duplicates().reset_index(drop=True)
            
            #locations must be a GeoDataFrame and in UTM -> convert!
            epsg_wgs = 4326
            original_epsg = "epsg:32632"
            #if a dataset_epsg is provided, change the original epsg. By default it is epsg:32632
            if 'dataset_epsg' in self.dataset_data[df_name]['options']:
                original_epsg = self.dataset_data[df_name]['options']['dataset_epsg']
                
            coordinates_utm = gpd.GeoDataFrame(all_stations)
            coordinates_utm['geometry'] = gpd.points_from_xy(
                all_stations['lng'], 
                all_stations['lat'], 
                crs = epsg_wgs
            )
            coordinates_utm = coordinates_utm.to_crs({'init': original_epsg})
            # set the columns with the X and Y coordinates in UTM
            coordinates_utm['lat'] = all_stations['lat']
            coordinates_utm['lng'] = all_stations['lng']
            coordinates_utm['utm_x'] = coordinates_utm['geometry'].x
            coordinates_utm['utm_y'] = coordinates_utm['geometry'].y
            odc_df = None
            for df_name in odc_datasets:
                print(f"Sampling {df_name}")
                #odc datasets to be merged
                odc_product = df_name
                
                datasets = self.dc.find_datasets(product=odc_product)
                cf_data = self.dc.load(datasets=datasets)
                cf_sel = cf_data.squeeze().sel(
                    y=xr.DataArray(coordinates_utm.geometry.y.values, dims=['index']), 
                    x=xr.DataArray(coordinates_utm.geometry.x.values, dims=['index']), 
                    method='nearest'
                )
                
                cf_var_name = list(cf_data.data_vars.keys())[0]
                cf_df = cf_sel.to_dataframe()
                cf_df.rename(columns={cf_var_name:odc_product},inplace=True)
                cf_df.drop(['time','spatial_ref'],axis=1,inplace=True)

                del cf_data
                if odc_df is None:
                    odc_df = cf_df.copy()
                    odc_df = pd.concat([coordinates_utm, odc_df], axis=1)
                    odc_df = odc_df.drop(['geometry', 'utm_x', 'utm_y', 'x', 'y'], axis=1)
                else:
                    odc_df = pd.concat([odc_df, cf_df[odc_product]], axis=1)
                
                odc_df = odc_df.dropna(subset=[odc_product])
                new_value_columns = [odc_product]

                if self.dataset_data[odc_product]['options']['encode']:
                    encoding_dict_option = self.dataset_data[odc_product]['options']['encoding_mapping']
                    #encoding mapping can be a funcition.
                    # if it is an option, pass the function ref. Else, pass the get function of the dictionary mapping
                    if callable(encoding_dict_option):
                        encoding_function = encoding_dict_option
                    else:
                        encoding_function = encoding_dict_option.get

                    # get unique encoded values
                    encoding_data = odc_df[odc_product].squeeze().values
                    encoding_data = encoding_data.flatten()
                    encoding_cat = np.unique(encoding_data)
                    mapped_encoded_cat = np.vectorize(encoding_function)(encoding_cat)
                    mapped_encoded_cat = np.unique(mapped_encoded_cat).reshape(-1,1)
                    del encoding_data

                    encoder = OneHotEncoder()
                    encoder.fit(mapped_encoded_cat)

                    #replace the column with the encoded 
                    new_value_columns = encoder.get_feature_names_out([odc_product])
                    new_column = np.vectorize(encoding_function)(odc_df[odc_product])
                    
                    new_column = encoder.transform(new_column.reshape(-1,1))
                    encoded_df = pd.DataFrame(new_column.toarray(), columns=new_value_columns)
                    encoded_df = encoded_df.reset_index(drop=True)
                    odc_df = odc_df.reset_index(drop=True)
                    
                    odc_df = pd.concat([odc_df, encoded_df], axis=1)
                    odc_df = odc_df.drop(columns=[odc_product])
                
                #the list of all columns with data
                self.processor_data_value_columns = list(self.processor_data_value_columns) + list(new_value_columns)
                
            if odc_df is not None:
                print("merging ODC datasets")
                #add coordinates
                odc_df = odc_df.round({'lat': 5, 'lng': 5})
                merged_df = merged_df.round({'lat': 5, 'lng': 5})
                merged_df = pd.merge(merged_df, odc_df, how='outer', on=['lat', 'lng'])
            
            print("datasets merged!")
            
            if interpolate:
                print("Starting interpolation of missing values")
                #Interpolating missing data                
                for interpolation_column in self.processor_data_value_columns:
                    for dataset_name in self.dataset_names:
                        
                        if self.dataset_data[dataset_name]['options']['type'] == 'odc':
                            value_columns_dataset = [dataset_name]
                        else:
                            value_columns_dataset = self.dataset_data[dataset_name]['options']['value_columns']
                            
                        if interpolation_column in value_columns_dataset:
                            interpolation_method = self.dataset_data[dataset_name]['options']['interpolation']
                            merged_df = merged_df.sort_values(by='date')
                            initial_date = merged_df.iloc[0].date
                            end_date = merged_df.iloc[-1].date
                            date_range = [initial_date, end_date]
                            
                            print(f'------------------------- Interpolating {interpolation_column} -------------------------')
                            if interpolation_method == 'NN':
                                interp.single_NN(merged_df, interpolation_column, date_range)
                                
                            elif interpolation_method == 'IDW':
                                interp.single_IDW_new(merged_df, interpolation_column, date_range)
                            else: #default use NN
                                interp.single_NN(merged_df, interpolation_column, date_range)
                print("Finished interpolation of missing values")
                
            merged_df = merged_df.dropna()
            self.processor_data = merged_df.copy()
               
        else:
            print("No datasets to merge")
            
            
    def generate_training_data(
        self, 
        pollutant_column, 
        pollutant_threshold, 
        predictor_column_name='exc',
        predictor_classes=[0,1],
        date_range=None, 
        train_percentage=0.8,
        test_percentage=0.1,
        validation_percentage=0.1,
        random_partition=False,
        balanced=False
    ):
        df = self.processor_data.copy()
        #calculate the exceeded column
        df[predictor_column_name] = np.where(df[pollutant_column].gt(pollutant_threshold), 1, 0)

        #if a date range is provided
        if date_range is not None:
            df = df.loc[
                (df['date'].dt.date >= date_range[0]) &
                (df['date'].dt.date <= date_range[1])
            ]
    
        #change order of columns to put the pollutant exceedence first
        #df = df.drop([pollutant_column], axis=1)
        df_columns = list(df.columns)
        df_columns.remove(predictor_column_name)
        df_columns = [predictor_column_name] + df_columns
        df = df[df_columns]
        
        #if random partition is desired
        if random_partition:
            df = df.sample(frac=1)
        df = df.reset_index(drop=True)
            
        #partition the dataset in training, testing, and validation
        if balanced: 
            training = pd.DataFrame()
            testing = pd.DataFrame()
            validation = pd.DataFrame()
            
            for class_label in predictor_classes:
                class_df = df.loc[df[predictor_column_name] == class_label]
                
                n = len(class_df)
                train_n = int(n * train_percentage)
                test_n = int(n * test_percentage)
                validation_n = int(n * validation_percentage)
                #if the partitions do not cover all data, assign any overflow to the training data
                if (train_n + test_n + validation_n) != n:
                    train_n += n - (train_n + test_n + validation_n)
                #partition the data according to the percentages
                part_1 = train_n
                part_2 = train_n + test_n
                part_3 = train_n + test_n + validation_n
                
                training = pd.concat(
                    [training, class_df.iloc[0 : part_1]]
                )
                testing = pd.concat(
                    [testing, class_df.iloc[part_1 : part_2]]
                )
                validation = pd.concat(
                    [validation, class_df.iloc[part_2 : part_3]]
                )  
            
        else:
            n = len(df)
            train_n = int(n * train_percentage)
            test_n = int(n * test_percentage)
            validation_n = int(n * validation_percentage)
            #if the partitions do not cover all data, assign any overflow to the training data
            if (train_n + test_n + validation_n) != n:
                train_n += n - (train_n + test_n + validation_n)
            
            #partition the data according to the percentages
            part_1 = train_n
            part_2 = train_n + test_n
            part_3 = train_n + test_n + validation_n
            
            training = df.iloc[0 : part_1]
            testing = df.iloc[part_1 : part_2]
            validation = df.iloc[part_2 : part_3]
        
        return training, testing, validation
    
    
    def generate_prediction_data(self, options):
        """
        options: {
            "variables": [],
            "sampling": "grid" | "locations",
            "grid_options": { #only if the sampling is grid
                "shapefile_path": str,
                "xdelta": int, #meter separation for grid in x coordinates
                "ydelta": int, #meter separation for grid in y coordinates
                "interpolation_method": "NN" | "IDW",
            },
            "frequency": "day" | "month" | "year",
            "aggregation": {"var1": "aggregation1", etc}, #
            "date_range": [date, date],
            "day": date, #conditional only when frequency is day
            "month": int, #conditional only when frequency is month
            "year_range": [], #conditional only when frequency is year or month
        }
        """
        df = self.processor_data.copy()
        variable_columns = options['variables'].copy()
        df_columns = df.columns

        #only use the columns that exist in the dataframe
        included_columns = []
        for varia in variable_columns:
            if varia in df_columns: included_columns.append(varia)

        keep_variables = ['lat','lng','date'] + included_columns
        df = df[keep_variables]
        if 'date_range' in options:
            date_range = options['date_range']
            df = df.loc[
                (df['date'].dt.date >= date_range[0]) &
                (df['date'].dt.date <= date_range[1])
            ]
            
        if 'sampling' not in options:
            print("Error: sampling option must be set.")
            return
        
        if 'frequency' not in options:
            print("Error: frequency option must be set.")
            return
        
        if 'variables' not in options:
            print("Error: variables option must be set.")
            return
        
        if 'aggregation' not in options:
            print("Error: aggregation option must be set.")
            return
        
        frequency = options["frequency"]            
        
        if frequency == self.FREQ_DAY:
            if 'day' not in options:
                print("'day' option must be specified with a frequency of 'day'")
                return
            df = df.loc[df['date'] == options['day']]
            return df.copy()
            
        else:
            aggregation = deepcopy(options['aggregation'])
            for varia in variable_columns:
                if varia not in df_columns: del aggregation[varia]

            df = df.assign(year=df['date'].dt.year, month=df['date'].dt.month)
            df['t'] = (df['year'].astype(str) + '-' + df['month'].astype(str))
            if frequency == self.FREQ_MONTH:
                df = df.loc[
                    (df['month'] == options['month']) &
                    (df['year'] >= options['year_range'][0]) &
                    (df['year'] <= options['year_range'][1])
                ]
                df = df.groupby(['month','lat','lng'])
                
            if frequency == self.FREQ_YEAR:
                df = df.loc[
                    (df['year'] >= options['year_range'][0]) &
                    (df['year'] <= options['year_range'][1])
                ]
                df = df.groupby(['year','lat','lng'])
                
            df = df.agg(aggregation).reset_index()
                
            if options['sampling'] == 'grid':
                if 'grid_options' not in options:
                    print("Error: grid options not specified. Returning station locations instead.")
                else:
                    gridded_df = pd.DataFrame()
                    for value_column in options['variables']:
                        print(f"Interpolating to grid for variable {value_column}")
                        interpolated, original_df = interp.interpolate(
                            value_column, 
                            options['grid_options']['interpolation_method'], 
                            options['grid_options']['shapefile_path'], 
                            df, 
                            xdelta=options['grid_options']['xdelta'], 
                            ydelta=options['grid_options']['ydelta'],
                            visual_output=False
                        )
                        gridded_df[value_column] = interpolated[options['grid_options']['interpolation_method']]
                        if 'lat' not in list(gridded_df.columns):
                            gridded_df['lat'] = interpolated['centroids'].y
                        if 'lng' not in list(gridded_df.columns):
                            gridded_df['lng'] = interpolated['centroids'].x
                    
                    #assign the dataframe with the intepolated columns to the dataframe to return
                    df = gridded_df.copy()
            
            final_columns = list(df.columns)
            if 'date' in final_columns: df = df.drop(['date'], axis=1)
            if 'month' in final_columns: df = df.drop(['month'], axis=1)
            if 'year' in final_columns: df = df.drop(['year'], axis=1)
            if 'geometry' in final_columns: df = df.drop(['geometry'], axis=1)
            return df.copy()
            
    def train_model(self, training, testing, validation):
        pass
    
    def predict(self, prediction_data):
        pass
    #multi-step pipelines
