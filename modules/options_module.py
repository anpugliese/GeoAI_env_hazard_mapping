from datetime import date 

no_proc_options = {
    "pm10": {
        "columns": ['lat', 'lng', 'date', 'pm10'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["pm10"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        #"remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "humidity": {
        "columns": ['lat', 'lng', 'date', 'humidity'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["humidity"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "global_radiation": {
        "columns": ['lat', 'lng', 'date', 'global_radiation'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["global_radiation"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "winds": {
        "columns": ['lat', 'lng', 'date', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        #"remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "temperature": {
        "columns": ['lat', 'lng', 'date', 'temperature'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["temperature"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "precipitation": {
        "columns": ['lat', 'lng', 'date', 'precipitation'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["precipitation"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "sum",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "IDW"
    }
}

proc_options = {
    "pm10": {
        "columns": ['lat', 'lng', 'date', 'value'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["value"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        #"remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "humidity": {
        "columns": ['lat', 'lng', 'date', 'value'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["value"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "global_radiation": {
        "columns": ['lat', 'lng', 'date', 'value'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": ["value"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "winds": {
        "columns": ['lat', 'lng', 'date'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d",
        "value_columns": [],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        #"remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "temperature": {
        "columns": ['lat', 'lng', 'date', 'value'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "value_columns": ["value"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "mean",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },
    
    "precipitation": {
        "columns": ['lat', 'lng', 'date', 'value'],
        "type": "timeseries",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "value_columns": ["value"],
        "start_date": date(2016,1,1),
        "end_date": date(2021,12,31),
        "frequency": "day",
        "aggregation_function": "sum",
        "remove_outliers": True,
        #"remove_outliers_window": 30,
        "interpolation": "NN"
    },

    "densita_popolazione": {
        "type": "odc",
        "dataset_epsg": "epsg:32632",
        "interpolation": "NN"
    },

    "water_distance": {
        "type": "odc",
        "dataset_epsg": "epsg:32632",
        "interpolation": "NN"
    }
}