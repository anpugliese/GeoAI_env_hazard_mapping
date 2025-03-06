import pickle
import xarray as xr
import numpy as np
import pandas as pd
from math import sqrt
pd.set_option('display.max_columns', None)

#----------------Utils--------------------------
import pandas as pd
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np

#----------------Interpolation--------------------------
#This is a interpolation module that uses scipy interpolation methods.
from modules import interpolation_module as interp
from shapely.geometry import box
from scipy.interpolate import griddata, interpn

from scipy.interpolate import Rbf
from scipy.interpolate import RBFInterpolator

#----------------Machine Learning--------------------------

#----------------TensorFlow--------------------------
import tensorflow as tf
#tf.compat.v1.disable_v2_behavior() 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
# fix random seed for reproducibility
tf.random.set_seed(7)

#----------------Sklearn--------------------------
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class MLProcessor:
    def __init__(self, train, test):
        self.training_data = train.copy()
        self.testing_data = test.copy()
        
        #scalers
        scaler_train = StandardScaler()
        scaler_train.fit(self.training_data.drop('exc', axis=1))
        self.scaler = scaler_train
        
        self.trained_model = None
        self.model_type = None


    """
    params:
        model_type: 'svm' | "RF" | "RF_regressor" | etc
        y_column: The column that contains the y values
        model_options: specific options of the model that is being implemented
    """
    def train_model(self, model_type, y_column, model_options={}):
        train_x = self.training_data.copy()
        test_x = self.testing_data.copy()
        
        train_y = train_x.copy()[y_column]
        train_x = train_x.drop(y_column, axis=1)
        test_y = test_x.copy()[y_column]
        test_x = test_x.drop(y_column, axis=1)
        
        self.model_type = model_type
        
        #scalers
        training_data_normalized = self.scaler.transform(train_x.copy())
        testing_data_normalized = self.scaler.transform(test_x.copy())

        if model_options['normalized']:
            training_data_for_model_x = training_data_normalized
            testing_data_for_model_x = testing_data_normalized
        else:
            training_data_for_model_x = train_x
            testing_data_for_model_x = test_x

        if model_type == 'svm':    
            random_state = model_options['random_state'] if 'random_state' in model_options else None
            probability = model_options['probability'] if 'probability' in model_options else True
            kernel = model_options['kernel'] if 'kernel' in model_options else 'rbf'
            n_jobs = model_options['n_jobs'] if 'n_jobs' in model_options else None 
            svm_params = {
                "kernel": [kernel],
                "probability": [probability], 
                "random_state": [random_state],
                "verbose": [0]
            }
            if random_state is not None: svm_params["random_state"] = [random_state]

            self.trained_model = RSCV(
                svm.SVC(),
                svm_params,
                n_iter = 3,
                n_jobs=n_jobs
            )

            self.trained_model.fit(training_data_for_model_x, train_y)
            score = self.trained_model.score(testing_data_for_model_x, test_y)
            print(f"SVM score: {score}")
            return self.trained_model
            
        
        elif model_type == 'RF' or model_type == 'rf':
            n_estimators = model_options['n_estimators'] if 'n_estimators' in model_options else 100
            random_state = model_options['random_state'] if 'random_state' in model_options else None
            n_jobs = model_options['n_jobs'] if 'n_jobs' in model_options else None 
            max_depth = model_options['max_depth'] if 'max_depth' in model_options else None 
            
            rf_params = {
                "n_estimators": [n_estimators],
                "n_jobs" :[n_jobs],
                "max_depth": [max_depth],
                "verbose": [False]
            }
            if random_state is not None: rf_params["random_state"] = [random_state]

            self.trained_model = RSCV(
                RandomForestClassifier(),
                rf_params,
                n_iter = 10,
                n_jobs=n_jobs
            )

            #Train
            self.trained_model.fit(training_data_for_model_x, train_y)
            score = self.trained_model.score(testing_data_for_model_x, test_y)
            
            print(f"RF score: {score}")
            return self.trained_model
        
        elif model_type == 'lstm':
            activation = model_options['activation'] if 'activation' in model_options else 'sigmoid'
            metrics = model_options['metrics'] if 'metrics' in model_options else ['accuracy']
            optimizer = model_options['optimizer'] if 'optimizer' in model_options else 'adam'
            loss = model_options['loss'] if 'loss' in model_options else 'binary_crossentropy'

            # create the model
            self.trained_model = Sequential()
            self.trained_model.add(LSTM(100))
            self.trained_model.add(Dense(1, activation=activation))
            self.trained_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            # reshape input to be 3D [samples, timesteps, features]
            train_X = training_data_for_model_x.reshape((training_data_for_model_x.shape[0], 1, training_data_for_model_x.shape[1]))
            test_X = testing_data_for_model_x.reshape((testing_data_for_model_x.shape[0], 1, testing_data_for_model_x.shape[1]))
            #Train the model
            self.trained_model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=3, batch_size=64)
            print(self.trained_model.summary())
            # Evaluate the model
            test_loss, test_accuracy = self.trained_model.evaluate(test_X, test_y)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
            
            return self.trained_model

        else:
            print("No model specified")
            return
        
    
    def predict_probabilities(self, predict_df, predict_options={}):
        df = predict_df.copy()
        normalized = predict_options['normalized'] if 'normalized' in predict_options else False
        if normalized:
            predict_scaler = self.scaler
            df = predict_scaler.transform(df)
            
        return self.trained_model.predict_proba(df)
    
    def predict(self, predict_df, predict_options={}):
        df = predict_df.copy()
        normalized = predict_options['normalized'] if 'normalized' in predict_options else False
        if normalized:
            predict_scaler = self.scaler
            df = predict_scaler.transform(df)
            
        if self.model_type == 'lstm':
            df = df.reshape((df.shape[0], 1, df.shape[1]))
        return self.trained_model.predict(df)
    
    def lime_predict(self,predict_array):
        flat_instance = predict_array
        print(flat_instance.shape)
        # Reshape the flat instance back to 3D for the LSTM input
        instance_3d = flat_instance.reshape(
            (flat_instance.shape[0], 1, flat_instance.shape[1])
        ) # Reshaping to [1, timesteps, features]

        # Use the trained LSTM model for predictions
        preds = self.trained_model.predict(instance_3d)
        # Return the probabilities for both classes
        return np.column_stack((1 - preds, preds))  # for binary classification
    
    def save_model(self, file_path):
        pickle.dump(self.trained_model, open(file_path, 'wb'))
        
    def load_model(self, file_path):
        loaded_model = pickle.load(open(file_path, 'rb'))
        self.trained_model = loaded_model

    def score_model(self, y_column='exc'):
        test_x = self.testing_data.copy()
        test_y = test_x.copy()[y_column]
        test_x = test_x.drop(y_column, axis=1)
        model_type = self.model_type
        #scalers
        testing_data_normalized = self.scaler.transform(test_x.copy())

        if model_type == 'svm':    
            score = self.trained_model.score(testing_data_normalized, test_y)
            return score
            
        
        elif model_type == 'RF' or model_type == 'rf':
            score = self.trained_model.score(test_x, test_y)
            return score
        
        elif model_type == 'lstm':
            # reshape input to be 3D [samples, timesteps, features]
            test_X = testing_data_normalized.reshape((testing_data_normalized.shape[0], 1, testing_data_normalized.shape[1]))
            # Evaluate the model
            test_loss, test_accuracy = self.trained_model.evaluate(test_X, test_y)
            return test_accuracy

        else:
            print("No model specified")
            return

        