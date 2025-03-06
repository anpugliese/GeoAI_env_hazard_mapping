import numpy as np
import pandas as pd
import pickle
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

class AIProcessor:
    def __init__(self, model_type='SVM', task_type='classification', hyperparameter_tuning=True):
        self.model_type = model_type
        self.task_type = task_type
        self.hyperparameter_tuning = hyperparameter_tuning
        self.model = None
        self.scaler = StandardScaler()
        self.explainer = None
    
    def build_ann(self, optimizer='adam', neurons=[64], activation='relu'):
        model = Sequential()

        # Input Layer
        model.add(Dense(neurons[0], activation=activation, input_shape=(self.input_dim,)))

        # Hidden Layers
        for n in neurons[1:]:  # Add layers based on the list of neurons
            model.add(Dense(n, activation=activation))

        # Output Layer
        output_activation = 'linear' if self.task_type == 'regression' else 'sigmoid'
        loss_function = 'mse' if self.task_type == 'regression' else 'binary_crossentropy'
        
        model.add(Dense(1, activation=output_activation))
        
        # Compile Model
        model.compile(optimizer=optimizer, loss=loss_function, metrics=['mse' if self.task_type == 'regression' else 'accuracy'])
        return model

    
    def build_lstm(self, units=50, optimizer='adam', dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=(self.time_steps, self.input_dim)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear' if self.task_type == 'regression' else 'sigmoid'))
        model.compile(optimizer=optimizer, loss='mse' if self.task_type == 'regression' else 'binary_crossentropy', metrics=['mse' if self.task_type == 'regression' else 'accuracy'])
        return model
    
    def set_data(self,partition,normalize,data):
        if partition:
            #If partition is required, the expected params are X and y 
            #X is all the independent data
            #y is the dependent variable array
            #X and y will be partitioned to X_train, X_test, y_train, and y_test
            X = data['X']
            y = data['y']
            #X = self.scaler.fit_transform(X)
            if self.task_type == 'classification':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        else:
            #Otherwise, the partition is made externally and the params are training and testing
            #the dependent variable must be extracted to build X_train, X_test, y_train, and y_test
            
            training = data['train']
            testing = data['test']
            y_column = data['target']
            training_columns = data['training_columns']
            
            y_train = training.copy()[y_column]
            X_train = training.copy()[training_columns]
            y_test = testing.copy()[y_column]
            X_test = testing[training_columns]

        if normalize:
            self.normalized = True
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        if self.model_type == 'LSTM':
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            
            
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def fit(self):
        try:
            '''
            X = self.scaler.fit_transform(X)
            if self.task_type == 'classification':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            '''
            

            if self.model_type == 'SVM':
                model_class = SVC if self.task_type == 'classification' else SVR
            elif self.model_type == 'RF':
                model_class = RandomForestClassifier if self.task_type == 'classification' else RandomForestRegressor
            elif self.model_type == 'XGBoost':
                model_class = xgb.XGBClassifier if self.task_type == 'classification' else xgb.XGBRegressor
            
            if self.model_type in ['SVM', 'RF', 'XGBoost']:
                param_grid = {'C': [0.1, 1, 10]} if self.model_type == 'SVM' else {'n_estimators': [50, 100, 200]}
                self.model = GridSearchCV(model_class(), param_grid, cv=3) if self.hyperparameter_tuning else model_class()
            elif self.model_type == 'ANN':
                model_wrapper = KerasRegressor if self.task_type == 'regression' else KerasClassifier
                self.model = GridSearchCV(model_wrapper(build_fn=self.build_ann, epochs=50, batch_size=32, verbose=0), {'neurons': [[64], [128], [64, 32], [128, 64], [128, 64, 32]], 'activation': ['relu', 'tanh'], 'optimizer': ['adam', 'sgd']}, cv=3) if self.hyperparameter_tuning else self.build_ann()
            elif self.model_type == 'LSTM':
                self.input_dim = self.X_train.shape[2]
                self.time_steps = 1  # Adjust this as needed for LSTM
                model_wrapper = KerasRegressor if self.task_type == 'regression' else KerasClassifier
                self.model = GridSearchCV(model_wrapper(build_fn=self.build_lstm, epochs=10, batch_size=32, verbose=0), {'units': [50, 100], 'dropout_rate': [0.2, 0.3], 'optimizer': ['adam', 'rmsprop']}, cv=3) if self.hyperparameter_tuning else self.build_lstm()
            
            self.model.fit(self.X_train, self.y_train)
            
            if self.hyperparameter_tuning:
                print("Grid Search Results:")
                results = pd.DataFrame(self.model.cv_results_)
                print(results[['params', 'mean_test_score', 'rank_test_score']])
                print("Best Parameters:", self.model.best_params_)
            
            #self.X_test, self.y_test = X_test, y_test
            return self.model
        except Exception as ex: 
            print(ex)
            
    
    def predict(self, X, normalized=False):
        if normalized:
            X = self.scaler.transform(X)
        if (self.model_type == 'RF' or self.model_type == 'XGBoost') and self.model_type == 'classification':
            return self.model.predict_proba(X)[:,1]
        else:
            return self.model.predict(X)
    
    def assess(self, format='text'):
        if self.task_type == 'classification':
            y_pred = self.model.predict(self.X_test)
            y_pred = np.where(y_pred >= 0.7, 1, 0)
        else:
            y_pred = self.model.predict(self.X_test)
        
        if self.task_type == 'classification':
            if format == 'text':
                print("Classification Report:")
                #print(classification_report(self.y_test, y_pred))
                print("Accuracy:", accuracy_score(self.y_test, y_pred))
                print("Precision:", precision_score(self.y_test, y_pred))
                print("Recall:", recall_score(self.y_test, y_pred))
                print("F1 Score:", f1_score(self.y_test, y_pred))
            elif format == 'tabular':
                return [accuracy_score(self.y_test, y_pred), precision_score(self.y_test, y_pred), recall_score(self.y_test, y_pred), f1_score(self.y_test, y_pred)]

        else:
            if format == 'text':
                print("Regression Metrics:")
                print("Mean Absolute Error (MAE):", mean_absolute_error(self.y_test, y_pred))
                print("Mean Squared Error (MSE):", mean_squared_error(self.y_test, y_pred))
                print("Root Mean Squared Error (RMSE):", np.sqrt(mean_squared_error(self.y_test, y_pred)))
                print("R² Score:", r2_score(self.y_test, y_pred))
            elif format == 'tabular':
                return [mean_absolute_error(self.y_test, y_pred), mean_squared_error(self.y_test, y_pred), np.sqrt(mean_squared_error(self.y_test, y_pred)), r2_score(self.y_test, y_pred)]

    def save_model(self, file_path):
        pickle.dump(self.model, open(file_path, 'wb'))
        
    def load_model(self, file_path):
        loaded_model = pickle.load(open(file_path, 'rb'))
        self.model = loaded_model

    def save_scaler(self, file_path):
        pickle.dump(self.scaler, open(file_path, 'wb'))
        
    def load_scaler(self, file_path):
        loaded_scaler = pickle.load(open(file_path, 'rb'))
        self.scaler = loaded_scaler

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
    

    def score_model(self, y_column='exc'):
        test_x = self.X_test
        test_y = self.y_test
        model_type = self.model_type
        #scalers
        #testing_data_normalized = self.scaler.transform(test_x.copy())

        if model_type == 'SVM':    
            score = self.model.score(test_x, test_y)
            return score
            
        
        elif model_type == 'RF' or model_type == 'rf':
            score = self.model.score(test_x, test_y)
            return score
        
        elif model_type == 'LSTM':
            # reshape input to be 3D [samples, timesteps, features]
            #test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
            # Evaluate the model
            test_loss, test_accuracy = self.model.evaluate(test_x, test_y)
            return test_accuracy
        
        elif model_type == 'XGBoost':
            score = self.model.score(test_x, test_y)
            return score

        else:
            print("No model specified")
            return
    
    def explain_model(self):
        if self.model == 'XGBoost':
            if self.explainer == None and self.X_train != None:
                #initialize explainer based on the model
                explainer = shap.Explainer(self.model)   
                #Define SHAP explanation of the train dataset             
                shap_values = explainer(self.X_train)
                #Display beeswarm plot of the training data SHAP explanation
                shap.plots.beeswarm(shap_values)
                #Display bar plot of absolute SHAP values explanations
                shap.plots.bar(shap_values)
                return shap_values

        else:
            return None

    def explain_prediction(self, input):
        if self.explainer != None:
            shap_values = self.explainer(input)
            #Display waterfall plot of SHAP explanation of the provided input
            shap.plots.waterfall(shap_values[0])
            #Display beeswarm plot of SHAP explanation of the provided input
            shap.plots.beeswarm(shap_values[0])
        
        else:
            print('Please initialize a explainer with the explain_model function')
            return