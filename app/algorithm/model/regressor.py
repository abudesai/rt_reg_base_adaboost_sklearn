#Import required libraries
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error


model_fname = "model.save"

MODEL_NAME = "reg_base_adaboost"

class Regressor(): 
    
    def __init__(self, n_estimators=250, learning_rate=1e-2, loss="square", **kwargs) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = np.float(learning_rate)
        self.loss= loss
        
        self.model = self.build_model()
        
        
        
    def build_model(self): 
        model = AdaBoostRegressor(n_estimators= self.n_estimators, learning_rate= self.learning_rate, 
        random_state=42, loss=self.loss)
        return model
    
    
    def fit(self, train_X, train_y):        
                
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            # return self.model.score(x_test, y_test)        
            preds = self.model.predict(x_test)
            mse = mean_squared_error(y_test, preds, squared=False)
            return mse   

    
    def save(self, model_path): 
        joblib.dump(self.model, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path): 
        adaboost = joblib.load(os.path.join(model_path, model_fname))
        return adaboost


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    try: 
        model = Regressor.load(model_path)        
    except: 
        raise Exception(f'''Error loading the trained {MODEL_NAME} model. 
            Do you have the right trained model in path: {model_path}?''')
    return model
