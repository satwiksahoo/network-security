import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def read_yaml_file(file_path : str) -> dict:

    try:
        with open(file_path , 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:

        raise NetworkSecurityException(e ,sys)    
    



# def write_yaml_file (file_path:str ,content: object ,replace : bool = False ) -> None:
   
#     try:
#         if replace:
                 
#                  if os.path.exists(file_path):
#                      os.remove(file_path)

#                  os.makedirs(os.path.dirname(file_path) , exist_ok=True)


#                  with open(file_path, 'w') as file:
#                   yaml.dump(content , file)
       
#     except Exception as e:
#         print(f"Error writing YAML file: {e}")



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        with open(file_path, 'w') as file:
            yaml.dump(content, file)

    except Exception as e:
        print(f"Error writing YAML file: {e}")


def save_numpy_array_data(file_path : str , array : np.array):

    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,  'wb') as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e ,sys)        
    

def save_object(file_path : str , obj : object) -> None:
    try:
        logging.info('Entered he save_object method of MainUtils class')
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path , 'wb') as file_obj:
            pickle.dump(obj , file_obj)

        logging.info('exited the save object method of mainutils class') 

    except Exception as e:
        raise NetworkSecurityException(e , sys)   



def load_object(file_path : str) -> object :
    try:
        if not os.path.exists(file_path):
            raise Exception(f'file {file_path} doesnt exist')
        with open(file_path , 'rb') as file_obj :
            print(file_obj) 
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e ,sys)   
    
def load_numpy_array(file_path : str) -> np.array:
    try:
        with open(file_path , 'rb') as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise NetworkSecurityException(e , sys)
    


def evaluate_model(x_train , y_train , x_test , y_test , models , params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model , param , cv = 3)
            gs.fit(x_train , y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train , y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train , y_train_pred)

            test_model_score = r2_score(y_test , y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise NetworkSecurityException(e , sys)


# def evaluate_model(x_train, y_train, x_test, y_test, models, params):

#     report = {}

#     for model_name, model in models.items():
#         param_grid = params[model_name]

#         gs = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
#         gs.fit(x_train, y_train)

#         best_model = gs.best_estimator_

#         y_pred = best_model.predict(x_test)

#         acc_score = accuracy_score(y_test, y_pred)

#         report[model_name] = acc_score

#     return report





