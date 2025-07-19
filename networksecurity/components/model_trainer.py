import sys
import os
import numpy  as np
import pandas as pd
import mlflow
from networksecurity.constants.training_pipeline import TARGET_COLUMN , DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact , dataValidationArtifact , ModelTrainerArtifact 
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


from networksecurity.utils.main_utils.utils import save_object, load_object , load_numpy_array , evaluate_model
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier , GradientBoostingClassifier , RandomForestClassifier)

class ModelTrainer:
    def __init__(self , model_trainer_config : ModelTrainerConfig , data_transformation_artifact : DataTransformationArtifact):

        try:

            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            


        except Exception as e:
            raise NetworkSecurityException(e , sys)
    def track_mlflow(self , best_model , classificationmetrics , run_name):
        
        mlflow.set_tracking_uri("http://localhost:5000")
        
        with mlflow.start_run(run_name=run_name):
            f1_score = classificationmetrics.f1_score
            precision_score = classificationmetrics.precision_score
            recall_score = classificationmetrics.recall_score
            
            mlflow.log_metric('f1_Score'  , f1_score)
            mlflow.log_metric('precision_Score'  , precision_score)
            mlflow.log_metric('recall_Score'  , recall_score)
            
            # mlflow.sklearn.log_model(best_model , 'best model')
            
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path="model")

        
        
        pass
        
        
    def train_model(self , x_train , y_train , x_test , y_test):
        models = {
    # 'ada boost': AdaBoostClassifier(),
    'gradient boosting': GradientBoostingClassifier(verbose=1),
    'random forest': RandomForestClassifier(verbose=1),
    'decision tree': DecisionTreeClassifier(),
    'logistic': LogisticRegression(verbose=1)
     }

        params = {
    # 'ada boost': {
    #     'n_estimators': [50, 100, 200],
    #     'algorithm': ['SAMME', 'SAMME.R']
    # },
    'gradient boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1]
    },
    'random forest': {
        'n_estimators': [100, 200],
        'max_features': ['sqrt', 'log2']
    },
    'decision tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10]
    },
    'logistic': {
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200]
    }
     }

        
        model_report: dict = evaluate_model(x_train=x_train , y_train=y_train ,x_test = x_test , y_test  = y_test , models = models , params = params)
        
        best_model_Score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_Score)]

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(x_train)

        classification_Train_metric = get_classification_score(y_true=y_train, y_pred = y_train_pred)
        # self.trac
        
        
        self.track_mlflow(best_model , classification_Train_metric , run_name = 'train')  ########

        y_test_pred = best_model.predict(x_test)

        classification_Test_metric = get_classification_score(y_true=y_test, y_pred = y_test_pred)
        
        self.track_mlflow(best_model , classification_Test_metric , run_name = 'test')  ########
        

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)

        os.makedirs(model_dir_path, exist_ok=True)

        Network_model = NetworkModel(preprocessor=preprocessor , model = best_model)

        save_object(self.model_trainer_config.trained_model_file_path , obj=NetworkModel)

        model_trainer_Artifact = ModelTrainerArtifact(trained_model_file_path = self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_Train_metric , 
                             test_metric_artifact=classification_Test_metric)
        
        return model_trainer_Artifact
    #     model = {


    #          'ada boost' : AdaBoostClassifier() ,
    #          'gradient boosting': GradientBoostingClassifier(verbose = 1) ,
    #            'random forest' : RandomForestClassifier(verbose = 1) , 
    #            'decision tree': DecisionTreeClassifier(), 
    #            'logistic' : LogisticRegression(verbose = 1)

    #     }


    #     param = {
    # 'ada boost': {
    #     'n_estimators': [50, 100, 200],
    #     # 'learning_rate': [0.01, 0.1, 1],
    #     'algorithm': ['SAMME', 'SAMME.R']
    # },

    # 'gradient boosting': {
    #     'n_estimators': [100, 200],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     # 'max_depth': [3, 5, 7],
    #     # 'subsample': [0.8, 1.0]
    # },

    # 'random forest': {
    #     'n_estimators': [100, 200],
    #     # 'max_depth': [None, 10, 20],
    #     'max_features': ['sqrt', 'log2'],
    #     # 'min_samples_split': [2, 5],
    #     # 'min_samples_leaf': [1, 2]
    # },

    # 'decision tree': {
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': [None, 5, 10],
    #     # 'min_samples_split': [2, 5, 10],
    #     # 'min_samples_leaf': [1, 2, 4]
    # },

    # 'logistic': {
    #     'penalty': ['l2'],
    #     'C': [0.01, 0.1, 1, 10],
    #     'solver': ['liblinear', 'lbfgs'],
    #     'max_iter': [100, 200]
    #  }
    #   }


     

        
        
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train , y_train , x_test , y_test = (  train_arr[: , :-1] , train_arr[: , -1] , train_arr[: , :-1] , train_arr[: , -1]  )
            

            model = self.train_model(x_train , y_train , x_test , y_test)
        except Exception as e:
            raise NetworkSecurityException(e ,sys)

