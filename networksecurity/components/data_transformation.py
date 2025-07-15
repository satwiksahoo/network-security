import sys
import os
import numpy  as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import TARGET_COLUMN , DATA_TRANSFORMATION_IMPUTER_PARAMS

from networksecurity.entity.artifact_entity import DataTransformationArtifact , dataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data , save_object

class DataTransformation:
    def __init__(self , data_validation_artifact : dataValidationArtifact , data_transformation_config : DataTransformationConfig):
        try:
            self.data_Validation_Artifact : dataValidationArtifact = data_validation_artifact

            self.data_transformation_config : DataTransformationConfig = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e ,sys)


    @staticmethod
    def read_data(file_path)->pd.DataFrame:


        try:
            return pd.read_csv(file_path)   
        except Exception as e:
     
            raise NetworkSecurityException(e , sys)    
        
    def get_data_transformer_object(cls)-> Pipeline:

        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            logging.info('initialise knnimputer')

            processor:Pipeline = Pipeline([('imputer',imputer)])
            return processor


        except Exception as e:
            raise NetworkSecurityException(e ,sys)



        
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info('Entered the initiate data transformation block')

        try:


            # -------------------------
            print("Train Path:", self.data_transformation_config.transformed_train_file_path)
            print("Test Path:", self.data_transformation_config.transformed_test_file_path)
            print("Object Path:", self.data_transformation_config.transformed_object_file_path)
            print("Target directory:", self.data_transformation_config.data_transformation_dir)



            
            # -------------------------
            logging.info('starting data Transofrmation')
            train_df = DataTransformation.read_data(self.data_Validation_Artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_Validation_Artifact.valid_test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN] , axis = 1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1 , 0)




            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN] , axis = 1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1 , 0)


            prepocessor = self.get_data_transformer_object()
            prepocessor_obj = prepocessor.fit(input_feature_train_df)
            transformed_input_train_feature = prepocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = prepocessor_obj.transform(input_feature_test_df)


            train_arr = np.c_[transformed_input_train_feature , np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature , np.array(target_feature_test_df)]


            os.makedirs(self.data_transformation_config.data_transformation_dir, exist_ok=True) #



            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path , array = train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path , array = test_arr)


            save_object(self.data_transformation_config.transformed_object_file_path , prepocessor_obj)

            data_transformation_artifact = DataTransformationArtifact(

                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path, 
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path, 
            )

            return data_transformation_artifact


















        except Exception as e:
            raise NetworkSecurityException(e ,sys)
