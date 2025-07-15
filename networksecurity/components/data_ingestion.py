import pymongo.mongo_client
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.entity.config_entity import DataIngestionConfig
import os 
import sys
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

class data_ingestion:

    def __init__(self , data_ingestion_config : DataIngestionConfig):
        try:
            
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e , sys)
        


    def export_collection_Dataframe(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if '_id' in df.columns.to_list():
               df = df.drop(columns = ['_id'] ,axis = 1)

            df.replace({'na' : np.nan} , inplace = True)   
            return df
        except Exception as e:
            raise NetworkSecurityException(e ,sys) 


    def export_Data_to_feature_store(self , dataframe:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path , exist_ok=True)
            dataframe.to_csv(feature_store_file_path , index=False, header = True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e , sys)

    def split_data_as_train_set(self , dataframe : pd.DataFrame):
        try:
            train_set , test_set = train_test_split(dataframe , test_size=self.data_ingestion_config.train_test_split_ratio , random_state=42)
            logging.info('performed train test split')
            
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path , exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index = False , header = True)

            test_set.to_csv(self.data_ingestion_config.test_file_path , index = False , header = True)

            logging.info('exported train test file')



        except Exception as e:
            raise NetworkSecurityException(e ,sys)     
        
    def initiate_Data_ingestion(self):
        try:
            dataframe = self.export_collection_Dataframe()
            dataframe = self.export_Data_to_feature_store(dataframe)
            self.split_data_as_train_set(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(training_file_path=self.data_ingestion_config.training_file_path , test_file_path= self.data_ingestion_config.test_file_path)

            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e ,sys)    