import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
import pymongo.mongo_client
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from dotenv import load_dotenv


load_dotenv()

MONGO_DB_URL = os.getenv('MONGO_DB_URL')

print(MONGO_DB_URL)

ca = certifi.where()


class network_Data_Extract():

    def __init__(self):

        try:
            pass

        except Exception as e:
            raise NetworkSecurityException(e , sys)
        


    def csv_to_json(self , file_path):
        try:
            data = pd.read_csv(file_path)

            data.reset_index(drop = True , inplace= True)

            records = list(json.loads(data.T.to_json()).values())

            return records
            

        except Exception as e:
            raise NetworkSecurityException(e , sys)
        

    def insert_Data_to_mongodb(self ,records , database , collection):

        try:
            self.database = database
            self.collection  = collection
            self.records = records
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e , sys)
        

if __name__ == '__main__':

    File_Path = 'network_Data/phisingData.csv'
    DATABASE = 'network-security-database'
    collection = 'network-data' 
    network = network_Data_Extract()
    records = network.csv_to_json(file_path = File_Path)
    no_of_record = network.insert_Data_to_mongodb(records , DATABASE , collection)

    print(no_of_record)
    print(records)
  







