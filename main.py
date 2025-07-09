from networksecurity.components.data_ingestion import data_ingestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig , TrainingPipelineConfig
import sys


if __name__ =='__main__':
   
   try:
      logging.info('ENTERED THE TRIED BLOCK')
      trainingpipelineconfig = TrainingPipelineConfig()
      dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)

      
      dataingestion = data_ingestion(data_ingestion_config = dataingestionconfig )

      artifact = dataingestion.initiate_Data_ingestion()
      print(artifact)



   except Exception as e:
      raise NetworkSecurityException(e,sys)    




