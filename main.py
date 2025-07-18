from networksecurity.components.data_ingestion import data_ingestion 
from networksecurity.components.data_validation import Datavalidation 
from networksecurity.components.data_transformation import DataTransformation 
from networksecurity.components.model_trainer import ModelTrainer 
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig , TrainingPipelineConfig , DataValidationConfig ,DataTransformationConfig , ModelTrainerConfig
import sys
from networksecurity.entity.artifact_entity import DataIngestionArtifact , dataValidationArtifact


if __name__ =='__main__':
   
   try:
      logging.info('ENTERED THE TRIED BLOCK')

      trainingpipelineconfig = TrainingPipelineConfig()

      dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)

      
      dataingestion = data_ingestion(data_ingestion_config = dataingestionconfig )

      artifact = dataingestion.initiate_Data_ingestion()
      # print(artifact)

      logging.info('data validation started')


      datavalidationconfig = DataValidationConfig(trainingpipelineconfig)

      data_validation = Datavalidation(
    data_ingestion_artifact=artifact,
    data_validation_arifact=None,  # You can leave it None for now if not used
    data_validation_config=datavalidationconfig
)

      data_validation_artifact = data_validation.initiate_data_validation()

      logging.info('data validation completed')



      logging.info('data TRANSFORMATION started')

      datatransformationconfig = DataTransformationConfig(trainingpipelineconfig)

      data_transformation  = DataTransformation(

         data_validation_artifact = data_validation_artifact ,
         data_transformation_config = datatransformationconfig




      )

      data_transformation_Artifact  = data_transformation.initiate_data_transformation()





      logging.info('data TRANSFORMATION completed')


      logging.info('model trainer started')

      pipeline_config = TrainingPipelineConfig()  # create an instance
# modeltrainerconfig = ModelTrainerConfig(pipeline_config)

      modeltrainerconfig = ModelTrainerConfig(pipeline_config)
      model_trainer = ModelTrainer(model_trainer_config=modeltrainerconfig, data_transformation_artifact=data_transformation_Artifact)

      model_trainer_artifact = model_trainer.initiate_model_trainer()

      logging.info('model trainer completed')












   except Exception as e:
      raise NetworkSecurityException(e,sys)    




