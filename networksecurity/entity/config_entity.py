from datetime import datetime
import os
from networksecurity.constants import training_pipeline


class TrainingPipelineConfig:

    def __init__(self , timestamp = datetime.now()):
        timestamp = timestamp.strftime('%m_%d_%Y_%H_%M_%S')
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.timestamp : str = timestamp





class DataIngestionConfig:
    def __init__(self , training_pipeline_config: TrainingPipelineConfig):

        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir ,training_pipeline.DATA_INGESETION_DIR_NAME)

        # self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir ,training_pipeline.DATA_INGESETION_FEATURE_STORE_DIR)
        self.feature_store_file_path: str = os.path.join(
    self.data_ingestion_dir,
    training_pipeline.DATA_INGESETION_FEATURE_STORE_DIR,
    training_pipeline.FILE_NAME
)

        self.training_file_path: str = os.path.join(self.data_ingestion_dir ,training_pipeline.DATA_INGESETION_INGESTED_DIR , training_pipeline.TRAIN_FILE_NAME)

        self.test_file_path: str = os.path.join(self.data_ingestion_dir ,training_pipeline.DATA_INGESETION_INGESTED_DIR , training_pipeline.TEST_FILE_NAME)

        self.train_test_split_ratio:float = training_pipeline.DATA_INGESETION_TRAIN_TEST_SPLIT_RATIO

        self.collection_name : str = training_pipeline.DATA_INGESETION_COLLECTION_NAME

        self.database_name :str = training_pipeline.DATA_INGESETION_DATABASE_NAME  


class DataValidationConfig:

     def __init__(self , training_pipeline_config: TrainingPipelineConfig):

        # self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir ,training_pipeline.DATA_VALIDATION_DIR_NAME)

        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir , training_pipeline.DATA_VALIDATION_DIR_NAME)

        self.valid_data_dir: str = os.path.join(self.data_validation_dir ,training_pipeline.DATA_VALIDATION_VALID_DIR)

        self.invalid_data_dir: str = os.path.join(self.data_validation_dir ,training_pipeline.DATA_VALIDATION_INVALID_DIR)


        self.valid_train_file_path:str = os.path.join(self.valid_data_dir , training_pipeline.TRAIN_FILE_NAME )

        self.valid_test_file_path:str = os.path.join(self.valid_data_dir , training_pipeline.TEST_FILE_NAME )


        self.invalid_train_file_path:str = os.path.join(self.invalid_data_dir , training_pipeline.TRAIN_FILE_NAME )
        self.invalid_test_file_path:str = os.path.join(self.invalid_data_dir , training_pipeline.TEST_FILE_NAME )

        self.drift_report_file_path = os.path.join(self.data_validation_dir , 
                                                   training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR , 
                                                   training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
    





class DataTransformationConfig:

    def __init__(self , training_pipeline_config: TrainingPipelineConfig):
        # self.data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir , training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        # self.transformed_train_file_path : str = os.path.join(self.data_transformation_dir , training_pipeline.DATA_TRANSFORMATION_DIR_NAME ,
        #                                                        training_pipeline.TRAIN_FILE_NAME.replace('csv' , 'npy'))
        
        # self.transformed_test_file_path : str = os.path.join(self.data_transformation_dir , training_pipeline.DATA_TRANSFORMATION_DIR_NAME ,
        #                                                        training_pipeline.TEST_FILE_NAME.replace('csv' , 'npy'))
        

        # self.transformed_object_file_path : str = os.path.join(self.data_transformation_dir , training_pipeline.DATA_TRANSFORMATION_DIR_NAME ,
        #                                                        training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)
        


        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(self.data_transformation_dir, training_pipeline.TRAIN_FILE_NAME.replace('csv','npy'))
        self.transformed_test_file_path  = os.path.join(self.data_transformation_dir, training_pipeline.TEST_FILE_NAME.replace('csv','npy'))
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir, training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)



class ModelTrainerConfig:

    def __init__(self , training_pipeline_config : TrainingPipelineConfig):

        self.model_trainer_dir : str = os.path.join(training_pipeline_config.artifact_dir , training_pipeline.MODEL_TRAINER_DIR_NAME)

        self.trained_model_file_path : str = os.path.join(self.model_trainer_dir , training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR ,training_pipeline.MODEL_FILE_NAME)
        self.expected_Accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE

        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD


