from networksecurity.entity.artifact_entity import DataIngestionArtifact , dataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd 
import os , sys
from networksecurity.utils.main_utils.utils import read_yaml_file ,write_yaml_file
 

class Datavalidation:
    def __init__(self , data_ingestion_artifact  = DataIngestionArtifact , data_validation_arifact = dataValidationArtifact , 
                 data_validation_config = DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_arifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise NetworkSecurityException(e , sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:


        try:
            return pd.read_csv(file_path)   
        except Exception as e:
     
            raise NetworkSecurityException(e , sys)
        



    def validate_number_of_cols(self, dataframe:pd.DataFrame)-> bool:
        try:
            number_of_columns = len(self._schema_config) 
            logging.info(f'required number of columns : {number_of_columns}')
            logging.info(f'dataframe has columns : {len(dataframe.columns)}')

            if len(dataframe.columns) == number_of_columns :
                return True
            
            return False


        except Exception as e:
     
            raise NetworkSecurityException(e , sys) 
        

    def detect_dataset_drift(self , base_df , current_df , threshold=0.05) -> bool:
        try:
            

            status = True 
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]

                is_same_dist =  ks_2samp(d1 , d2)

                if threshold< is_same_dist.pvalue:
                    is_found = False

                else:
                    is_found = True
                    status = False
                report.update(
                    {
                        column:{

                            'p_value':float(is_same_dist.pvalue),
                            'drift_status' : is_found


                        }
                    }

                )    

            
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path , exist_ok=True)


            write_yaml_file(file_path = drift_report_file_path , content = report)


            return status



        except Exception as e:
     
            raise NetworkSecurityException(e , sys) 


        


    def initiate_data_validation(self) -> dataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.training_file_path
            test_file_path  = self.data_ingestion_artifact.test_file_path

            train_dataframe = Datavalidation.read_data(train_file_path)
            test_dataframe = Datavalidation.read_data(test_file_path)


            status_train = self.validate_number_of_cols(dataframe=train_dataframe)

            if not status_train:
                error_message = 'train dataframe does not contain all columns'

            status_test = self.validate_number_of_cols(dataframe=test_dataframe)

            if not status_test:
                error_message = 'test dataframe does not contain all columns'

            status = self.detect_dataset_drift(base_df=train_dataframe,  current_df=test_dataframe)    

            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)

            os.makedirs(dir_path , exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index = False,header = True
            )


            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index = False,header = True
            )


            data_validaiton_artifact =  dataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.training_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,



            )



            return data_validaiton_artifact


            
        except Exception as e:
            raise NetworkSecurityException(e , sys)

    



