import os
import sys
import numpy as np
import pandas as pd

TARGET_COLUMN = 'Result'
PIPELINE_NAME : str= 'NetworkSecurity'
ARTIFACT_DIR : str  = 'Artifacts'
FILE_NAME :str = 'phishingData.csv'

TRAIN_FILE_NAME : str = 'train.csv'
TEST_FILE_NAME : str = 'test.csv'

DATA_INGESETION_COLLECTION_NAME : str = 'network-data'
DATA_INGESETION_DATABASE_NAME : str = 'network-security-database'
DATA_INGESETION_DIR_NAME : str = 'data_ingestion'
DATA_INGESETION_FEATURE_STORE_DIR : str = 'feature_store'
DATA_INGESETION_INGESTED_DIR : str = 'ingested'
DATA_INGESETION_TRAIN_TEST_SPLIT_RATIO : float = 0.2
