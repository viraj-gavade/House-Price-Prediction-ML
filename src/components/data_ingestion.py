from src.exception_handler import CustomMLException
from src.logger import logging
from dataclasses import dataclass
import os 
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path : str = os.path.join('Artifcats/Data','raw.csv')
    train_data_path : str = os.path.join('Artifcats/Data','train.csv')
    test_data_path : str = os.path.join('Artifcats/Data','test.csv')


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            logging.info('** PHASE 1 : Initiating the Data Ingestion Pipeline **')
            logging.info('Loading the dataset from the source')
            df = pd.read_csv('data/AmesHousing.csv')
            logging.info(f'Data Loaded sucessfully : {df.shape}')

            logging.info('Creating the artifcats Directory')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            logging.info('Artifacts directory created successfully')

            logging.info('Applying the train test split to split the data')
            train_df , test_df = train_test_split(df,test_size=0.2,random_state=42)
            logging.info('Train test split applied successfully')
            logging.info(f'Train Shape : {train_df.shape} , Test Shape : {test_df.shape}')


            logging.info('Saving the raw data to the artifacts folder')
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Raw data saved successfully to artifacts folder')

            

            logging.info('Saving the train data to the artifacts folder')
            df.to_csv(self.data_ingestion_config.train_data_path,index=False)
            logging.info('train data saved successfully to artifacts folder')


            logging.info('Saving the test data to the artifacts folder')
            df.to_csv(self.data_ingestion_config.test_data_path,index=False)
            logging.info('test data saved successfully to artifacts folder')



            logging.info('** PHASE 1 : Data Ingestion Pipeline Completed Successfully! **')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )


        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)