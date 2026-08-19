from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.exception_handler import CustomMLException
import sys
from src.logger import logging


class TrainingPipeline:
    def initiate_model_training_pipeline(self):
        try:
            logging.info('** Initating the model training pipeline **')
            ingestion_obj = DataIngestion()
            train_path , test_path = ingestion_obj.initiate_data_ingestion()

            validation_obj = DataValidation()
            validation_report = validation_obj.initiate_data_validation(train_path,test_path)

            if validation_report.status == 'Failed':
                logging.info('Data validation failed')
                return

            logging.info('Data validation passed')

            

        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)
        

if __name__ == "__main__":
    try:
        train_obj = TrainingPipeline()
        train_obj.initiate_model_training_pipeline()
        
    except Exception as e:
        logging.info(f'Error Occured : {e}')
        raise CustomMLException(e,sys)