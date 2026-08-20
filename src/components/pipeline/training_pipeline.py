from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
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
            logging.info('Data validation passed')

            transformation_obj = DataTransformation()
            X_train_transformed,X_test_transformed,y_train_transformed,y_test_transformed,preprocessor_path = transformation_obj.initiate_data_transformation(train_path,test_path)
            logging.info(f'X_train_transformed : {X_train_transformed.shape} ,X_test_transformed : {X_test_transformed.shape} ')
            logging.info(f'y_train_transformed : {y_train_transformed.shape} ,y_test_transformed : {y_test_transformed.shape} ')
            logging.info(f'preprocessor_path : {preprocessor_path} ')
            

            

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