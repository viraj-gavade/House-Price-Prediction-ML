from src.exception_handler import CustomMLException
from src.logger import logging
from dataclasses import dataclass
import os 
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import field 


class DataValidationConfig:
    expected_datatypes : dict = field(default_factory=lambda: {
    "Order": "int64",
    "PID": "int64",
    "MS SubClass": "int64",
    "MS Zoning": "object",
    "Lot Frontage": "float64",
    "Lot Area": "int64",
    "Street": "object",
    "Alley": "object",
    "Lot Shape": "object",
    "Land Contour": "object",
    "Utilities": "object",
    "Lot Config": "object",
    "Land Slope": "object",
    "Neighborhood": "object",
    "Condition 1": "object",
    "Condition 2": "object",
    "Bldg Type": "object",
    "House Style": "object",
    "Overall Qual": "int64",
    "Overall Cond": "int64",
    "Year Built": "int64",
    "Year Remod/Add": "int64",
    "Roof Style": "object",
    "Roof Matl": "object",
    "Exterior 1st": "object",
    "Exterior 2nd": "object",
    "Mas Vnr Type": "object",
    "Mas Vnr Area": "float64",
    "Exter Qual": "object",
    "Exter Cond": "object",
    "Foundation": "object",
    "Bsmt Qual": "object",
    "Bsmt Cond": "object",
    "Bsmt Exposure": "object",
    "BsmtFin Type 1": "object",
    "BsmtFin SF 1": "float64",
    "BsmtFin Type 2": "object",
    "BsmtFin SF 2": "float64",
    "Bsmt Unf SF": "float64",
    "Total Bsmt SF": "float64",
    "Heating": "object",
    "Heating QC": "object",
    "Central Air": "object",
    "Electrical": "object",
    "1st Flr SF": "int64",
    "2nd Flr SF": "int64",
    "Low Qual Fin SF": "int64",
    "Gr Liv Area": "int64",
    "Bsmt Full Bath": "float64",
    "Bsmt Half Bath": "float64",
    "Full Bath": "int64",
    "Half Bath": "int64",
    "Bedroom AbvGr": "int64",
    "Kitchen AbvGr": "int64",
    "Kitchen Qual": "object",
    "TotRms AbvGrd": "int64",
    "Functional": "object",
    "Fireplaces": "int64",
    "Fireplace Qu": "object",
    "Garage Type": "object",
    "Garage Yr Blt": "float64",
    "Garage Finish": "object",
    "Garage Cars": "float64",
    "Garage Area": "float64",
    "Garage Qual": "object",
    "Garage Cond": "object",
    "Paved Drive": "object",
    "Wood Deck SF": "int64",
    "Open Porch SF": "int64",
    "Enclosed Porch": "int64",
    "3Ssn Porch": "int64",
    "Screen Porch": "int64",
    "Pool Area": "int64",
    "Pool QC": "object",
    "Fence": "object",
    "Misc Feature": "object",
    "Misc Val": "int64",
    "Mo Sold": "int64",
    "Yr Sold": "int64",
    "Sale Type": "object",
    "Sale Condition": "object",
    "SalePrice": "int64"
})
    missing_cols: list = field(default_factory=list)
    target_variable : str = 'SalePrice'
    unexpected_cols: list = field(default_factory=list)
    expected_features_columns_list : list =field(default_factory=lambda:['Order',
 'PID',
 'MS SubClass',
 'MS Zoning',
 'Lot Frontage',
 'Lot Area',
 'Street',
 'Alley',
 'Lot Shape',
 'Land Contour',
 'Utilities',
 'Lot Config',
 'Land Slope',
 'Neighborhood',
 'Condition 1',
 'Condition 2',
 'Bldg Type',
 'House Style',
 'Overall Qual',
 'Overall Cond',
 'Year Built',
 'Year Remod/Add',
 'Roof Style',
 'Roof Matl',
 'Exterior 1st',
 'Exterior 2nd',
 'Mas Vnr Type',
 'Mas Vnr Area',
 'Exter Qual',
 'Exter Cond',
 'Foundation',
 'Bsmt Qual',
 'Bsmt Cond',
 'Bsmt Exposure',
 'BsmtFin Type 1',
 'BsmtFin SF 1',
 'BsmtFin Type 2',
 'BsmtFin SF 2',
 'Bsmt Unf SF',
 'Total Bsmt SF',
 'Heating',
 'Heating QC',
 'Central Air',
 'Electrical',
 '1st Flr SF',
 '2nd Flr SF',
 'Low Qual Fin SF',
 'Gr Liv Area',
 'Bsmt Full Bath',
 'Bsmt Half Bath',
 'Full Bath',
 'Half Bath',
 'Bedroom AbvGr',
 'Kitchen AbvGr',
 'Kitchen Qual',
 'TotRms AbvGrd',
 'Functional',
 'Fireplaces',
 'Fireplace Qu',
 'Garage Type',
 'Garage Yr Blt',
 'Garage Finish',
 'Garage Cars',
 'Garage Area',
 'Garage Qual',
 'Garage Cond',
 'Paved Drive',
 'Wood Deck SF',
 'Open Porch SF',
 'Enclosed Porch',
 '3Ssn Porch',
 'Screen Porch',
 'Pool Area',
 'Pool QC',
 'Fence',
 'Misc Feature',
 'Misc Val',
 'Mo Sold',
 'Yr Sold',
 'Sale Type',
 'Sale Condition',
 'SalePrice'])

class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
    

    def check_expected_cols(self ,df:pd.DataFrame)->dict:
        try:
            logging.info('Checking the expected Columns : ')
            for col in df.columns:
                if col in self.data_validation_config.expected_features_columns_list:
                    logging.info(f'{col} is present in the dataset')
                else:
                    logging.info(f'{col} is not present in the dataset')
                    self.data_validation_config.missing_cols.append(col)
            logging.info(f'Missing columns : {self.data_validation_config.missing_cols}')
            return self.data_validation_config.missing_cols
        except Exception as e :
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)

    def check_unexpected_cols(self,df:pd.DataFrame)->dict:
        try:
            logging.info('Checking for unexpected columns : ')  
            for col in self.data_validation_config.expected_features_columns_list:
                if col not in df.columns:
                    logging.info(f'{col} is unexpected in the dataset')
                    self.data_validation_config.unexpected_cols.append(col)
            logging.info(f'Unexpected columns : {self.data_validation_config.unexpected_cols}')
            return self.data_validation_config.unexpected_cols
        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)
    

    def check_expected_dtype(self ,df:pd.DataFrame)->dict:
        try:
            data_type_errors = {}
            logging.info('Checking the expected dtypes')
            for col_name , expected_dtype in self.data_validation_config.expected_datatypes.items():
                if col_name not in df.columns:
                    continue
                actual_dtype = str(df[col_name].dtype)
                if actual_dtype != expected_dtype:
                    logging.info(f'{col_name} has unexpected dtype : {actual_dtype} , expected dtype : {expected_dtype}')
                    data_type_errors[col_name] = {'actual_dtype':actual_dtype,'expected_dtype':expected_dtype}
            logging.info(f'Data type errors : {data_type_errors}')
            return data_type_errors
            
        except Exception as e :
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)


    def check_missing_values(self,df:pd.DataFrame)->dict:
        try:
            missing_report = {}
            logging.info('Checking for missing values : ')
            for col in df.columns:
                if col not in self.data_validation_config.expected_features_columns_list:
                    continue
                missing_values = df[col].isnull().sum()
                if missing_values > 0:
                    logging.info(f'{col} has {missing_values} missing values')
                    missing_report[col] = {'column_name':col , 'missing_values':missing_values}
            logging.info(f'Missing values : {missing_report}')
            return missing_report
        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)

    def check_duplicates(self,df:pd.DataFrame)->int:
        try:
            logging.info('Checking for duplicate values : ')
            duplicates = df.duplicated().sum()
            logging.info(f'Duplicate values : {duplicates}')
            return duplicates
        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)
            
    def target_validation(self , df:pd.DataFrame)->bool:
        try:
            logging.info('Checking for target variable : ')
            if self.data_validation_config.target_variable not in df.columns:
                logging.info(f'{self.data_validation_config.target_variable} is not present in the dataset')
                return False
            logging.info(f'{self.data_validation_config.target_variable} is present in the dataset')
            return True
        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)
             
    def initiate_data_validation(self,train_path: str , test_path : str ):
        try:
            validation_report = {}
            logging.info('** PHASE 2 : Initiating the Data Validation Pipeline **')
            logging.info('Reading the train and test dataset')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test dataset read successfully')

            logging.info('Checking the expected Columns in train and test dataset')
            missing_cols_train = self.check_expected_cols(train_df)
            missing_cols_test = self.check_expected_cols(test_df)
            if (len(missing_cols_train) ==0) and (len(missing_cols_test) == 0):
                validation_report['expected_cols'] = True
                logging.info('Expected columns are present in the dataset')
            else:
                validation_report['expected_cols'] = {
                    'missing_train_cols' : missing_cols_train,
                    'missing_test_cols' : missing_cols_test
                }

            logging.info('Checking the unexpected columns in train and test dataset')
            unexpected_cols_train = self.check_unexpected_cols(train_df)
            unexpected_cols_test = self.check_unexpected_cols(test_df)
            if (len(unexpected_cols_train) == 0) and (len(unexpected_cols_test) == 0):
                validation_report['unexpected_cols'] = True
                logging.info('Unexpected columns are not present in the dataset')
            else:
                validation_report['unexpected_cols'] = {
                    'unexpected_train_cols' : unexpected_cols_train,
                    'unexpected_test_cols' : unexpected_cols_test
                }
            
            logging.info('Checking the expected data types in train and test dataset')
            dtype_error_train = self.check_expected_dtype(train_df)
            dtype_error_test = self.check_expected_dtype(test_df)
            if (len(dtype_error_train) == 0) and (len(dtype_error_test) == 0):
                validation_report['expected_dtype'] = True
                logging.info('Expected data types are present in the dataset')
            else:
                validation_report['expected_dtype'] = {
                    'dtype_error_train' : dtype_error_train,
                    'dtype_error_test' : dtype_error_test
                }

            logging.info('Checking for the missing values in train and test dataset')
            missing_report_train = self.check_missing_values(train_df)
            missing_report_test = self.check_missing_values(test_df)
            if (len(missing_report_train) == 0) and (len(missing_report_test) == 0):
                validation_report['missing_values'] = True
                logging.info('Missing values are not present in the dataset')
            else:
                validation_report['missing_values'] = {
                    'missing_train_report' : missing_report_train,
                    'missing_test_report' : missing_report_test
                }

            logging.info('Checking for the duplicate values in train and test dataset')
            train_duplicates = self.check_duplicates(train_df)
            test_duplicates = self.check_duplicates(test_df)
            if (train_duplicates == 0) and (test_duplicates == 0):
                validation_report['duplicates'] = True
                logging.info('Duplicate values are not present in the dataset')
            else:
                validation_report['duplicates'] = {
                    'train_duplicates' : train_duplicates,
                    'test_duplicates' : test_duplicates
                }

            logging.info('Target Validation on train and test data')
            train_target_exists = self.target_validation(train_df)
            test_target_exists = self.target_validation(test_df)
            if (train_target_exists==True) and (test_target_exists==True):
                validation_report['target_exists'] = True
                logging.info('Target variable is present in the dataset')
            else:
                validation_report['target_exists'] = {
                    'train_target_exists' : train_target_exists,
                    'test_target_exists' : test_target_exists
                }
            
            
            
        except Exception as e :
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)