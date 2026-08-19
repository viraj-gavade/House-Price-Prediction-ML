from src.exception_handler import CustomMLException
from src.logger import logging
from dataclasses import dataclass
import os 
import sys 
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import field 
from pydantic import BaseModel
from typing import Dict , List
import json

class ValidationReportSchema(BaseModel):
   expected_cols : bool | Dict[str,List[str]]
   unexpected_cols : bool | Dict[str,List[str]]
   expected_dtype : bool | Dict[str,Dict[str,str]]
   missing_values : bool | Dict[str,Dict[str,str]]
   duplicates : bool | Dict[str,int]
   target_exists : bool | Dict[str,bool]
   category_exists : Dict[str,bool]
   status : str
    

class DataValidationConfig:
    validation_report_path : str = os.path.join('Reports','validation_report.json')
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
    

    def check_expected_cols(self ,df:pd.DataFrame)->list[str]:
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

    def check_unexpected_cols(self,df:pd.DataFrame)->list[str]:
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
             
    def category_validation(self, df: pd.DataFrame) -> dict:
        try:
            logging.info("Validating categorical columns")

            valid_categories = {
                "MS Zoning": ['RL', 'RH', 'FV', 'RM', 'C (all)', 'I (all)', 'A (agr)'],
                "Street": ['Pave', 'Grvl'],
                "Alley": ['Pave', 'Grvl'],
                "Lot Shape": ['IR1', 'Reg', 'IR2', 'IR3'],
                "Land Contour": ['Lvl', 'HLS', 'Bnk', 'Low'],
                "Utilities": ['AllPub', 'NoSewr', 'NoSeWa'],
                "Lot Config": ['Corner', 'Inside', 'CulDSac', 'FR2', 'FR3'],
                "Land Slope": ['Gtl', 'Mod', 'Sev'],
                "Neighborhood": [
                    'NAmes', 'Gilbert', 'StoneBr', 'NWAmes', 'Somerst',
                    'BrDale', 'NPkVill', 'NridgHt', 'Blmngtn', 'NoRidge',
                    'SawyerW', 'Sawyer', 'Greens', 'BrkSide', 'OldTown',
                    'IDOTRR', 'ClearCr', 'SWISU', 'Edwards', 'CollgCr',
                    'Crawfor', 'Blueste', 'Mitchel', 'Timber', 'MeadowV',
                    'Veenker', 'GrnHill', 'Landmrk'
                ],
                "Condition 1": [
                    'Norm', 'Feedr', 'PosN', 'RRNe', 'RRAe',
                    'Artery', 'PosA', 'RRAn', 'RRNn'
                ],
                "Condition 2": [
                    'Norm', 'Feedr', 'PosA', 'PosN', 'Artery',
                    'RRNn', 'RRAe', 'RRAn'
                ],
                "Bldg Type": ['1Fam', 'TwnhsE', 'Twnhs', 'Duplex', '2fmCon'],
                "House Style": [
                    '1Story', '2Story', '1.5Fin', 'SFoyer',
                    'SLvl', '2.5Unf', '1.5Unf', '2.5Fin'
                ],
                "Roof Style": ['Hip', 'Gable', 'Mansard', 'Gambrel', 'Shed', 'Flat'],
                "Roof Matl": [
                    'CompShg', 'WdShake', 'Tar&Grv', 'WdShngl',
                    'Membran', 'ClyTile', 'Roll', 'Metal'
                ],
                "Exterior 1st": [
                    'BrkFace', 'VinylSd', 'Wd Sdng', 'CemntBd',
                    'HdBoard', 'Plywood', 'MetalSd', 'AsbShng',
                    'WdShing', 'Stucco', 'AsphShn', 'BrkComm',
                    'CBlock', 'PreCast', 'Stone', 'ImStucc'
                ],
                "Exterior 2nd": [
                    'Plywood', 'VinylSd', 'Wd Sdng', 'BrkFace',
                    'CmentBd', 'HdBoard', 'Wd Shng', 'MetalSd',
                    'ImStucc', 'Brk Cmn', 'AsbShng', 'Stucco',
                    'AsphShn', 'CBlock', 'Stone', 'PreCast', 'Other'
                ],
                "Mas Vnr Type": ['Stone', 'BrkFace', 'BrkCmn', 'CBlock'],
                "Exter Qual": ['TA', 'Gd', 'Ex', 'Fa'],
                "Exter Cond": ['TA', 'Gd', 'Fa', 'Po', 'Ex'],
                "Foundation": ['CBlock', 'PConc', 'Wood', 'BrkTil', 'Slab', 'Stone'],
                "Bsmt Qual": ['TA', 'Gd', 'Ex', 'Fa', 'Po'],
                "Bsmt Cond": ['Gd', 'TA', 'Po', 'Fa', 'Ex'],
                "Bsmt Exposure": ['Gd', 'No', 'Mn', 'Av'],
                "BsmtFin Type 1": ['BLQ', 'Rec', 'ALQ', 'GLQ', 'Unf', 'LwQ'],
                "BsmtFin Type 2": ['Unf', 'LwQ', 'BLQ', 'Rec', 'GLQ', 'ALQ'],
                "Heating": ['GasA', 'GasW', 'Grav', 'Wall', 'Floor', 'OthW'],
                "Heating QC": ['Fa', 'TA', 'Ex', 'Gd', 'Po'],
                "Central Air": ['Y', 'N'],
                "Electrical": ['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'],
                "Kitchen Qual": ['TA', 'Gd', 'Ex', 'Fa', 'Po'],
                "Functional": [
                    'Typ', 'Mod', 'Min1', 'Min2',
                    'Maj1', 'Maj2', 'Sev', 'Sal'
                ],
                "Fireplace Qu": ['Gd', 'TA', 'Po', 'Ex', 'Fa'],
                "Garage Type": [
                    'Attchd', 'BuiltIn', 'Basment',
                    'Detchd', 'CarPort', '2Types'
                ],
                "Garage Finish": ['Fin', 'Unf', 'RFn'],
                "Garage Qual": ['TA', 'Fa', 'Gd', 'Ex', 'Po'],
                "Garage Cond": ['TA', 'Fa', 'Gd', 'Ex', 'Po'],
                "Paved Drive": ['P', 'Y', 'N'],
                "Pool QC": ['Ex', 'Gd', 'TA', 'Fa'],
                "Fence": ['MnPrv', 'GdPrv', 'GdWo', 'MnWw'],
                "Misc Feature": ['Gar2', 'Shed', 'Othr', 'Elev', 'TenC'],
                "Sale Type": [
                    'WD ', 'New', 'COD', 'ConLI', 'Con',
                    'ConLD', 'Oth', 'ConLw', 'CWD', 'VWD'
                ],
                "Sale Condition": [
                    'Normal', 'Partial', 'Family',
                    'Abnorml', 'Alloca', 'AdjLand'
                ]
            }

            errors = {}

            for column, valid_values in valid_categories.items():

                if column not in df.columns:
                    errors[column] = {
                        "error": "Column not found in dataframe"
                    }
                    continue

                actual_values = set(df[column].dropna().unique())
                valid_values_set = set(valid_values)

                invalid_values = actual_values - valid_values_set

                if invalid_values:
                    errors[column] = {
                        "error": "Invalid categories",
                        "invalid_values": list(invalid_values),
                        "valid_values": valid_values
                    }

            if errors:
                logging.info(f"Category validation failed: {errors}")
            else:
                logging.info("All categorical columns are valid")

            return errors

        except Exception as e:
            logging.info(f"Exception Occurred: {e}")
            raise CustomMLException(e, sys)
    
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
            
            logging.info('Checking for the Categorical values on train and test dataset')
            category_error_train = self.category_validation(train_df)
            category_error_test = self.category_validation(test_df)
            if (len(category_error_train) == 0) and (len(category_error_test) == 0):
                validation_report['category_exists'] = True
                logging.info('Categorical values are present in the dataset')
            else:
                validation_report['category_exists'] = {
                    'category_train_error' : category_error_train,
                    'category_test_error' : category_error_test
                }   


            logging.info('Creating the validation status')
            if(False in validation_report):
                validation_report['status'] = 'Failed'
                logging.info('Data validation failed')
            else:
                validation_report['status'] = 'Passed'
                logging.info('Data validation passed')

            logging.info('Creating the validation report schema')
            report = ValidationReportSchema(
                expected_cols = validation_report['expected_cols'],
                unexpected_cols = validation_report['unexpected_cols'],
                expected_dtype = validation_report['expected_dtype'],
                missing_values = validation_report['missing_values'],
                duplicates = validation_report['duplicates'],
                target_exists = validation_report['target_exists'],
                category_exists = validation_report['category_exists'],
                status = validation_report['status']
            )
            logging.info('Validation report created successfully')
            logging.info(f'Saving report to {self.validation_report_path}')
            with open(self.validation_report_path,'w') as f:
                json.dump(report.dict(),f)
            logging.info('Report saved successfully')
            
            return report
        except Exception as e :
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)