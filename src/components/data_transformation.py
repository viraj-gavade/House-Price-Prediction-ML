from src.exception_handler import CustomMLException
from src.utils import save_object
from src.logger import logging
from dataclasses import dataclass , field
import os 
import sys 
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    preprocessor_path : str = os.path.join('Artifacts/Encoders','preprocessor.pkl')
    target_col : str = 'SalePrice'
    columns_to_drop : list = field(default_factory=lambda:['Order','PID'])



class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def create_features(self , df:pd.DataFrame)->pd.DataFrame:
        try:
            logging.info('Creating the new features')
            df = df.copy()
            logging.info('Creating Area Feature')
            df["TotalSF"] = (
                                df["Total Bsmt SF"].fillna(0)
                                + df["1st Flr SF"].fillna(0)
                                + df["2nd Flr SF"].fillna(0)
                            )

            logging.info('Creating the total bathrooms feature')
            df["TotalBathrooms"] = (
                                    df["Full Bath"].fillna(0)
                                    + 0.5 * df["Half Bath"].fillna(0)
                                    + df["Bsmt Full Bath"].fillna(0)
                                    + 0.5 * df["Bsmt Half Bath"].fillna(0)
                                )


            logging.info('Creating the total porch area')
            df["TotalPorchSF"] = (
                    df["Open Porch SF"].fillna(0)
                    + df["3Ssn Porch"].fillna(0)
                    + df["Enclosed Porch"].fillna(0)
                    + df["Screen Porch"].fillna(0)
                    + df["Wood Deck SF"].fillna(0)
                )

            logging.info('Creating the House Age Feature')
            df["HouseAge"] = (
                        df["Yr Sold"]
                        - df["Year Built"]
                    )

            logging.info('Creating the Remodling age feature')  
            df["RemodAge"] = (
                            df["Yr Sold"]
                            - df["Year Remod/Add"]
                        )  
            return df       

        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)

    def initiate_data_transformation(self , train_path : str , test_path : str):
        try:
            logging.info('** PHASE 3 : Initiating the Data Tranformation Pipeline **')
            logging.info('Reading the train and test dataset')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test dataset read successfully')

            logging.info('Separating features and target')
            X_train = train_df.drop(self.data_tranformation_config.target_col,axis=1)
            X_test = test_df.drop(self.data_tranformation_config.target_col,axis=1)
            y_train = train_df[self.data_tranformation_config.target_col]
            y_test = test_df[self.data_tranformation_config.target_col]
            logging.info('Separating features and target Completed successfully')
            logging.info(f'X_train : {X_train.shape} , X_test : {X_test.shape} , y_train : {y_train.shape} , y_test : {y_test.shape} ')

            logging.info(f'Dropping the Unnecessary Columns : {self.data_tranformation_config.columns_to_drop}')
            X_train.drop(columns=self.data_tranformation_config.columns_to_drop,axis=1,inplace=True)
            X_test.drop(columns=self.data_tranformation_config.columns_to_drop,axis=1,inplace=True)
            logging.info('Unnecessary Columns dropped successfully')
            logging.info(f'X_train : {X_train.shape} , X_test : {X_test.shape} , y_train : {y_train.shape} , y_test : {y_test.shape} ')

            
            logging.info('Creating the new features with feature engineering')
            X_train = self.create_features(X_train)
            X_test = self.create_features(X_test)
            logging.info(f'Raw Features : { len(X_train)}')
            logging.info('Features created successsfully')

            logging.info('Identifying the categorical and numerical columns')
            numerical_features = X_train.select_dtypes(
                        include=["int64", "float64"]
                    ).columns.tolist()

            categorical_features = X_train.select_dtypes(
                        include=["object"]
                    ).columns.tolist()

            logging.info(f'Numerical: {len(numerical_features)} ')
            logging.info(f'Categorical: {len(categorical_features)} ')


            logging.info('Creating numerical tranformation pipeline using SimpleImputer and StandardScalar')
            numerical_pipeline = Pipeline(
                        steps=[
                            (
                                "imputer",
                                SimpleImputer(strategy="median")
                            ),
                            (
                                "scaler",
                                StandardScaler()
                            )
                        ]
                    )

            logging.info('Creating categorical tranformation pipeline using SimpleImputer and OneHotEncoder')
            categorical_pipeline = Pipeline(
                                steps=[
                                    (
                                        "imputer",
                                        SimpleImputer(strategy="most_frequent")
                                    ),
                                    (
                                        "encoder",
                                        OneHotEncoder(
                                            handle_unknown="ignore",
                                            sparse_output=False
                                        )
                                    )
                                ]
                            )

            logging.info('Applying the column transformer to transform the data')
            preprocessor = ColumnTransformer(
                        transformers=[
                            (
                                "numerical",
                                numerical_pipeline,
                                numerical_features
                            ),
                            (
                                "categorical",
                                categorical_pipeline,
                                categorical_features
                            )
                        ]
                    )   

            logging.info('Fitting the column transformer')
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info('Saving the column transfomer')
            save_object(self.data_tranformation_config.preprocessor_path,preprocessor)
            X_test_transformed = preprocessor.transform(X_test)
            logging.info('Trainformation applied successfully')

            y_train_transformed = y_train
            y_test_transformed = y_test

            logging.info('** PHASE 3 : Data Tranformation Pipeline Completed Successfully! **')

            return (
                X_train_transformed,X_test_transformed,y_train_transformed,y_test_transformed,self.data_tranformation_config.preprocessor_path
            )



        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)
