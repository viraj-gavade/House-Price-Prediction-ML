from src.exception_handler import CustomMLException
from src.logger import logging
from dataclasses import dataclass , field
import sys
import os 
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# from xgboost import XGBRegressor
from src.utils import save_object
from sklearn.model_selection import RandomizedSearchCV
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("House Price Prediction")
@dataclass
class ModelTrainingConfig:
    best_base_model_path : str = os.path.join('Artifacts/Models','best_base_model.pkl')
    best_tunned_model_path : str = os.path.join('Artifacts/Models','best_tunned_model.pkl')
    models:dict = field(default_factory=lambda:{
          "Linear Regression": LinearRegression(),
        
            "Ridge": Ridge(
                alpha=1.0
            ),
        
            "Lasso": Lasso(
                alpha=0.001,
                max_iter=10000
            ),
        
            # "Decision Tree": DecisionTreeRegressor(
            #     random_state=42
            # ),
        
            # "Random Forest": RandomForestRegressor(
            #     n_estimators=200,
            #     random_state=42,
            # ),
        
            # "Gradient Boosting": GradientBoostingRegressor(
            #     random_state=42
            # ),
        
            # "XGBoost": XGBRegressor(
            #     n_estimators=200,
            #     learning_rate=0.05,
            #     max_depth=5,
            #     random_state=42,
            #     objective="reg:squarederror"
            # )
    })

import mlflow
from mlflow.tracking import MlflowClient


def is_model_better(r2_score: float, model_name: str) -> bool:
    """
    Compare the new model's R2 score with the currently
    registered model's R2 score.

    Returns:
        True  -> new model is better
        False -> existing model is better/equal
    """

    client = MlflowClient()

    try:
        # Get all versions of the registered model
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        if not versions:
            # No registered model exists yet
            return True

        # Get the latest registered model version
        latest_version = max(
            versions,
            key=lambda version: int(version.version)
        )

        # Get the MLflow run associated with that model version
        run = client.get_run(latest_version.run_id)

        best_r2_score = run.data.metrics.get("r2_score")

        if best_r2_score is None:
            # Existing model doesn't have R2 recorded
            return True

        print(f"Existing R2 : {best_r2_score}")
        print(f"New R2      : {r2_score}")

        if r2_score > best_r2_score:
            print("New model is better.")
            return True

        print("Existing model is better or equal.")
        return False

    except Exception as e:
        print(f"Error while comparing models: {e}")
        return True

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    
    
    def hyperparamter_tune_base_model(self,X_train_transformed:np.ndarray,y_train_transformed:np.ndarray,best_base_model_name : str ):
        try:
            param_grid = {}
            logging.info(f'Hyperparamter tunning the best base  model : {best_base_model_name}')
            model = self.model_trainer_config.models[best_base_model_name]
            if best_base_model_name == "Linear Regression":
                param_grid = {
                    "fit_intercept": [True, False]
                }

            elif best_base_model_name == "Ridge":
                param_grid = {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    "fit_intercept": [True, False],
                    "solver": ["auto", "svd", "cholesky", "lsqr"]
                }

            elif best_base_model_name == "Lasso":
                param_grid = {
                    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                    "fit_intercept": [True, False],
                    "max_iter": [5000, 10000, 20000],
                    "selection": ["cyclic", "random"]
                }

            elif best_base_model_name == "Decision Tree":
                param_grid = {
                    "max_depth": [None, 3, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": [None, "sqrt", "log2"]
                }

            elif best_base_model_name == "Random Forest":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 15, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                }

            elif best_base_model_name == "Gradient Boosting":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [2, 3, 5],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "subsample": [0.8, 1.0]
                }

            elif best_base_model_name == "XGBoost":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                    "min_child_weight": [1, 3, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            logging.info('Using the RandomizedSearchCV to tune the model')
            clf = RandomizedSearchCV(model,param_distributions=param_grid,random_state=42,cv=2,n_iter=5,scoring='r2')
            search = clf.fit(X_train_transformed,y_train_transformed)
            best_tunned_score = search.best_score_
            best_tunned_model = search.best_estimator_
            best_tunned_params = search.best_params_
            logging.info(f'Best Tunned Model : {best_tunned_model}')
            logging.info(f'Best Tunned Model Score : {best_tunned_score}')
            logging.info(f'Best Tunned Model Params : {best_tunned_params}')

            logging.info('Saving the best tunned base model')
            save_object(self.model_trainer_config.best_tunned_model_path,best_tunned_model)
            logging.info('Tunned model saved successfully')
            return(
                best_tunned_score,
                best_tunned_model
            )

        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)

    def train_and_evaluate_model(self , X_train_transformed:np.ndarray,X_test_transformed:np.ndarray,y_train_transformed:np.ndarray,y_test_transformed:np.ndarray,models:dict)->dict:
        try:
            logging.info('Training the multiple models and choosing the best model')
            model_report = {}
            for model_name , model in models.items():
                model = model
                logging.info(f'***** Training : {model_name} ******')
                logging.info(f'X_train_transformed : {X_train_transformed.shape} ,X_test_transformed : {X_test_transformed.shape} ')
                logging.info(f'y_train_transformed : {y_train_transformed.shape} ,y_test_transformed : {y_test_transformed.shape} ')

                logging.info(f'Fitting the {model_name}')
                model.fit(X_train_transformed,y_train_transformed)
                logging.info('Model fitted successfully')

                logging.info('Doing predictions with the fitted model')
                y_pred_train = model.predict(X_train_transformed)
                y_pred_test = model.predict(X_test_transformed)

                logging.info('Calculating the evalution metrics')
                mean_absolute_error_train = mean_absolute_error(y_train_transformed,y_pred_train)
                mean_squared_error_train = mean_squared_error(y_train_transformed,y_pred_train)
                r2_score_train = r2_score(y_train_transformed,y_pred_train)

                mean_absolute_error_test = mean_absolute_error(y_test_transformed,y_pred_test)
                mean_squared_error_test = mean_squared_error(y_test_transformed,y_pred_test)
                r2_score_test = r2_score(y_test_transformed,y_pred_test)

                logging.info("Evaluation Metrics Calulated Sucessfully")

                logging.info('Tracking the experiment with the mlflow')
                with mlflow.start_run(run_name=model_name) as run:
                    mlflow.log_params(
                        {
                            'model_name':model_name,
                            'Train Sample':len(X_train_transformed),
                            'Test Sample':len(X_test_transformed),
                            'n_features': X_train_transformed.shape[1]
                        }
                    )
                    mlflow.log_metrics({
                        'mean_absolute_error_train':mean_absolute_error_train,
                        'mean_squared_error_train':mean_squared_error_train,
                        'r2_score_train':r2_score_train,
                        'mean_absolute_error_test':mean_absolute_error_test,
                        'mean_squared_error_test':mean_squared_error_test,
                        'r2_score_test':r2_score_test

                    })
                    model_info = mlflow.sklearn.log_model(
                        model,
                        name=model_name,
                        serialization_format='pickle'
                    )
                model_report[model_name] = {
                    'mean_absolute_error_train':mean_absolute_error_train,
                    'mean_squared_error_train':mean_squared_error_train,
                    'r2_score_train':r2_score_train,
                    'mean_absolute_error_test':mean_absolute_error_test,
                    'mean_squared_error_test':mean_squared_error_test,
                    'r2_score_test':r2_score_test
                }
            logging.info('All Models trained successfully')
            return model_report    


        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)


    def initiate_model_training(self,X_train_transformed:np.ndarray,X_test_transformed:np.ndarray,y_train_transformed:np.ndarray,y_test_transformed:np.ndarray):
        try:
            logging.info('** PHASE 4 : Initiating the Model Trainer Pipeline **')
            model_report: dict = self.train_and_evaluate_model(X_train_transformed,X_test_transformed,y_train_transformed,y_test_transformed,self.model_trainer_config.models)
            best_base_model_name = None
            best_base_model_r2_score = 0
            best_base_model = None
            for model_name in model_report:
                current_r2_score = model_report[model_name]['r2_score_test']
                if  current_r2_score > best_base_model_r2_score:
                    best_base_model = self.model_trainer_config.models[model_name]
                    best_base_model_name = model_name
                    best_base_model_r2_score = current_r2_score 

            logging.info(f'Best Base Model :  {best_base_model}')
            logging.info(f'Type of Best Base Model :  , {type(best_base_model)}')
            logging.info(f'Best Base Model Name :  , {best_base_model_name}')
            logging.info(f'Best Base Model Score :  , {best_base_model_r2_score}')


            logging.info(f'Saving the best base model with r2 score : {best_base_model_r2_score} ')
            save_object(self.model_trainer_config.best_base_model_path,best_base_model)


            logging.info('Initiating the hyperparamter tunning')
            best_tunned_score, best_tunned_model = self.hyperparamter_tune_base_model(X_train_transformed,y_train_transformed,best_base_model_name)

            logging.info('Checking if tunned model is better than prod model')
            if(best_tunned_score > best_base_model_r2_score):
                logging.info('Tunned model is better than prod model registering it')
                model_info = mlflow.sklearn.log_model(
                        best_tunned_model,
                        name=model_name,
                        serialization_format='pickle'
                    )
                is_model_better(best_tunned_score,best_base_model_name)
                mlflow.register_model(
                                             model_uri=model_info.model_uri,
                                             name='HousePriceModel'
                                        )
                
                logging.info('Tunned Model registered on prod successfully')
            else:
                logging.info('Based model is better than tunned model registering it')
                model_info = mlflow.sklearn.log_model(
                        best_base_model,
                        name=model_name,
                        serialization_format='pickle'
                    )
                logging.info('Base model registered on prod successfully')

            logging.info('** PHASE 4 : Model Trainer Pipeline Completed Successfully! **')
        except Exception as e:
            logging.info(f'Error Occured : {e}')
            raise CustomMLException(e,sys)