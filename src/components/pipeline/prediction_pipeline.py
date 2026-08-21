from src.exception_handler import CustomMLException
from src.logger import logging
from src.utils import load_object

import os
import sys
import pandas as pd


class InferencePipeline:

    def prediction(self, features):

        try:
            logging.info(
                "*************** INITIATING THE INFERENCE PIPELINE ***************"
            )

            base_dir = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(
                            os.path.abspath(__file__)
                        )
                    )
                )
            )

            model_path = os.path.join(
                base_dir,
                "Artifacts",
                "Models",
                "best_base_model.pkl"
            )

            preprocessor_path = os.path.join(
                base_dir,
                "Artifacts",
                "Encoders",
                "preprocessor.pkl"
            )

            logging.info("Loading the model")
            model = load_object(model_path)
            logging.info("Model loaded successfully")

            logging.info("Loading the preprocessor")
            preprocessor = load_object(preprocessor_path)
            logging.info("Preprocessor loaded successfully")

            logging.info("Transforming the features")
            features_transformed = preprocessor.transform(features)
            logging.info("Features transformed successfully")

            logging.info("Predicting the result")
            prediction = model.predict(features_transformed)
            logging.info("Result predicted successfully")

            return prediction

        except Exception as e:
            logging.error(f"Error occurred during inference: {e}")
            raise CustomMLException(e, sys)


class CustomData:

    def __init__(
        self,
        MS_SubClass: int,
        MS_Zoning: str,
        Lot_Frontage: float,
        Lot_Area: int,
        Street: str,
        Alley: str,
        Lot_Shape: str,
        Land_Contour: str,
        Utilities: str,
        Lot_Config: str,
        Land_Slope: str,
        Neighborhood: str,
        Condition_1: str,
        Condition_2: str,
        Bldg_Type: str,
        House_Style: str,
        Overall_Qual: int,
        Overall_Cond: int,
        Year_Built: int,
        Year_Remod_Add: int,
        Roof_Style: str,
        Roof_Matl: str,
        Exterior_1st: str,
        Exterior_2nd: str,
        Mas_Vnr_Type: str,
        Mas_Vnr_Area: float,
        Exter_Qual: str,
        Exter_Cond: str,
        Foundation: str,
        Bsmt_Qual: str,
        Bsmt_Cond: str,
        Bsmt_Exposure: str,
        BsmtFin_Type_1: str,
        BsmtFin_SF_1: float,
        BsmtFin_Type_2: str,
        BsmtFin_SF_2: float,
        Bsmt_Unf_SF: float,
        Total_Bsmt_SF: float,
        Heating: str,
        Heating_QC: str,
        Central_Air: str,
        Electrical: str,
        First_Flr_SF: int,
        Second_Flr_SF: int,
        Low_Qual_Fin_SF: int,
        Gr_Liv_Area: int,
        Bsmt_Full_Bath: float,
        Bsmt_Half_Bath: float,
        Full_Bath: int,
        Half_Bath: int,
        Bedroom_AbvGr: int,
        Kitchen_AbvGr: int,
        Kitchen_Qual: str,
        TotRms_AbvGrd: int,
        Functional: str,
        Fireplaces: int,
        Fireplace_Qu: str,
        Garage_Type: str,
        Garage_Yr_Blt: float,
        Garage_Finish: str,
        Garage_Cars: float,
        Garage_Area: float,
        Garage_Qual: str,
        Garage_Cond: str,
        Paved_Drive: str,
        Wood_Deck_SF: int,
        Open_Porch_SF: int,
        Enclosed_Porch: int,
        Three_Ssn_Porch: int,
        Screen_Porch: int,
        Pool_Area: int,
        Pool_QC: str,
        Fence: str,
        Misc_Feature: str,
        Misc_Val: int,
        Mo_Sold: int,
        Yr_Sold: int,
        Sale_Type: str,
        Sale_Condition: str
    ):
        self.MS_SubClass = MS_SubClass
        self.MS_Zoning = MS_Zoning
        self.Lot_Frontage = Lot_Frontage
        self.Lot_Area = Lot_Area
        self.Street = Street
        self.Alley = Alley
        self.Lot_Shape = Lot_Shape
        self.Land_Contour = Land_Contour
        self.Utilities = Utilities
        self.Lot_Config = Lot_Config
        self.Land_Slope = Land_Slope
        self.Neighborhood = Neighborhood
        self.Condition_1 = Condition_1
        self.Condition_2 = Condition_2
        self.Bldg_Type = Bldg_Type
        self.House_Style = House_Style
        self.Overall_Qual = Overall_Qual
        self.Overall_Cond = Overall_Cond
        self.Year_Built = Year_Built
        self.Year_Remod_Add = Year_Remod_Add
        self.Roof_Style = Roof_Style
        self.Roof_Matl = Roof_Matl
        self.Exterior_1st = Exterior_1st
        self.Exterior_2nd = Exterior_2nd
        self.Mas_Vnr_Type = Mas_Vnr_Type
        self.Mas_Vnr_Area = Mas_Vnr_Area
        self.Exter_Qual = Exter_Qual
        self.Exter_Cond = Exter_Cond
        self.Foundation = Foundation
        self.Bsmt_Qual = Bsmt_Qual
        self.Bsmt_Cond = Bsmt_Cond
        self.Bsmt_Exposure = Bsmt_Exposure
        self.BsmtFin_Type_1 = BsmtFin_Type_1
        self.BsmtFin_SF_1 = BsmtFin_SF_1
        self.BsmtFin_Type_2 = BsmtFin_Type_2
        self.BsmtFin_SF_2 = BsmtFin_SF_2
        self.Bsmt_Unf_SF = Bsmt_Unf_SF
        self.Total_Bsmt_SF = Total_Bsmt_SF
        self.Heating = Heating
        self.Heating_QC = Heating_QC
        self.Central_Air = Central_Air
        self.Electrical = Electrical
        self.First_Flr_SF = First_Flr_SF
        self.Second_Flr_SF = Second_Flr_SF
        self.Low_Qual_Fin_SF = Low_Qual_Fin_SF
        self.Gr_Liv_Area = Gr_Liv_Area
        self.Bsmt_Full_Bath = Bsmt_Full_Bath
        self.Bsmt_Half_Bath = Bsmt_Half_Bath
        self.Full_Bath = Full_Bath
        self.Half_Bath = Half_Bath
        self.Bedroom_AbvGr = Bedroom_AbvGr
        self.Kitchen_AbvGr = Kitchen_AbvGr
        self.Kitchen_Qual = Kitchen_Qual
        self.TotRms_AbvGrd = TotRms_AbvGrd
        self.Functional = Functional
        self.Fireplaces = Fireplaces
        self.Fireplace_Qu = Fireplace_Qu
        self.Garage_Type = Garage_Type
        self.Garage_Yr_Blt = Garage_Yr_Blt
        self.Garage_Finish = Garage_Finish
        self.Garage_Cars = Garage_Cars
        self.Garage_Area = Garage_Area
        self.Garage_Qual = Garage_Qual
        self.Garage_Cond = Garage_Cond
        self.Paved_Drive = Paved_Drive
        self.Wood_Deck_SF = Wood_Deck_SF
        self.Open_Porch_SF = Open_Porch_SF
        self.Enclosed_Porch = Enclosed_Porch
        self.Three_Ssn_Porch = Three_Ssn_Porch
        self.Screen_Porch = Screen_Porch
        self.Pool_Area = Pool_Area
        self.Pool_QC = Pool_QC
        self.Fence = Fence
        self.Misc_Feature = Misc_Feature
        self.Misc_Val = Misc_Val
        self.Mo_Sold = Mo_Sold
        self.Yr_Sold = Yr_Sold
        self.Sale_Type = Sale_Type
        self.Sale_Condition = Sale_Condition

    def get_data_as_data_frame(self):

        try:
            data = {
                "MS SubClass": [self.MS_SubClass],
                "MS Zoning": [self.MS_Zoning],
                "Lot Frontage": [self.Lot_Frontage],
                "Lot Area": [self.Lot_Area],
                "Street": [self.Street],
                "Alley": [self.Alley],
                "Lot Shape": [self.Lot_Shape],
                "Land Contour": [self.Land_Contour],
                "Utilities": [self.Utilities],
                "Lot Config": [self.Lot_Config],
                "Land Slope": [self.Land_Slope],
                "Neighborhood": [self.Neighborhood],
                "Condition 1": [self.Condition_1],
                "Condition 2": [self.Condition_2],
                "Bldg Type": [self.Bldg_Type],
                "House Style": [self.House_Style],
                "Overall Qual": [self.Overall_Qual],
                "Overall Cond": [self.Overall_Cond],
                "Year Built": [self.Year_Built],
                "Year Remod/Add": [self.Year_Remod_Add],
                "Roof Style": [self.Roof_Style],
                "Roof Matl": [self.Roof_Matl],
                "Exterior 1st": [self.Exterior_1st],
                "Exterior 2nd": [self.Exterior_2nd],
                "Mas Vnr Type": [self.Mas_Vnr_Type],
                "Mas Vnr Area": [self.Mas_Vnr_Area],
                "Exter Qual": [self.Exter_Qual],
                "Exter Cond": [self.Exter_Cond],
                "Foundation": [self.Foundation],
                "Bsmt Qual": [self.Bsmt_Qual],
                "Bsmt Cond": [self.Bsmt_Cond],
                "Bsmt Exposure": [self.Bsmt_Exposure],
                "BsmtFin Type 1": [self.BsmtFin_Type_1],
                "BsmtFin SF 1": [self.BsmtFin_SF_1],
                "BsmtFin Type 2": [self.BsmtFin_Type_2],
                "BsmtFin SF 2": [self.BsmtFin_SF_2],
                "Bsmt Unf SF": [self.Bsmt_Unf_SF],
                "Total Bsmt SF": [self.Total_Bsmt_SF],
                "Heating": [self.Heating],
                "Heating QC": [self.Heating_QC],
                "Central Air": [self.Central_Air],
                "Electrical": [self.Electrical],
                "1st Flr SF": [self.First_Flr_SF],
                "2nd Flr SF": [self.Second_Flr_SF],
                "Low Qual Fin SF": [self.Low_Qual_Fin_SF],
                "Gr Liv Area": [self.Gr_Liv_Area],
                "Bsmt Full Bath": [self.Bsmt_Full_Bath],
                "Bsmt Half Bath": [self.Bsmt_Half_Bath],
                "Full Bath": [self.Full_Bath],
                "Half Bath": [self.Half_Bath],
                "Bedroom AbvGr": [self.Bedroom_AbvGr],
                "Kitchen AbvGr": [self.Kitchen_AbvGr],
                "Kitchen Qual": [self.Kitchen_Qual],
                "TotRms AbvGrd": [self.TotRms_AbvGrd],
                "Functional": [self.Functional],
                "Fireplaces": [self.Fireplaces],
                "Fireplace Qu": [self.Fireplace_Qu],
                "Garage Type": [self.Garage_Type],
                "Garage Yr Blt": [self.Garage_Yr_Blt],
                "Garage Finish": [self.Garage_Finish],
                "Garage Cars": [self.Garage_Cars],
                "Garage Area": [self.Garage_Area],
                "Garage Qual": [self.Garage_Qual],
                "Garage Cond": [self.Garage_Cond],
                "Paved Drive": [self.Paved_Drive],
                "Wood Deck SF": [self.Wood_Deck_SF],
                "Open Porch SF": [self.Open_Porch_SF],
                "Enclosed Porch": [self.Enclosed_Porch],
                "3Ssn Porch": [self.Three_Ssn_Porch],
                "Screen Porch": [self.Screen_Porch],
                "Pool Area": [self.Pool_Area],
                "Pool QC": [self.Pool_QC],
                "Fence": [self.Fence],
                "Misc Feature": [self.Misc_Feature],
                "Misc Val": [self.Misc_Val],
                "Mo Sold": [self.Mo_Sold],
                "Yr Sold": [self.Yr_Sold],
                "Sale Type": [self.Sale_Type],
                "Sale Condition": [self.Sale_Condition]
            }

            df = pd.DataFrame(data)

            df["TotalSF"] = (
                df["Total Bsmt SF"].fillna(0)
                + df["1st Flr SF"].fillna(0)
                + df["2nd Flr SF"].fillna(0)
            )

            df["TotalBathrooms"] = (
                df["Full Bath"].fillna(0)
                + 0.5 * df["Half Bath"].fillna(0)
                + df["Bsmt Full Bath"].fillna(0)
                + 0.5 * df["Bsmt Half Bath"].fillna(0)
            )

            df["TotalPorchSF"] = (
                df["Open Porch SF"].fillna(0)
                + df["3Ssn Porch"].fillna(0)
                + df["Enclosed Porch"].fillna(0)
                + df["Screen Porch"].fillna(0)
                + df["Wood Deck SF"].fillna(0)
            )

            df["HouseAge"] = df["Yr Sold"] - df["Year Built"]

            df["RemodAge"] = df["Yr Sold"] - df["Year Remod/Add"]

            return df

        except Exception as e:
            logging.error(f"Error while creating dataframe: {e}")
            raise CustomMLException(e, sys)