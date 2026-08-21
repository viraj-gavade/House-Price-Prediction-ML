from fastapi import FastAPI
from pydantic import BaseModel

from src.components.pipeline.prediction_pipeline import (
    CustomData,
    InferencePipeline
)


app = FastAPI(
    title="House Price Prediction API",
    description="API for predicting house prices",
    version="1.0.0"
)


class HousePredictionRequest(BaseModel):

    MS_SubClass: int
    MS_Zoning: str
    Lot_Frontage: float | None = None
    Lot_Area: int
    Street: str
    Alley: str | None = None
    Lot_Shape: str
    Land_Contour: str
    Utilities: str
    Lot_Config: str
    Land_Slope: str
    Neighborhood: str
    Condition_1: str
    Condition_2: str
    Bldg_Type: str
    House_Style: str
    Overall_Qual: int
    Overall_Cond: int
    Year_Built: int
    Year_Remod_Add: int
    Roof_Style: str
    Roof_Matl: str
    Exterior_1st: str
    Exterior_2nd: str
    Mas_Vnr_Type: str | None = None
    Mas_Vnr_Area: float | None = None
    Exter_Qual: str
    Exter_Cond: str
    Foundation: str
    Bsmt_Qual: str | None = None
    Bsmt_Cond: str | None = None
    Bsmt_Exposure: str | None = None
    BsmtFin_Type_1: str | None = None
    BsmtFin_SF_1: float | None = None
    BsmtFin_Type_2: str | None = None
    BsmtFin_SF_2: float | None = None
    Bsmt_Unf_SF: float | None = None
    Total_Bsmt_SF: float | None = None
    Heating: str
    Heating_QC: str
    Central_Air: str
    Electrical: str | None = None
    First_Flr_SF: int
    Second_Flr_SF: int
    Low_Qual_Fin_SF: int
    Gr_Liv_Area: int
    Bsmt_Full_Bath: float | None = None
    Bsmt_Half_Bath: float | None = None
    Full_Bath: int
    Half_Bath: int
    Bedroom_AbvGr: int
    Kitchen_AbvGr: int
    Kitchen_Qual: str
    TotRms_AbvGrd: int
    Functional: str
    Fireplaces: int
    Fireplace_Qu: str | None = None
    Garage_Type: str | None = None
    Garage_Yr_Blt: float | None = None
    Garage_Finish: str | None = None
    Garage_Cars: float | None = None
    Garage_Area: float | None = None
    Garage_Qual: str | None = None
    Garage_Cond: str | None = None
    Paved_Drive: str
    Wood_Deck_SF: int
    Open_Porch_SF: int
    Enclosed_Porch: int
    Three_Ssn_Porch: int
    Screen_Porch: int
    Pool_Area: int
    Pool_QC: str | None = None
    Fence: str | None = None
    Misc_Feature: str | None = None
    Misc_Val: int
    Mo_Sold: int
    Yr_Sold: int
    Sale_Type: str
    Sale_Condition: str


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(data: HousePredictionRequest):

    custom_data = CustomData(
        MS_SubClass=data.MS_SubClass,
        MS_Zoning=data.MS_Zoning,
        Lot_Frontage=data.Lot_Frontage,
        Lot_Area=data.Lot_Area,
        Street=data.Street,
        Alley=data.Alley,
        Lot_Shape=data.Lot_Shape,
        Land_Contour=data.Land_Contour,
        Utilities=data.Utilities,
        Lot_Config=data.Lot_Config,
        Land_Slope=data.Land_Slope,
        Neighborhood=data.Neighborhood,
        Condition_1=data.Condition_1,
        Condition_2=data.Condition_2,
        Bldg_Type=data.Bldg_Type,
        House_Style=data.House_Style,
        Overall_Qual=data.Overall_Qual,
        Overall_Cond=data.Overall_Cond,
        Year_Built=data.Year_Built,
        Year_Remod_Add=data.Year_Remod_Add,
        Roof_Style=data.Roof_Style,
        Roof_Matl=data.Roof_Matl,
        Exterior_1st=data.Exterior_1st,
        Exterior_2nd=data.Exterior_2nd,
        Mas_Vnr_Type=data.Mas_Vnr_Type,
        Mas_Vnr_Area=data.Mas_Vnr_Area,
        Exter_Qual=data.Exter_Qual,
        Exter_Cond=data.Exter_Cond,
        Foundation=data.Foundation,
        Bsmt_Qual=data.Bsmt_Qual,
        Bsmt_Cond=data.Bsmt_Cond,
        Bsmt_Exposure=data.Bsmt_Exposure,
        BsmtFin_Type_1=data.BsmtFin_Type_1,
        BsmtFin_SF_1=data.BsmtFin_SF_1,
        BsmtFin_Type_2=data.BsmtFin_Type_2,
        BsmtFin_SF_2=data.BsmtFin_SF_2,
        Bsmt_Unf_SF=data.Bsmt_Unf_SF,
        Total_Bsmt_SF=data.Total_Bsmt_SF,
        Heating=data.Heating,
        Heating_QC=data.Heating_QC,
        Central_Air=data.Central_Air,
        Electrical=data.Electrical,
        First_Flr_SF=data.First_Flr_SF,
        Second_Flr_SF=data.Second_Flr_SF,
        Low_Qual_Fin_SF=data.Low_Qual_Fin_SF,
        Gr_Liv_Area=data.Gr_Liv_Area,
        Bsmt_Full_Bath=data.Bsmt_Full_Bath,
        Bsmt_Half_Bath=data.Bsmt_Half_Bath,
        Full_Bath=data.Full_Bath,
        Half_Bath=data.Half_Bath,
        Bedroom_AbvGr=data.Bedroom_AbvGr,
        Kitchen_AbvGr=data.Kitchen_AbvGr,
        Kitchen_Qual=data.Kitchen_Qual,
        TotRms_AbvGrd=data.TotRms_AbvGrd,
        Functional=data.Functional,
        Fireplaces=data.Fireplaces,
        Fireplace_Qu=data.Fireplace_Qu,
        Garage_Type=data.Garage_Type,
        Garage_Yr_Blt=data.Garage_Yr_Blt,
        Garage_Finish=data.Garage_Finish,
        Garage_Cars=data.Garage_Cars,
        Garage_Area=data.Garage_Area,
        Garage_Qual=data.Garage_Qual,
        Garage_Cond=data.Garage_Cond,
        Paved_Drive=data.Paved_Drive,
        Wood_Deck_SF=data.Wood_Deck_SF,
        Open_Porch_SF=data.Open_Porch_SF,
        Enclosed_Porch=data.Enclosed_Porch,
        Three_Ssn_Porch=data.Three_Ssn_Porch,
        Screen_Porch=data.Screen_Porch,
        Pool_Area=data.Pool_Area,
        Pool_QC=data.Pool_QC,
        Fence=data.Fence,
        Misc_Feature=data.Misc_Feature,
        Misc_Val=data.Misc_Val,
        Mo_Sold=data.Mo_Sold,
        Yr_Sold=data.Yr_Sold,
        Sale_Type=data.Sale_Type,
        Sale_Condition=data.Sale_Condition
    )

    features = custom_data.get_data_as_data_frame()

    prediction_pipeline = InferencePipeline()

    prediction = prediction_pipeline.prediction(features)

    return {
        "prediction": float(prediction[0])
    }