from fastapi import FastAPI
from pydantic import BaseModel
from app.model.predict import predict_credit_risk

app = FastAPI(title="Credit Risk Prediction API")


class CreditInput(BaseModel):

    Age: int
    Sex: str
    Job: int
    Housing: str
    Saving_accounts: str
    Checking_account: str
    Credit_amount: int
    Duration: int
    Purpose: str


@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}


@app.post("/predict")
def predict(data: CreditInput):

    result = predict_credit_risk(data.dict())

    return result