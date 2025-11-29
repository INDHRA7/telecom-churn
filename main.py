from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import numpy as np
import pandas as pd
import joblib

app = FastAPI()

# Load Templates & Static
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load Models
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
columns = joblib.load("model/columns.pkl")     # Expected input columns after encoding


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    Contract: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...)
):

    # Collect user inputs
    input_data = {
        "Gender": Gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "Contract": Contract,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode
    df_processed = pd.get_dummies(df)

    # Ensure correct column order
    for col in columns:
        if col not in df_processed:
            df_processed[col] = 0

    df_processed = df_processed[columns]

    # Apply scaling
    scaled_data = scaler.transform(df_processed)

    # Predict
    prediction = model.predict(scaled_data)[0]
    result = "Customer Will Churn " if prediction == 1 else "Customer Will Not Churn "

    # Return to same HTML page
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result}
    )
