from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import uvicorn

# Load model, scaler, and feature template
with open("frauddetection_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("template_columns.pkl", "rb") as f:
    template_columns = pickle.load(f)

app = FastAPI(title="Fraud Detection Service")

# Define the input schema
class TransactionInput(BaseModel):
    cc_num: int
    merchant: str
    category: str
    amt: float
    city_pop: int
    state: str

def preprocess_input(data, template_columns, scaler):
    df_input = pd.DataFrame([data])
    df_input['trans_hour'] = 12
    df_input['trans_day'] = 15
    df_input['trans_month'] = 6
    df_input['gender'] = 'Male'
    df_input['age_bin'] = '30-39'
    df_input['distance'] = 10

    # One-hot encode
    df_input = pd.get_dummies(df_input, columns=['category', 'state', 'gender', 'age_bin'], drop_first=True)

    # Align with training features
    for col in template_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[template_columns]

    # Scale numeric values
    numeric_cols = ['amt', 'city_pop', 'distance', 'trans_hour', 'trans_day', 'trans_month']
    df_input[numeric_cols] = scaler.transform(df_input[numeric_cols])

    return df_input

@app.post("/predict")
def predict(input_data: TransactionInput):
    try:
        # Convert input to dict and preprocess
        input_dict = input_data.dict()
        input_df = preprocess_input(input_dict, template_columns, scaler)

        # Predict
        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob > 0.5 else 0
        result = "Fraud" if prediction == 1 else "Not Fraud"

        return {
            "prediction": prediction,
            "fraud_probability": float(round(prob, 4)),
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}

# Optional local run
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)