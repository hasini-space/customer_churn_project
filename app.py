from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize the FastAPI Application instance
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production endpoint for evaluating subscription cancellation liabilities."
)

# Load the serialized pipeline once when the application spins up
MODEL_PATH = "models/churn_pipeline.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Artifact error: Could not find model file at {MODEL_PATH}. Run main.py first.")

# Define the structured data schema expected from API requests
class CustomerData(BaseModel):
    Age: int
    Tenure: int
    MonthlyCharges: float
    Contract: str
    PaymentMethod: str
    PaperlessBilling: str

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": True}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert incoming JSON payload to a pandas DataFrame shape matching the original input
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Calculate raw binary prediction and probabilities
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 4),
            "risk_status": "High Risk" if probability > 0.5 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference execution failed: {str(e)}")