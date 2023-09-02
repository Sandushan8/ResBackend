from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils import intiate_prediction
app = FastAPI()

# Define a Pydantic model for the input data
class InputData(BaseModel):
    array: List[int]
    # Add more features as needed

# Define a response model
class PredictionResponse(BaseModel):
    prediction: str
    
@app.get('/')
def read_root():
    return {"message": "Hello, Welcome!"}
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: InputData):
    try:
        # Extract input data from the Pydantic model
        prediction = intiate_prediction(data.array)
        # Return the prediction as a JSON response
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
