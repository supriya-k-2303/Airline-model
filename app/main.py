import os
import pickle
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------------------------------
# LOAD SAVED ARTIFACTS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")


def load(filename):
    filepath = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filename} not found in saved_models")

    with open(filepath, "rb") as f:
        return pickle.load(f)


model = load("model.pkl")
scaler = load("scaler.pkl")
target_encoder = load("target_encoder.pkl")
label_encoders = load("label_encoders.pkl")
feature_columns = load("feature_columns.pkl")

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------
app = FastAPI(
    title="Airline Satisfaction API",
    version="1.0.0",
    description="Predict Airline Passenger Satisfaction using Machine Learning"
)

# --------------------------------------------------
# INPUT MODEL
# --------------------------------------------------
class Passenger(BaseModel):
    Gender: str
    Age: int
    Customer_Type: str
    Type_of_Travel: str
    Class: str
    Flight_Distance: float
    Departure_Delay: float
    Arrival_Delay: float
    Departure_Arrival_Time_Convenience: int
    Ease_of_Online_Booking: int
    Check_in_Service: int
    Online_Boarding: int
    Gate_Location: int
    On_board_Service: int
    Seat_Comfort: int
    Leg_Room_Service: int
    Cleanliness: int
    Food_and_Drink: int
    In_flight_Service: int
    In_flight_Wifi_Service: int
    In_flight_Entertainment: int
    Baggage_Handling: int


# --------------------------------------------------
# RESPONSE MODEL
# --------------------------------------------------
class PredictionResponse(BaseModel):
    prediction: str
    risk_segment: str
    confidence_percent: float


# --------------------------------------------------
# PREPROCESS INPUT
# --------------------------------------------------
def preprocess(passenger: Passenger):

    data = passenger.model_dump()

    categorical_columns = [
        "Gender",
        "Customer_Type",
        "Type_of_Travel",
        "Class"
    ]

    for col in categorical_columns:

        encoder = label_encoders[col]

        if data[col] in encoder.classes_:
            data[col] = encoder.transform([data[col]])[0]
        else:
            raise ValueError(
                f"Invalid value '{data[col]}' supplied for column '{col}'"
            )

    df = pd.DataFrame([data])

    df = df.reindex(columns=feature_columns)

    X_scaled = scaler.transform(df)

    return X_scaled


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
def predict(passenger: Passenger):

    try:

        X = preprocess(passenger)

        prediction = model.predict(X)[0]

        probability = float(
            model.predict_proba(X)[0][prediction]
        )

        label = target_encoder.inverse_transform(
            [prediction]
        )[0]

        # Business-friendly segmentation
        if label == "Satisfied":

            if probability >= 0.80:
                segment = "Highly Satisfied"

            elif probability >= 0.60:
                segment = "Satisfied"

            else:
                segment = "Satisfied but At Risk"

        elif label == "Neutral or Dissatisfied":

            if probability >= 0.80:
                segment = "Highly Dissatisfied"

            else:
                segment = "Needs Improvement"

        else:
            segment = "Unknown"

        return PredictionResponse(
            prediction=label,
            risk_segment=segment,
            confidence_percent=round(probability * 100, 2)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/", tags=["Health"])
def home():

    return {
        "status": "API Running",
        "message": "Airline Satisfaction Prediction System is Live"
    }