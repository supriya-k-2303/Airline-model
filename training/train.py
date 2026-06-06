import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ---------------- LOAD DATA ----------------
df = pd.read_csv("C:\\Users\\Supriya\\Desktop\\placement_project\\data\\DS-DATA.csv", low_memory=False)

# normalize column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

# rename for consistency
df = df.rename(columns={
    "Customer Type": "Customer_Type",
    "Type of Travel": "Type_of_Travel",
    "Departure and Arrival Time Convenience": "Departure_Arrival_Time_Convenience",
    "Food and Drink": "Food_and_Drink",
    "In-flight Service": "In_flight_Service",
    "In-flight Wifi Service": "In_flight_Wifi_Service",
    "In-flight Entertainment": "In_flight_Entertainment",
    "Baggage Handling": "Baggage_Handling",
    "Flight Distance": "Flight_Distance",
    "Departure Delay": "Departure_Delay",
    "Arrival Delay": "Arrival_Delay",
})

# ---------------- TARGET ----------------
y = df["Satisfaction"]
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# ---------------- FEATURES ----------------
X = df.drop(columns=["Satisfaction", "ID"], errors="ignore")

categorical_cols = ["Gender", "Customer_Type", "Type_of_Travel", "Class"]

label_encoders = {}

# encode categorical
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# fill missing numeric safely
for col in X.columns:
    if X[col].dtype != "int32" and X[col].dtype != "int64":
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- MODEL ----------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_scaled, y)

# ---------------- SAVE ----------------
with open("saved_models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("saved_models/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

with open("saved_models/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("saved_models/feature_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("TRAINING COMPLETED SUCCESSFULLY")