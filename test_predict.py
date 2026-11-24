import pandas as pd
from app.predictor import predict_fraud, FEATURES, THRESHOLD

print("Threshold:", THRESHOLD)
print("Features:", FEATURES)

df = pd.read_csv("data/raw/creditcard.csv")
fraud_rows = df[df["Class"] == 1].head(5)

for _, row in fraud_rows.iterrows():
    print(predict_fraud(row[FEATURES].to_dict()))