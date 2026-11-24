from fastapi import FastAPI
from app.schemas import TransactionRequest
from app.predictor import predict_fraud

app = FastAPI()


@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is running"
    }
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "Random Forest",
        "api": "Credit Card Fraud Detection API"
    }


@app.post("/predict")
def predict(transaction: TransactionRequest):

    transaction_data = transaction.model_dump()

    result = predict_fraud(transaction_data)

    return result