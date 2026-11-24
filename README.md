# 💳 Credit Card Fraud Detection

An end-to-end machine learning project that predicts whether a credit card
transaction is legitimate or fraudulent, served through a FastAPI backend
and a Streamlit web interface.



## Features

- Fraud prediction using a trained Random Forest model
- FastAPI REST backend (`/predict` endpoint) with Swagger docs
- Interactive Streamlit UI with model performance metrics
- One-click demo buttons (sample fraud / sample legitimate transaction)
- Fraud probability with configurable decision threshold

## Model Performance

Evaluated on a held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 99.95% |
| Fraud Precision | 91.25% |
| Fraud Recall | 76.84% |
| F1-Score | 83.43% |
| ROC-AUC | 96.94% |
| PR-AUC | 81.20% |

The dataset is highly imbalanced (fraud is about 0.17% of transactions), so
accuracy alone is misleading. Precision, recall and PR-AUC are the key metrics.

## Tech Stack

Python, scikit-learn, XGBoost, pandas, FastAPI, Uvicorn, Streamlit

## Project Structure

```
Credit_Card_Fraud_Detection/
├── app/
│   ├── main.py            # FastAPI backend
│   ├── predictor.py       # Model loading and prediction
│   ├── schemas.py         # Request/response schemas
│   └── streamlit_app.py   # Frontend UI
├── models/                # Trained model package
├── samples/               # Demo transactions
├── src/                   # Training code
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/rai432951-dev/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

## Run

Start the backend (Terminal 1):

```bash
uvicorn app.main:app --reload
```

Start the frontend (Terminal 2):

```bash
streamlit run app/streamlit_app.py
```

- API docs: http://127.0.0.1:8000/docs
- Web app: http://localhost:8501

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
contains 284,807 transactions, of which 492 are fraudulent. Features V1–V28 are
PCA-transformed, and `Time` and `Amount` are the original features.

The dataset is not included in this repository. To retrain, download it and
place it at `data/raw/creditcard.csv`.

## Limitations

- Fraud recall is about 77%, so some fraudulent transactions will be missed.
- The model is trained on a single public dataset and is meant for learning
  and demonstration, not production use.

## Author

Shivani