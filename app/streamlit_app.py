import streamlit as st
import requests
import json
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# Default values (keys ke saath session_state me rakhenge)
defaults = {"Time": 406.0, "Amount": 149.62}
defaults.update({f"V{i}": 0.0 for i in range(1, 29)})

for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def load_sample(name):
    with open(SAMPLES_DIR / f"{name}.json") as f:
        data = json.load(f)
    for k, v in data.items():
        st.session_state[k] = float(v)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.markdown(
    """
    **AI-powered transaction risk analysis**

    Enter the transaction details below to determine whether
    a transaction is likely to be legitimate or fraudulent.
    """
)

st.divider()


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📊 Model Performance")

st.caption(
    "Performance metrics are calculated on the held-out test dataset "
    "using the Random Forest model."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        "99.95%"
    )

with col2:
    st.metric(
        "Fraud Precision",
        "91.25%"
    )

with col3:
    st.metric(
        "Fraud Recall",
        "76.84%"
    )


col4, col5, col6 = st.columns(3)

with col4:
    st.metric(
        "F1-Score",
        "83.43%"
    )

with col5:
    st.metric(
        "ROC-AUC",
        "96.94%"
    )

with col6:
    st.metric(
        "PR-AUC",
        "81.20%"
    )


st.divider()


# --------------------------------------------------
# TRANSACTION INFORMATION
# --------------------------------------------------
st.subheader("⚡ Quick Demo")

if (SAMPLES_DIR / "fraud.json").exists():
    demo_col1, demo_col2, _ = st.columns([1, 1, 2])
    with demo_col1:
        st.button(
            "🚨 Load Sample Fraud",
            on_click=load_sample,
            args=("fraud",),
            use_container_width=True
        )
    with demo_col2:
        st.button(
            "✅ Load Sample Legit",
            on_click=load_sample,
            args=("legit",),
            use_container_width=True
        )
else:
    st.caption("Run make_samples.py to enable demo buttons.")

st.divider()

st.subheader("💳 Transaction Information")

col1, col2 = st.columns(2)

with col1:

        time = st.number_input(
        "Transaction Time",
        key="Time",
        help="Time value from the dataset."
    )

with col2:

        amount = st.number_input(
        "Transaction Amount",
        key="Amount",
        min_value=0.0,
        help="Transaction amount."
    )
                    

st.divider()


# --------------------------------------------------
# ANONYMIZED FEATURES
# --------------------------------------------------

st.subheader("🔐 Transaction Features")

st.caption(
    "V1–V28 are anonymized numerical features provided by the dataset."
)

features = {}


with st.expander(
    "Open V1–V28 Features",
    expanded=True
):

    columns = st.columns(4)

    for i in range(1, 29):

        with columns[(i - 1) % 4]:

            features[f"V{i}"] = st.number_input(
                f"V{i}",
                key=f"V{i}",
                format="%.6f"
            )
st.divider()


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.subheader("🔍 Transaction Analysis")

button_col1, button_col2, button_col3 = st.columns(
    [1, 1, 1]
)

with button_col2:

    check_transaction = st.button(
        "🔍 Analyze Transaction",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# API REQUEST
# --------------------------------------------------

if check_transaction:

    transaction = {
        "Time": time,
        "Amount": amount
    }

    transaction.update(features)

    try:

        with st.spinner("Analyzing transaction..."):

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=transaction,
                timeout=10
            )

        if response.status_code == 200:

            result = response.json()

            st.divider()

            # --------------------------------------------------
            # PREDICTION RESULT
            # --------------------------------------------------

            st.subheader("📋 Prediction Result")

            if result["prediction"] == 1:

                st.error(
                    "🚨 FRAUDULENT TRANSACTION"
                )

            else:

                st.success(
                    "✅ LEGITIMATE TRANSACTION"
                )


            # --------------------------------------------------
            # FRAUD PROBABILITY
            # --------------------------------------------------

            probability = result["fraud_probability"]

            st.markdown(
                "### Fraud Probability"
            )

            st.metric(
                "Risk Probability",
                f"{probability * 100:.4f}%"
            )

            st.progress(
                min(probability, 1.0)
            )

            st.write(
                f"**Decision Threshold:** "
                f"{result['threshold'] * 100:.0f}%"
            )


            # --------------------------------------------------
            # EXPLANATION
            # --------------------------------------------------

            if result["prediction"] == 1:

                st.warning(
                    "The predicted fraud probability is at or above "
                    "the configured decision threshold."
                )

            else:

                st.info(
                    "The predicted fraud probability is below "
                    "the configured decision threshold."
                )


        else:

            st.error(
                f"API Error: {response.status_code}"
            )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI server."
        )

        st.info(
            "Make sure FastAPI is running with: "
            "uvicorn app.main:app --reload"
        )


    except requests.exceptions.Timeout:

        st.error(
            "⏱️ The API request timed out."
        )


    except Exception as error:

        st.error(
            f"Unexpected error: {error}"
        )