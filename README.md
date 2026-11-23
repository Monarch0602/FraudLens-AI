# FraudLens AI

### Explainable AI-Powered Fraud Detection & Transaction Investigation

FraudLens AI is a machine-learning-based fraud detection system designed to identify suspicious financial transactions, quantify transaction risk, and provide interpretable insights to support fraud investigation.

Rather than relying solely on classification accuracy, FraudLens focuses on the challenges of imbalanced fraud detection, probability-based risk scoring, decision threshold optimization, and model explainability.

---

## Key Features

- **Machine Learning Fraud Detection** — Random Forest classifier for transaction fraud prediction.
- **Risk Scoring** — Probability-based assessment of transaction fraud risk.
- **Threshold Optimization** — Classification threshold selected using validation data.
- **Explainable AI** — SHAP-based feature attribution to understand model predictions.
- **Risk Signal Detection** — Rule-based identification of suspicious transaction characteristics.
- **Investigation Recommendations** — Action-oriented recommendations based on transaction risk and detected signals.
- **Interactive Dashboard** — Streamlit interface for transaction analysis and model insights.
- **Model Evaluation** — Accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix analysis.

---

## Machine Learning Performance

The final Random Forest model was evaluated on a held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 98.47% |
| Balanced Accuracy | 85.86% |
| Precision | 72.50% |
| Recall | 72.50% |
| F1-Score | 72.50% |
| ROC-AUC | 98.68% |
| PR-AUC | 64.89% |

The classification threshold was selected using validation data to balance precision and recall.

The evaluation emphasizes fraud-specific metrics because a high accuracy score alone can be misleading when fraudulent transactions represent a small proportion of the dataset.

---

## System Architecture

    Transaction Data
          |
          v
    Data Preprocessing
          |
          v
    Feature Engineering
          |
          v
    Random Forest Classifier
          |
          v
    Fraud Probability Estimation
          |
          v
    Optimized Decision Threshold
          |
          v
    Transaction Risk Classification
          |
    +---------------------+
    |                     |
    v                     v
    SHAP Explanations   Risk Signals
    |                     |
    +----------+----------+
               |
               v
    Investigation Insights
               |
               v
    Streamlit Dashboard

---

## Explainable AI

FraudLens integrates SHAP-based feature attribution to help understand the factors influencing model predictions.

The explainability layer provides insight into:

- Features contributing to elevated fraud probability
- Factors influencing individual transaction predictions
- Relative contribution of input features to model decisions

This helps make model outputs more interpretable instead of treating predictions as black-box decisions.

---

## Investigation Intelligence

The project includes additional investigation-oriented components:

- Risk signal generation
- Suspicious transaction indicators
- Risk categorization
- Investigation recommendations
- Structured investigation report generation

These components are intended to support human review and should not be treated as definitive evidence of fraud.

---

## Dataset

The project uses a transaction dataset containing 7,200 records.

| Dataset characteristic | Value |
|---|---:|
| Total transactions | 7,200 |
| Legitimate transactions | 7,000 |
| Fraudulent transactions | 200 |
| Fraud prevalence | 2.78% |

The dataset contains transaction-related attributes used for fraud classification.

The dataset is not included in the GitHub repository.

---

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Explainability | SHAP |
| Visualization | Matplotlib, Plotly |
| Application | Streamlit |
| Model Persistence | Joblib |

---

## Project Structure

    FraudLens-AI/
    ├── src/
    │   ├── data/
    │   ├── evaluation/
    │   ├── explainability/
    │   ├── features/
    │   ├── investigation/
    │   ├── models/
    │   ├── risk/
    │   └── utils/
    ├── models/
    │   └── saved/
    ├── reports/
    ├── scripts/
    ├── data/
    ├── app.py
    ├── investigation_app.py
    ├── requirements.txt
    └── README.md

---

## Installation

Clone the repository:

    git clone https://github.com/YOUR-USERNAME/FraudLens-AI.git

Navigate into the project directory:

    cd FraudLens-AI

Create a virtual environment:

    python -m venv venv

Activate it on Windows:

    venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

---

## Running the Application

Launch the investigation dashboard:

    streamlit run investigation_app.py

Alternatively, launch the primary application:

    streamlit run app.py

---

## Model Evaluation Strategy

FraudLens uses a structured evaluation workflow:

1. Data preprocessing and feature engineering
2. Stratified train-validation-test split
3. Model training and comparison
4. Validation-based decision threshold selection
5. Final evaluation on held-out test data
6. Explainability and fraud risk analysis

The test set is kept separate from threshold selection to reduce evaluation leakage.

---

## Limitations

- Model performance depends on the dataset and its underlying distribution.
- Historical transaction patterns may not capture evolving fraud strategies.
- Risk signals may generate false positives.
- Model predictions should be used as decision-support information, not as standalone proof of fraud.
- Performance on unseen real-world transaction data may differ from reported test metrics.

---

## Future Improvements

- Real-time transaction monitoring
- Advanced fraud pattern detection
- Temporal anomaly detection
- Model drift monitoring
- API-based deployment
- Expanded investigation intelligence
- Integration with real-world financial transaction systems

---

## Disclaimer

FraudLens AI is an educational and analytical machine learning project. Its predictions are not financial, legal, or operational determinations and should not be used as the sole basis for decisions affecting individuals.

---

**Developed as a machine learning and explainable AI project focused on financial fraud detection, risk analytics, and investigation support.**