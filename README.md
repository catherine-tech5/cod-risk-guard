# COD Risk Guard
### AI-Powered Cash-on-Delivery Risk Assessment System

## 1. Project Overview

COD Risk Guard is an AI-powered pre-shipment decision-support system designed to reduce Return-to-Origin (RTO) losses in e-commerce businesses.

The system analyses historical order data, estimates the RTO risk of incoming COD orders, and recommends one of four actions:

1. **Ship as-is** – Dispatch low-risk orders directly.
2. **WhatsApp confirmation** – Confirm customer availability before dispatch.
3. **Prepaid nudge** – Offer a small discount to encourage prepaid payment.
4. **Hold for manual review** – Send very high-risk orders for human verification.

The objective is to balance operational costs with customer experience and avoid unnecessarily cancelling genuine orders.

## 2. Key Features

- Historical order data analysis and feature engineering.
- Machine learning-based RTO risk estimation using Logistic Regression.
- Four-action configurable decision engine.
- Editable business thresholds through JSON configuration.
- WhatsApp confirmation and prepaid discount simulator.
- Interactive Streamlit dashboard.
- September 2026 out-of-time model evaluation.
- Order-level decision log and CSV export.
- Scenario-based financial impact estimation.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data cleaning and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Logistic Regression | RTO risk prediction |
| Joblib | Model persistence |
| Streamlit | Interactive dashboard |
| Plotly | Data visualization |

## 4. Project Structure

```text
cod-risk-guard/
│
├── app.py
├── analyze_data.py
├── feature_analysis.py
├── train_model.py
├── evaluate_model.py
├── decision_engine.py
├── whatsapp_simulator.py
├── september_backtest.py
├── business_economics.py
│
├── business_config.json
├── requirements.txt
│
├── orders_sample.csv
├── cleaned_orders.csv
├── rto_model.joblib
│
├── september_backtest.csv
├── september_decision_summary.csv
│
├── README.md
├── AI_USAGE_LOG.md
├── .gitignore
│
└── venv/
```

The virtual environment (`venv/`) should not be committed to GitHub.

## 5. Installation and Setup

### Prerequisites

- Python 3.12 or compatible version.
- pip package manager.
- Visual Studio Code or another Python editor.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd cod-risk-guard
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

The repository should include the dataset, cleaned data and trained model required by the dashboard.

## 6. Machine Learning Approach

A Logistic Regression classification model is used to estimate the likelihood of an order becoming RTO.

### Input Features

- Order value
- Number of items
- Customer's previous order count
- Customer's previous RTO count
- Order hour
- Order month
- State
- Courier
- Marketing source

Numerical features are imputed and scaled. Categorical features are imputed and one-hot encoded.

The dataset is split chronologically to reduce information leakage from future orders into model training.

### September 2026 Backtesting

The model was trained using April–August 2026 orders and evaluated on September 2026 orders.

| Metric | September result |
|---|---:|
| Orders evaluated | 329 |
| Actual RTO rate | 26.14% |
| ROC-AUC | 0.6822 |
| PR-AUC | 0.4051 |
| Precision at 0.18 | 31.56% |
| Recall at 0.18 | 89.53% |
| F1 score at 0.18 | 46.67% |

The 0.18 classification threshold was selected using earlier validation data. September was kept as the out-of-time test period.

## 7. Business Decision Configuration

Business rules can be edited in `business_config.json`.

Current risk thresholds:

| Estimated risk | Recommended action |
|---|---|
| Below 15% | Ship as-is |
| 15% to below 30% | WhatsApp confirmation |
| 30% to below 55% | Prepaid nudge |
| 55% and above | Hold for manual review |

The prepaid discount is currently ₹20, and the confirmation response window is 24 hours.

Automatic cancellation is disabled to protect genuine customers.

These thresholds are initial business assumptions and require further validation through a controlled pilot.

## 8. Business Economics

The expected RTO cost is calculated using:

- Forward shipping: ₹65
- Return shipping: ₹65
- Packaging and handling: ₹25
- Expected damage cost: 8% × ₹180

Expected RTO cost = ₹169.40.

Using the assignment's COD RTO rate of 28% and prepaid RTO rate of 4%, the estimated RTO cost reduction per successful COD-to-prepaid conversion is ₹40.66 before discount.

With a ₹20 prepaid discount, the illustrative net benefit is ₹20.66 per successful conversion.

## 9. Dashboard

The Streamlit dashboard provides:

- Historical order overview.
- Four recommended decision categories.
- Decision distribution chart.
- Scenario-based financial impact.
- Order-level decision log.
- CSV download.
- Interactive WhatsApp confirmation and prepaid offer simulator.

## 10. Limitations and Future Improvements

- The dataset is limited to approximately 2,000 historical orders.
- Model ranking performance is moderate and requires improvement.
- Predicted risk scores have not been calibrated.
- Address-quality checks and LLM-based Hinglish address interpretation are not yet integrated into the decision engine.
- WhatsApp interaction is simulated; no live WhatsApp API is connected.
- Payment-link generation and payment collection are not integrated.
- Financial savings are scenario estimates, not verified realised savings.
- Business thresholds require controlled experimentation and monitoring.

Future improvements include probability calibration, better address validation, a real WhatsApp integration, controlled intervention experiments, and continuous model monitoring.

## 11. Disclaimer

COD Risk Guard is a prototype decision-support system developed for an Associate Software Engineer take-home assignment. Model predictions are estimates and should not independently determine customer cancellation decisions.