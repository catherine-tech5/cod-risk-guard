# AI Usage Log — COD Risk Guard

## 1. AI Tools Used

**Tool:** ChatGPT

**Purpose:** Used as a development assistant during the take-home assignment.

AI assistance was used for:
- Understanding the assignment requirements and planning the project architecture.
- Explaining data cleaning, feature analysis and machine learning concepts.
- Supporting Python implementation and debugging.
- Designing the four-action decision workflow.
- Reviewing business economics and threshold assumptions.
- Developing the Streamlit dashboard and WhatsApp interaction simulator.
- Reviewing model evaluation results and preparing documentation.

All code was reviewed, executed and tested locally. Model outputs and financial calculations were checked against the supplied dataset and assignment assumptions.

## 2. Examples of Weak or Incorrect AI Suggestions

### Example 1 — Misleading accuracy interpretation

**Initial issue:** Accuracy from the baseline classifier appeared relatively high, even though the classifier failed to identify any RTO orders.

**Correction:** Precision, recall, F1 score, ROC-AUC, PR-AUC and the confusion matrix were examined instead of relying on accuracy alone.

**Learning:** Accuracy can be misleading when the dataset contains unequal numbers of positive and negative outcomes.

### Example 2 — Overly aggressive cancellation risk

**Initial issue:** A high-risk order could be interpreted as a candidate for automatic cancellation.

**Correction:** The final prototype disables automatic cancellation and sends very high-risk orders to manual review.

**Learning:** A model prediction should support business decisions, not independently reject genuine customers.

### Example 3 — Financial savings treated too confidently

**Initial issue:** Estimated savings from interventions could be mistaken for guaranteed or realised savings.

**Correction:** The dashboard labels financial results as scenario estimates and exposes intervention effectiveness and prepaid conversion as editable assumptions.

**Learning:** Predicted risk reduction must be validated through actual customer responses and delivery outcomes.

## 3. Human Validation and Testing

The following components were tested locally:

- Data cleaning and exploratory analysis.
- Feature analysis.
- Model training and evaluation.
- September 2026 chronological backtest.
- Four-action decision engine.
- WhatsApp YES, NO, unclear and no-response scenarios.
- Prepaid discount calculation.
- Streamlit dashboard and interactive financial assumptions.
- CSV decision-log export.

## 4. Limitations of AI Assistance

AI-generated code and recommendations were not treated as automatically correct. They were checked through execution, output inspection and comparison with the assignment requirements.

The final prototype remains an experimental decision-support system. Its model performance, predicted probabilities, intervention effectiveness and financial assumptions require further validation before production use.