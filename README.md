# Telco Customer Churn Prediction 

An end-to-end classification pipeline developed on the Telco Customer Churn dataset. This project benchmarks multiple machine learning algorithms, handles class imbalance, optimizes hyperparameters via randomized cross-validation, and implements a cost-benefit framework to tune the classification threshold for maximum retention profitability.

---

## 1. Project Workflow

1. **Exploratory Data Analysis & Cleaning:**
   - Identified and resolved blank whitespace entries in `TotalCharges`, casting to numeric floats and imputing missing records.
   - Assessed distribution skewness across tenure, monthly charges, and total charges.
   - Identified class distribution: ~73.5% retained vs. ~26.5% churned.

2. **Feature Engineering & Preprocessing:**
   - Binned continuous tenure into categorical cohorts (`Tenure_Group`).
   - Encoded categorical features via one-hot encoding (`drop_first=True` to prevent multicollinearity).
   - Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.

3. **Model Benchmarking:**
   - Evaluated 4 distinct architectures: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and XGBoost Classifier.
   - Mitigated class imbalance using `class_weight='balanced'` and XGBoost's `scale_pos_weight`.

4. **Hyperparameter Tuning:**
   - Conducted `RandomizedSearchCV` on XGBoost (5-fold Stratified CV optimizing for F1-score) over `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, and `scale_pos_weight`.

5. **Cost-Benefit Analysis & Threshold Optimization:**
   - Standard `.predict()` uses an arbitrary 0.5 probability cutoff.
   - Evaluated decision thresholds from `0.30` to `0.70` against a retention utility function incorporating False Negative penalties.
   - Selected **`0.30`** as the optimal operating threshold to maximize net financial return.

6. **Deployment Artifact:**
   - Wrapped the tuned model and custom decision threshold into a scikit-learn compatible estimator (`CustomThresholdModel`) and exported via `joblib`.

---

## 2. Model Evaluation & Benchmark Results

Because the dataset is imbalanced (~73:27), evaluation focuses on **ROC-AUC**, **F1-Score**, and **Minority Recall** alongside standard Accuracy.

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.79 | 0.60 | 0.55 | 0.57 | 0.835 |
| **Decision Tree** | 0.73 | 0.50 | 0.51 | 0.50 | 0.658 |
| **Random Forest** | 0.79 | 0.63 | 0.52 | 0.57 | 0.824 |
| **XGBoost (Tuned  0.50)** | **0.80** | **0.63** | 0.60 | 0.61 | **0.842** |
| **XGBoost (Tuned  0.30 Threshold)** | 0.75 | 0.53 | **0.78** | **0.63** | **0.842** |

> **Business Impact:** Lowering the threshold to `0.30` boosts churn recall from ~60% to ~78%. While this introduces additional false positives (retention outreach costs), it substantially cuts down high-penalty false negatives (lost customer lifetime value), maximizing net business profit.

---

## 3. Key Feature Drivers

Top features driving churn predictions identified by the tuned XGBoost model:
- **Contract Type:** `Month-to-month` contracts represent the strongest churn indicator, whereas 1-year and 2-year contracts correlate heavily with retention.
- **Internet Service:** `Fiber optic` users display higher attrition rates, indicating sensitivity to pricing or service performance.
- **Tenure:** Higher tenure significantly lowers churn risk.
- **Financial Features:** High recurring `MonthlyCharges` increase customer attrition probability.

---

## 4. Repository Structure
```text
telco-customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   └── churn_prediction_pipeline.ipynb
├── models/
│   └── churn_model_xgb.pkl
├── .gitignore
├── requirements.txt
└── README.md
