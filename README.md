# AI-Powered Customer Churn Prediction & Business Intelligence

## IBM SkillsBuild Data Analytics & AI Internship — Final Project

**Project type:** Data Analytics + Machine Learning + Business Intelligence  
**Candidate:** Ajay Sharma  
**Primary objective:** Predict customer churn and convert predictions into actionable retention decisions.

---

## 1. Problem Statement

Customer churn reduces recurring revenue and increases customer-acquisition costs. A business needs an early-warning system that identifies customers who are likely to leave, explains the main churn drivers, and helps the retention team prioritize interventions.

This project builds an end-to-end analytics and AI workflow:

**Data → Cleaning → EDA → KPI Analysis → Machine Learning → Risk Scoring → Revenue-at-Risk → Business Recommendations**

---

## 2. Project Objectives

1. Measure overall customer churn.
2. Compare churn across plans, contract types and regions.
3. Identify factors associated with churn.
4. Train predictive classification models.
5. Generate individual customer churn-risk scores.
6. Segment customers into Low, Medium and High risk.
7. Estimate potential revenue exposure.
8. Recommend practical retention actions.

---

## 3. Dataset

The included dataset contains **5,000 synthetic customer records**.

Important: the dataset is intentionally synthetic and contains no real customer PII. It is designed for portfolio/internship demonstration while preserving realistic business relationships.

Main variables include:

- Customer ID
- Age
- Region
- Plan
- Contract Type
- Payment Method
- Tenure
- Monthly Usage
- Monthly Fee
- Total Spend
- Support Tickets
- Late Payments
- Discount
- Satisfaction Score
- Churn

---

## 4. Analytics & AI Methodology

### Data Preparation
- Data-type validation
- Missing-value handling
- Numerical feature scaling
- Categorical one-hot encoding
- Stratified 80/20 train-test split

### Exploratory Analysis
- Overall churn KPI
- Churn by contract type
- Churn by plan
- Regional analysis
- Customer risk distribution
- Revenue-at-risk analysis

### Machine Learning
Two classification models are evaluated:

1. Logistic Regression — interpretable baseline
2. Random Forest — non-linear ensemble model

Model selection is based primarily on **ROC-AUC**, with accuracy, precision, recall and F1 used as supporting metrics.

---

## 5. Key Demonstration Result

The strongest model in this project achieved approximately:

**ROC-AUC: 0.913**

The model is then used to calculate a churn probability for every customer and assign:

- **Low Risk**
- **Medium Risk**
- **High Risk**

The risk score is combined with customer value to support retention prioritization.

---

## 6. Business Recommendations

### 1. Prioritize monthly-contract customers
Monthly customers have greater churn exposure. Offer targeted retention or upgrade pathways before renewal.

### 2. Act on low satisfaction
Customers with low satisfaction should receive service recovery before relying only on discounts.

### 3. Investigate repeated support issues
High support-ticket customers should be reviewed for unresolved problems, onboarding gaps and product friction.

### 4. Combine risk with customer value
A high-risk, high-value customer should receive higher retention priority than a low-value customer with the same probability of churn.

### 5. Measure retention outcomes
Track churn prevented, revenue saved, campaign response and false-positive interventions so the model can be improved over time.

---

## 7. Project Files

| File | Purpose |
|---|---|
| `main.py` | Main Python machine-learning and analytics code |
| `requirements.txt` | Python dependencies |
| `customer_churn_dataset.csv` | Project dataset |
| `AI_Customer_Churn_Analytics_Project_Report.pdf` | Final professional project report |
| `AI_Customer_Churn_Analytics.xlsx` | KPI and analytics workbook |
| `README.md` | Project documentation |

---

## 8. How to Run

```bash
pip install -r requirements.txt
python main.py
```

The code expects `customer_churn_dataset.csv` in the same directory.

---

## 9. Limitations

The dataset is synthetic, so the reported model performance is a demonstration of the analytics workflow rather than production evidence.

For a production deployment, the next steps would include:

- Historical real customer data
- Cross-validation
- Probability calibration
- Model drift monitoring
- Explainability analysis
- Controlled retention experiments
- Measurement of actual revenue saved

---

## 10. Conclusion

This project demonstrates how data analytics can progress from descriptive reporting to predictive decision support. Instead of only identifying customers who already churned, the system estimates which customers may churn next, quantifies potential revenue exposure, and provides a practical framework for retention prioritization.
