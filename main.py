"""
IBM SkillsBuild Data Analytics & AI Internship
Project: AI-Powered Customer Churn Prediction & Business Intelligence

This script:
1. Loads customer_churn_dataset.csv
2. Performs basic EDA
3. Trains Logistic Regression and Random Forest classifiers
4. Compares metrics
5. Creates churn-risk scores for customers
6. Exports model-ready results

Note: The included dataset is synthetic, created for demonstration and portfolio use.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("customer_churn_dataset.csv")

target = "Churn"
drop_cols = ["Customer_ID", "Churn_Label", "Signup_Date", "Last_Interaction_Date"]
features = [c for c in df.columns if c not in [target] + drop_cols]
X, y = df[features], df[target]

cat_cols = [c for c in features if X[c].dtype == "object"]
num_cols = [c for c in features if c not in cat_cols]

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1200),
    "Random Forest": RandomForestClassifier(
        n_estimators=350, max_depth=10, min_samples_leaf=4,
        random_state=42, class_weight="balanced"
    )
}

for name, model in models.items():
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:,1]
    print(f"\n{name}")
    print("Accuracy :", round(accuracy_score(y_test, pred), 4))
    print("Precision:", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall   :", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1       :", round(f1_score(y_test, pred, zero_division=0), 4))
    print("ROC-AUC  :", round(roc_auc_score(y_test, prob), 4))
