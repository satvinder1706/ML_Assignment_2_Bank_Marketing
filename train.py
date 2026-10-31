"""
Train five classification models on the UCI Bank Marketing dataset.
Run this file in BITS Virtual Lab before deploying the Streamlit app.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42

# Dataset loading:
# Preferred: place the downloaded UCI file "bank-full.csv" beside this script.
# The UCI CSV is semicolon-separated, so use sep=";".
local_csv = ROOT / "bank-full.csv"
if local_csv.exists():
    print(f"Loading local dataset: {local_csv}")
    full_data = pd.read_csv(local_csv, sep=";")
    if "y" not in full_data.columns:
        raise ValueError("bank-full.csv must contain the target column named 'y'.")
    y = full_data.pop("y").copy()
    X = full_data.copy()
else:
    print("bank-full.csv not found. Trying download from UCI...")
    try:
        dataset = fetch_ucirepo(id=222)
        X = dataset.data.features.copy()
        y = dataset.data.targets.iloc[:, 0].copy()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the dataset from UCI. Download bank-full.csv "
            "from the UCI Bank Marketing dataset page, place it in the same "
            "folder as train.py, and run this script again. "
            f"Original download error: {exc}"
        )

# Avoid target leakage: call duration is known only after the marketing call.
if "duration" in X.columns:
    X = X.drop(columns=["duration"])

y = y.astype(str).str.strip().str.lower()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
with open(MODEL_DIR / "label_classes.json", "w") as f:
    json.dump(label_encoder.classes_.tolist(), f)

numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features)
], remainder="drop")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=RANDOM_STATE, stratify=y_encoded
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                             class_weight="balanced", random_state=RANDOM_STATE),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    # LinearSVC is substantially faster than SVC(probability=True) on this
    # one-hot encoded dataset. decision_function() is used to calculate AUC.
    "Support Vector Machine": LinearSVC(
        C=1.0, class_weight="balanced", max_iter=10000,
        dual="auto", random_state=RANDOM_STATE
    )
}

rows = []
for display_name, estimator in models.items():
    print(f"Training {display_name}...")
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        scores = pipe.predict_proba(X_test)[:, 1]
    else:
        scores = pipe.decision_function(X_test)

    metrics = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    }
    rows.append(metrics)
    safe_name = display_name.lower().replace(" ", "_").replace("-", "")
    joblib.dump(pipe, MODEL_DIR / f"{safe_name}.joblib")
    print(metrics)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(ROOT / "metrics.csv", index=False)

# Keep a labelled hold-out set for the Streamlit evaluation demonstration.
test_df = X_test.copy()
test_df["y"] = label_encoder.inverse_transform(y_test)
test_df.to_csv(ROOT / "test_data.csv", index=False)

metadata = {
    "dataset": "UCI Bank Marketing (dataset id 222)",
    "instances": int(len(X)),
    "features_used": X.columns.tolist(),
    "target": "y (term deposit subscription: yes/no)",
    "positive_class": str(label_encoder.classes_[1]),
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
    "excluded_feature": "duration (post-call target leakage)"
}
with open(ROOT / "training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete. Created model/*.joblib, metrics.csv and test_data.csv.")
print(metrics_df.to_string(index=False))
