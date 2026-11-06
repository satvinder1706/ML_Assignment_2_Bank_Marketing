"""
Train five classifiers on the UCI Bank Marketing dataset.
Run: python train.py

Creates model/*.joblib, model/label_classes.json, test_data.csv,
metrics.csv and training_metadata.json in the same folder as this script.
"""
from pathlib import Path
import csv
import json
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Load local UCI file if present; otherwise fetch dataset 222.
local_csv = ROOT / "bank-full.csv"
if local_csv.exists():
    print(f"Loading local dataset: {local_csv}")
    full_data = pd.read_csv(local_csv, sep=";")
    if "y" not in full_data.columns:
        raise ValueError("bank-full.csv must contain a target column named 'y'.")
    y = full_data.pop("y").copy()
    X = full_data.copy()
else:
    print("bank-full.csv not found; downloading UCI dataset 222...")
    dataset = fetch_ucirepo(id=222)
    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

# Exclude duration: it is only known after the marketing call.
if "duration" in X.columns:
    X = X.drop(columns=["duration"])

y = y.astype(str).str.strip().str.lower()
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
if len(label_encoder.classes_) != 2:
    raise ValueError(f"Expected binary target; found {label_encoder.classes_.tolist()}")

with open(MODEL_DIR / "label_classes.json", "w", encoding="utf-8") as f:
    json.dump(label_encoder.classes_.tolist(), f, indent=2)

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
    X, y_encoded, test_size=0.20, random_state=RANDOM_STATE,
    stratify=y_encoded
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=10, class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": LinearSVC(
        C=1.0, class_weight="balanced", max_iter=10000,
        dual="auto", random_state=RANDOM_STATE
    )
}

metric_rows = []
model_filenames = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knearest_neighbors.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

for display_name, estimator in models.items():
    print(f"Training {display_name}...")
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    scores = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else pipeline.decision_function(X_test)
    )
    metric_rows.append({
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    })
    joblib.dump(pipeline, MODEL_DIR / model_filenames[display_name])

# Save labelled hold-out data as a standard comma-separated CSV.
test_df = X_test.copy()
test_df["y"] = label_encoder.inverse_transform(y_test)
test_path = ROOT / "test_data.csv"
test_df.to_csv(
    test_path, index=False, sep=",", encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)

# Read it back and fail immediately if the target column is not separate.
check_df = pd.read_csv(test_path)
check_df.columns = check_df.columns.astype(str).str.strip()
if "y" not in check_df.columns or check_df.shape[1] != test_df.shape[1]:
    raise RuntimeError(
        "Saved test_data.csv failed validation. "
        f"Columns read back: {check_df.columns.tolist()}"
    )

metrics_df = pd.DataFrame(metric_rows)
metrics_df.to_csv(
    ROOT / "metrics.csv", index=False, encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)

metadata = {
    "dataset": "UCI Bank Marketing (dataset id 222)",
    "instances": int(len(X)),
    "features_used": X.columns.tolist(),
    "target": "y (term deposit subscription: yes/no)",
    "classes": label_encoder.classes_.tolist(),
    "test_rows": int(len(test_df)),
    "test_columns": test_df.columns.tolist(),
    "test_size": 0.20,
    "random_state": RANDOM_STATE,
    "excluded_feature": "duration (post-call target leakage)"
}
with open(ROOT / "training_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nTraining complete.")
print(f"Validated test_data.csv: {check_df.shape[0]} rows x {check_df.shape[1]} columns")
print(f"Target column y present: {'y' in check_df.columns}")
print(f"Models saved in: {MODEL_DIR}")
print(metrics_df.to_string(index=False))
