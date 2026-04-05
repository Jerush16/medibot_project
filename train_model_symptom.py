"""
train_model_symptom.py
Medibot — Symptom → Disease Prediction Model
Datasets: Training.csv, Testing.csv, symptom_precaution.csv
Output:  model/symptom_model.pkl
         model/symptom_features.pkl
         model/symptom_classes.pkl
         model/disease_precautions.pkl
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_CSV      = os.path.join(DATA_DIR, "Training.csv")
TEST_CSV       = os.path.join(DATA_DIR, "Testing.csv")
PRECAUTION_CSV = os.path.join(DATA_DIR, "symptom_precaution.csv")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("[1/5] Loading datasets...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Drop unnamed trailing columns if any
train_df = train_df.loc[:, ~train_df.columns.str.contains("^Unnamed")]
test_df  = test_df.loc[:,  ~test_df.columns.str.contains("^Unnamed")]
print(f"      Train: {train_df.shape}  |  Test: {test_df.shape}")

# ── 2. Clean & Encode ─────────────────────────────────────────────────────────
print("[2/5] Cleaning and encoding...")
TARGET       = "prognosis"
symptom_cols = [c for c in train_df.columns if c != TARGET]

X_train = train_df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_train = train_df[TARGET].str.strip()

X_test  = test_df[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_test  = test_df[TARGET].str.strip()

print(f"      Features : {len(symptom_cols)}")
print(f"      Classes  : {y_train.nunique()} diseases")

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("[3/5] Training RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("[4/5] Evaluating on test set...")
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"      Test Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))

# ── 5. Save Model Artifacts ───────────────────────────────────────────────────
print("[5/5] Saving model and artifacts...")

with open(os.path.join(MODEL_DIR, "symptom_model.pkl"), "wb") as f:
    pickle.dump(clf, f)

with open(os.path.join(MODEL_DIR, "symptom_features.pkl"), "wb") as f:
    pickle.dump(list(symptom_cols), f)

with open(os.path.join(MODEL_DIR, "symptom_classes.pkl"), "wb") as f:
    pickle.dump(list(clf.classes_), f)

# ── 6. Precautions dict ───────────────────────────────────────────────────────
print("      Loading precautions...")
prec_df = pd.read_csv(PRECAUTION_CSV)
prec_df.columns = [c.strip().lower().replace(" ", "_") for c in prec_df.columns]

precaution_cols = [c for c in prec_df.columns if "precaution" in c]
disease_col     = [c for c in prec_df.columns if "disease" in c][0]

precautions = {}
for _, row in prec_df.iterrows():
    disease = str(row[disease_col]).strip()
    precs   = [
        str(row[c]).strip() for c in precaution_cols
        if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
    ]
    precautions[disease] = precs

with open(os.path.join(MODEL_DIR, "disease_precautions.pkl"), "wb") as f:
    pickle.dump(precautions, f)

print(f"      Precautions saved for {len(precautions)} diseases.")
print("\n✅  Symptom model training complete!")
print(f"    → model/symptom_model.pkl")
print(f"    → model/symptom_features.pkl")
print(f"    → model/symptom_classes.pkl")
print(f"    → model/disease_precautions.pkl")