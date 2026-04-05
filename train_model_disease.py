"""
train_model_disease.py
Medibot — Patient Profile → Disease Prediction Model
Dataset : Disease_symptom_and_patient_profile_dataset.csv
Output  : model/disease_model.pkl
          model/disease_features.pkl
          model/disease_classes.pkl
          model/disease_encoders.pkl   ← fixes old bug (encoders were never saved)

Profile Features:
  Fever, Cough, Fatigue, Difficulty Breathing  (binary Yes/No)
  Age (numeric), Gender (Male/Female),
  Blood Pressure (Low/Normal/High),
  Cholesterol Level (Low/Normal/High)

NOTE: "Outcome Variable" column is dropped — it is not a feature.
NOTE: Only 349 rows / 116 classes → class_weight='balanced' + warn user.
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

PROFILE_CSV = os.path.join(DATA_DIR, "Disease_symptom_and_patient_profile_dataset.csv")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("[1/5] Loading patient profile dataset...")
df = pd.read_csv(PROFILE_CSV)
print(f"      Shape: {df.shape}")
print(f"      Columns: {list(df.columns)}")

# ── 2. Clean ──────────────────────────────────────────────────────────────────
print("[2/5] Cleaning...")

df.columns = df.columns.str.strip()

# Drop "Outcome Variable" — leaks diagnosis result, not a real feature
if "Outcome Variable" in df.columns:
    df.drop(columns=["Outcome Variable"], inplace=True)
    print("      Dropped 'Outcome Variable' column")

TARGET = "Disease"
if TARGET not in df.columns:
    match = [c for c in df.columns if c.lower() == "disease"]
    if match:
        df.rename(columns={match[0]: TARGET}, inplace=True)
    else:
        raise ValueError(f"Could not find 'Disease' column. Columns: {list(df.columns)}")

FEATURE_COLS = [c for c in df.columns if c != TARGET]

df.dropna(subset=[TARGET], inplace=True)
print(f"      Rows after drop: {len(df)}")
print(f"      Unique diseases: {df[TARGET].nunique()}")

if df[TARGET].nunique() > 50:
    print("      Warning: 116 classes with only 349 rows — accuracy will be limited.")
    print("        This model handles profile-based routing. Symptom model is primary.")

# ── 3. Encode Features ────────────────────────────────────────────────────────
print("[3/5] Encoding features...")

encoders = {}

BINARY_COLS = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
CATEG_COLS  = ["Gender", "Blood Pressure", "Cholesterol Level"]

for col in BINARY_COLS:
    if col in df.columns:
        df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(0).astype(int)

for col in CATEG_COLS:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str).str.strip()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"      Encoded '{col}': {list(le.classes_)}")

if "Age" in df.columns:
    df["Age"].fillna(df["Age"].median(), inplace=True)

X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
y = df[TARGET].str.strip()

print(f"      Final feature set: {list(X.columns)}")

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("[4/5] Training RandomForestClassifier...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=y if y.value_counts().min() > 1 else None
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("[5/5] Evaluating...")
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"      Test Accuracy: {acc * 100:.2f}%")
print("      (Low accuracy expected — only ~3 rows/class on average)")
print(classification_report(y_test, y_pred, zero_division=0))

# ── 6. Save ───────────────────────────────────────────────────────────────────
print("      Saving model and encoders...")

with open(os.path.join(MODEL_DIR, "disease_model.pkl"), "wb") as f:
    pickle.dump(clf, f)

with open(os.path.join(MODEL_DIR, "disease_features.pkl"), "wb") as f:
    pickle.dump(list(FEATURE_COLS), f)

with open(os.path.join(MODEL_DIR, "disease_classes.pkl"), "wb") as f:
    pickle.dump(list(clf.classes_), f)

with open(os.path.join(MODEL_DIR, "disease_encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

print("\n✅  Disease profile model training complete!")
print(f"    → model/disease_model.pkl")
print(f"    → model/disease_features.pkl")
print(f"    → model/disease_classes.pkl")
print(f"    → model/disease_encoders.pkl")