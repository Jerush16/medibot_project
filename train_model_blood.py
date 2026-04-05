"""
train_model_blood.py
Medibot — Blood Test (CBC) → Disease Prediction Model
Dataset : diagnosed_cbc_data_v4.csv
Output  : model/blood_model.pkl
          model/blood_features.pkl
          model/blood_classes.pkl

CBC Features (14):
  WBC, LYMp, NEUTp, LYMn, NEUTn, RBC, HGB, HCT,
  MCV, MCH, MCHC, PLT, PDW, PCT

Classes (9):
  Healthy, Normocytic hypochromic anemia,
  Normocytic normochromic anemia, Iron deficiency anemia,
  Thrombocytopenia, Other microcytic anemia,
  Leukemia, Macrocytic anemia, Leukemia with thrombocytopenia
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "data"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

CBC_CSV = os.path.join(DATA_DIR, "diagnosed_cbc_data_v4.csv")

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("[1/5] Loading CBC dataset...")
df = pd.read_csv(CBC_CSV)
print(f"      Shape: {df.shape}")

# ── 2. Inspect & Validate ─────────────────────────────────────────────────────
print("[2/5] Validating data...")
TARGET   = "Diagnosis"
FEATURES = [c for c in df.columns if c != TARGET]

print(f"      Features  : {FEATURES}")
print(f"      Classes   : {df[TARGET].nunique()}")
print("      Class distribution:")
for cls, cnt in df[TARGET].value_counts().items():
    print(f"        {cnt:4d}  {cls}")

# Check for nulls
null_counts = df[FEATURES].isnull().sum()
if null_counts.any():
    print(f"      ⚠ Nulls found — filling with column median")
    for col in FEATURES:
        df[col].fillna(df[col].median(), inplace=True)
else:
    print("      ✓ No null values found")

X = df[FEATURES].astype(float)
y = df[TARGET].str.strip()

# ── 3. Train / Test Split ─────────────────────────────────────────────────────
print("[3/5] Splitting data (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"      Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── 4. Train ──────────────────────────────────────────────────────────────────
print("[4/5] Training RandomForestClassifier...")
# class_weight='balanced' handles minority classes (Leukemia with thrombocytopenia=11 rows)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

# Cross-validation score
cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
print(f"      5-Fold CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
print("[5/5] Evaluating on hold-out test set...")
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
print(f"      Test Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred, zero_division=0))

# ── 6. Save ───────────────────────────────────────────────────────────────────
print("      Saving model artifacts...")

with open(os.path.join(MODEL_DIR, "blood_model.pkl"), "wb") as f:
    pickle.dump(clf, f)

with open(os.path.join(MODEL_DIR, "blood_features.pkl"), "wb") as f:
    pickle.dump(list(FEATURES), f)

with open(os.path.join(MODEL_DIR, "blood_classes.pkl"), "wb") as f:
    pickle.dump(list(clf.classes_), f)

print("\n✅  Blood model training complete!")
print(f"    → model/blood_model.pkl")
print(f"    → model/blood_features.pkl")
print(f"    → model/blood_classes.pkl")