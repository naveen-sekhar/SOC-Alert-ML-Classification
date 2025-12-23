# catboost_full_train_fixed.py
# Requirements:
# pip install catboost scikit-learn joblib pandas scipy

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier, Pool

# ---------- Config ----------
DATA_PATH = "cleaned_smart_siem_dataset.csv"   # produced by preprocessing
OUT_DIR = "catboost_models"
os.makedirs(OUT_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 2000
SVD_COMPONENTS = 50
RANDOM_STATE = 42
TEST_SIZE = 0.20

CAT_COLUMNS = ["Alert Name", "Severity", "Protocol"]  # keep as strings
TEXT_COLUMN = "Executive Summary"
TARGET_COLUMNS = ["Status", "Category", "Action Taken"]

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH)
print("Loaded rows:", df.shape[0])

# ---------- Prepare text: TF-IDF + SVD ----------
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1,2), stop_words="english")
X_text_sparse = tfidf.fit_transform(df[TEXT_COLUMN].astype(str))
print("TF-IDF shape:", X_text_sparse.shape)

svd = TruncatedSVD(n_components=min(SVD_COMPONENTS, max(1, X_text_sparse.shape[1]-1)), random_state=RANDOM_STATE)
X_text_reduced = svd.fit_transform(X_text_sparse)
print("SVD reduced shape:", X_text_reduced.shape)

# Save text preprocessors
joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
joblib.dump(svd, os.path.join(OUT_DIR, "svd.joblib"))

# ---------- Build feature DataFrame for CatBoost ----------
# Keep categorical columns as strings (CatBoost will handle them)
X_cat = df[CAT_COLUMNS].astype(str).reset_index(drop=True)

# Numeric columns: everything except CAT_COLUMNS, TEXT_COLUMN, TARGET_COLUMNS
exclude = set(CAT_COLUMNS + [TEXT_COLUMN] + TARGET_COLUMNS)
numeric_cols = [c for c in df.columns if c not in exclude]
X_num = df[numeric_cols].reset_index(drop=True).astype(float)

# Text reduced -> DataFrame
txt_df = pd.DataFrame(X_text_reduced, columns=[f"txt_{i}" for i in range(X_text_reduced.shape[1])])

# Concatenate all parts
X_df = pd.concat([X_cat, X_num, txt_df], axis=1)
print("Final feature shape:", X_df.shape)

# Indices / names of categorical features for CatBoost
cat_feature_names = CAT_COLUMNS  # pass names to Pool

# ---------- Prepare targets (label encode) ----------
label_encoders = {}
y = pd.DataFrame()
for t in TARGET_COLUMNS:
    le = LabelEncoder()
    y[t] = le.fit_transform(df[t].astype(str))
    label_encoders[t] = le
# save label encoders
joblib.dump(label_encoders, os.path.join(OUT_DIR, "label_encoders.joblib"))

# ---------- Train/test split (stratify by Status to maintain distribution) ----------
X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(
    X_df, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y["Status"]
)

print("Train shape:", X_train_df.shape, "Test shape:", X_test_df.shape)

# ---------- Optional: Save train/test CSVs for evaluation scripts ----------
SAVE_SPLITS = os.getenv("SAVE_TRAIN_TEST_SPLITS", "0") not in {"0", "false", "False", ""}
if SAVE_SPLITS:
    X_train_df.to_csv("train_features.csv", index=False)
    X_test_df.to_csv("test_features.csv", index=False)

    # Save labels per target to avoid ambiguity (each target has different label encoding)
    for target in TARGET_COLUMNS:
        y_train_df[[target]].to_csv(f"train_labels_{target}.csv", index=False)
        y_test_df[[target]].to_csv(f"test_labels_{target}.csv", index=False)

    print("Saved train/test feature CSVs and per-target label CSVs for evaluation.")
else:
    print("Skipping train/test CSV export (set SAVE_TRAIN_TEST_SPLITS=1 to enable).")

# ---------- Train one CatBoost per target (auto-select loss/metric) ----------
models = {}
for target in TARGET_COLUMNS:
    print("\n========== Training target:", target, "==========")
    y_train = y_train_df[target].values
    y_test = y_test_df[target].values

    # decide if binary or multiclass
    n_classes = len(np.unique(y_train))
    if n_classes <= 2:
        loss = "Logloss"
        eval_metric = "AUC"  # AUC is fine for binary; change to "Accuracy" if preferred
    else:
        loss = "MultiClass"
        eval_metric = "MultiClass"

    # Build Pool objects; pass cat_features as names
    train_pool = Pool(data=X_train_df, label=y_train, cat_features=cat_feature_names)
    val_pool = Pool(data=X_test_df, label=y_test, cat_features=cat_feature_names)

    clf = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        loss_function=loss,
        eval_metric=eval_metric,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=100,
        verbose=100
    )

    clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Predictions and evaluation
    y_pred = clf.predict(X_test_df)
    # flatten if needed
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()

    # Print classification report using original label names
    target_names = label_encoders[target].classes_
    print(f"\nClassification report for {target}:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=target_names))

    # Save model
    model_path = os.path.join(OUT_DIR, f"catboost_{target}.cbm")
    clf.save_model(model_path)
    models[target] = clf

# Save models dict for convenience
joblib.dump(models, os.path.join(OUT_DIR, "catboost_models_dict.joblib"))

print("\nAll models trained & saved in folder:", OUT_DIR)
