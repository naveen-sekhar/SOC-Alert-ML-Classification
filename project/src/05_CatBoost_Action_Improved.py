# retrain_action_with_weights.py
import os, joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import classification_report, confusion_matrix

IN = "fe_smart_siem_dataset.csv"
OUT_DIR = "catboost_action_improved"
os.makedirs(OUT_DIR, exist_ok=True)

# Load engineered dataset
df = pd.read_csv(IN)

# Columns
TEXT_COLUMN = "Executive Summary"
CAT_COLUMNS = ["Alert Name", "Severity", "Protocol"]
TARGET = "Action Taken"

# Prepare TF-IDF + SVD for text (same settings as before)
tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
X_text = tfidf.fit_transform(df[TEXT_COLUMN].astype(str))
svd = TruncatedSVD(n_components=min(50, max(1, X_text.shape[1]-1)), random_state=42)
X_text_red = svd.fit_transform(X_text)

joblib.dump(tfidf, os.path.join(OUT_DIR, "tfidf.joblib"))
joblib.dump(svd, os.path.join(OUT_DIR, "svd.joblib"))

# Build feature frame: categories + numeric + text reduced
exclude = set(CAT_COLUMNS + [TEXT_COLUMN, "Status", "Category", "Action Taken"])
numeric_cols = [c for c in df.columns if c not in exclude]
X_cat = df[CAT_COLUMNS].astype(str).reset_index(drop=True)
X_num = df[numeric_cols].reset_index(drop=True).astype(float)
txt_df = pd.DataFrame(X_text_red, columns=[f"txt_{i}" for i in range(X_text_red.shape[1])])

X = pd.concat([X_cat, X_num, txt_df], axis=1)

# Target encode
le = LabelEncoder()
y = le.fit_transform(df[TARGET].astype(str))
joblib.dump(le, os.path.join(OUT_DIR, "action_label_encoder.joblib"))

# Train/test split (stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Compute class weights
classes = np.unique(y_train)
cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# CatBoost expects list mapped to class indices order
class_weights_list = [float(w) for w in cw]
print("Class weights:", class_weights_list)

# Cat feature names
cat_feature_names = CAT_COLUMNS

# Pools
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature_names)
val_pool = Pool(data=X_test, label=y_test, cat_features=cat_feature_names)

# Train
clf = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function="MultiClass",
    eval_metric="MultiClass",
    class_weights=class_weights_list,
    random_seed=42,
    early_stopping_rounds=80,
    verbose=100
)

clf.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Predict & eval
y_pred = clf.predict(X_test)
if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] == 1:
    y_pred = y_pred.ravel()

print("\nClassification report (Action Taken):")
print(classification_report(y_test, y_pred, zero_division=0, target_names=le.classes_))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
clf.save_model(os.path.join(OUT_DIR, "catboost_action_improved.cbm"))
joblib.dump(clf, os.path.join(OUT_DIR, "catboost_action_improved.joblib"))

print("Saved improved Action model & artifacts to:", OUT_DIR)
