import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss, make_scorer, cohen_kappa_score
import numpy as np

# Load the training data
train_df = pd.read_csv('train.csv')

# --- Feature Sets ---
clinical_features = [
    'Age_Baseline', 'Age_Diag', 'BMI', 'BSA', 'Genre', 'Epaiss_max', 
    'Gradient', 'TVNS', 'FEVG', 'ATCD_MS', 'SYNCOPE', 'Diam_OG'
]
snp_cols = [f'SNP{i}' for i in range(1, 289)]
snp_cols_prs = [f'SNP{i}' for i in range(1, 76)]

# Create PRS feature
train_df['PRS'] = train_df[snp_cols_prs].sum(axis=1)

feature_sets = {
    "Clinical Only": clinical_features,
    "Clinical + Variant": clinical_features + ['Variant.Pathogene'],
    "Clinical + Variant + PRS": clinical_features + ['Variant.Pathogene', 'PRS'],
    "Clinical + Variant + All SNPs": clinical_features + ['Variant.Pathogene'] + snp_cols,
}

# --- Target Variables ---
y_severity = train_df['OUTCOME SEVERITY']
y_mace = train_df['OUTCOME MACE']

# --- Cross-Validation Evaluation ---

# Custom scorer for log loss
def log_loss_scorer_custom(y_true, y_pred_proba):
    return log_loss(y_true, y_pred_proba)

# Scorer for QWK
qwk_scorer = make_scorer(cohen_kappa_score, weights='quadratic')

for name, features in feature_sets.items():
    print(f"--- Evaluating: {name} ---")
    X = train_df[features]

    # --- OUTCOME SEVERITY ---
    severity_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Cross-validated log loss
    # The 'log_loss' string scorer in scikit-learn expects probabilities of the positive class
    cv_log_loss = -cross_val_score(severity_model, X, y_severity, cv=5, scoring='neg_log_loss').mean()
    print(f"  OUTCOME SEVERITY (Log Loss): {cv_log_loss:.4f}")

    # --- OUTCOME MACE ---
    mace_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    # Cross-validated QWK
    cv_qwk = cross_val_score(mace_model, X, y_mace, cv=5, scoring=qwk_scorer).mean()
    print(f"  OUTCOME MACE (QWK): {cv_qwk:.4f}")
    
    # Cross-validated accuracy
    cv_accuracy = cross_val_score(mace_model, X, y_mace, cv=5, scoring='accuracy').mean()
    print(f"  OUTCOME MACE (Accuracy): {cv_accuracy:.4f}")

    # Cross-validated ROC AUC
    # Note: For multi-class, this calculates ROC AUC for each class vs. rest
    cv_roc_auc = cross_val_score(mace_model, X, y_mace, cv=5, scoring='roc_auc_ovr').mean()
    print(f"  OUTCOME MACE (ROC AUC OVR): {cv_roc_auc:.4f}")
    print("")

