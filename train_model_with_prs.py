import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Prioritize SNPs 1 to 75 as per the documentation
snp_cols_for_prs = [f'SNP{i}' for i in range(1, 76)]

train_df['PRS'] = train_df[snp_cols_for_prs].sum(axis=1)
test_df['PRS'] = test_df[snp_cols_for_prs].sum(axis=1)


clinical_features = [
    'Age_Baseline', 'Age_Diag', 'BMI', 'BSA', 'Genre', 'Epaiss_max', 
    'Gradient', 'TVNS', 'FEVG', 'ATCD_MS', 'SYNCOPE', 'Diam_OG'
]
features_to_use = clinical_features + ['Variant.Pathogene', 'PRS']

X_train = train_df[features_to_use]
X_test = test_df[features_to_use]

print("Training model for OUTCOME SEVERITY with PRS...")
y_severity = train_df['OUTCOME SEVERITY']

severity_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
severity_model.fit(X_train, y_severity)

severity_probabilities = severity_model.predict_proba(X_test)[:, 1]
print("OUTCOME SEVERITY model training complete.")

print("\nTraining model for OUTCOME MACE with PRS...")
y_mace = train_df['OUTCOME MACE']

mace_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
mace_model.fit(X_train, y_mace)

mace_predictions = mace_model.predict(X_test)
print("OUTCOME MACE model training complete.")

submission_df = pd.DataFrame({'trustii_id': test_df['trustii_id']})
submission_df['OUTCOME_SEVERITY'] = severity_probabilities
submission_df['OUTCOME_MACE'] = mace_predictions

submission_df['OUTCOME_MACE'] = submission_df['OUTCOME_MACE'].astype(int)

submission_file = 'submission_with_prs.csv'
submission_df.to_csv(submission_file, index=False)

print(f"\nSubmission file '{submission_file}' created successfully.")
print("Top 5 rows of the submission file:")
print(submission_df.head())
