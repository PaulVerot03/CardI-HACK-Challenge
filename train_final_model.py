import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')



snp_cols_prs = [f'SNP{i}' for i in range(1, 76)]
train_df['PRS'] = train_df[snp_cols_prs].sum(axis=1)
test_df['PRS'] = test_df[snp_cols_prs].sum(axis=1)



clinical_features = [
    'Age_Baseline', 'Age_Diag', 'BMI', 'BSA', 'Genre', 'Epaiss_max', 
    'Gradient', 'TVNS', 'FEVG', 'ATCD_MS', 'SYNCOPE', 'Diam_OG'
]
snp_cols = [f'SNP{i}' for i in range(1, 289)]


features_severity = clinical_features + ['Variant.Pathogene'] + snp_cols


features_mace = clinical_features + ['Variant.Pathogene']

X_train_severity = train_df[features_severity]
X_test_severity = test_df[features_severity]

X_train_mace = train_df[features_mace]
X_test_mace = test_df[features_mace]


print("Training final model for OUTCOME SEVERITY...")
y_severity = train_df['OUTCOME SEVERITY']

severity_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
severity_model.fit(X_train_severity, y_severity)

severity_probabilities = severity_model.predict_proba(X_test_severity)[:, 1]
print("OUTCOME SEVERITY model training complete.")


print("\nTraining final model for OUTCOME MACE...")
y_mace = train_df['OUTCOME MACE']

mace_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
mace_model.fit(X_train_mace, y_mace)

mace_predictions = mace_model.predict(X_test_mace)
print("OUTCOME MACE model training complete.")


submission_df = pd.DataFrame({'trustii_id': test_df['trustii_id']})
submission_df['OUTCOME_SEVERITY'] = severity_probabilities
submission_df['OUTCOME_MACE'] = mace_predictions

submission_df['OUTCOME_MACE'] = submission_df['OUTCOME_MACE'].astype(int)

submission_file = 'final_submission.csv'
submission_df.to_csv(submission_file, index=False)

print(f"\nSubmission file '{submission_file}' created successfully.")
print("Top 5 rows of the submission file:")
print(submission_df.head())
