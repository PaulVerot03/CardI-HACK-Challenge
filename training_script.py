import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, cohen_kappa_score
import joblib
import os
import logging
from datetime import datetime
import argparse

# --- Constants ---
CLINICAL_COLS = ['Age_Baseline', 'Age_Diag', 'BMI', 'BSA', 'Genre', 'Epaiss_max', 
                 'Gradient', 'TVNS', 'FEVG', 'ATCD_MS', 'SYNCOPE', 'Diam_OG']
SNP_COLS = [f'SNP{i}' for i in range(1, 289)]
GENETIC_COLS = ['Variant.Pathogene'] + SNP_COLS
FEATURES_TO_USE = CLINICAL_COLS + GENETIC_COLS

TARGET_SEVERITY = 'OUTCOME SEVERITY'
TARGET_MACE = 'OUTCOME MACE'

MODEL_DIR = 'models'
OUTPUT_DIR = 'output'
DATA_DIR = 'data'

def load_data():
    """Loads the training and testing data."""
    logging.info("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    logging.info("Data loaded successfully.")
    return train_df, test_df

def get_features(df):
    """Selects and returns the features from the dataframe."""
    return df[FEATURES_TO_USE]

def run_training(n_estimators=1000, n_jobs=2, timestamp=None):
    """
    Main function to run the training and prediction pipeline.
    Accepts n_estimators, n_jobs, and an optional timestamp.
    """
    # --- Timestamp for output files ---
    TIMESTAMP = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Create output and data directories ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Configure logging ---
    # Remove existing handlers to avoid duplication if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, f"training_{TIMESTAMP}.log")),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Starting training with n_estimators={n_estimators} and n_jobs={n_jobs}")

    train_data, test_data = load_data()
    severity_model_path, mace_model_path = train_and_evaluate(train_data, n_estimators, n_jobs, TIMESTAMP)
    predict_and_generate_submission(test_data, severity_model_path, mace_model_path, TIMESTAMP)
    logging.info("Training and prediction pipeline finished.")


def train_and_evaluate(train_df, n_estimators, n_jobs, timestamp):
    """Trains and evaluates models for severity and MACE outcomes."""
    logging.info("--- Training and Evaluating Models ---")
    
    X = get_features(train_df)
    y_severity = train_df[TARGET_SEVERITY]
    y_mace = train_df[TARGET_MACE]

    # Split data for validation
    X_train, X_val, y_severity_train, y_severity_val, y_mace_train, y_mace_val = train_test_split(
        X, y_severity, y_mace, test_size=0.2, random_state=42, stratify=y_mace
    )

    # --- Severity Model ---
    logging.info("Training Severity Model...")
    severity_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced', n_jobs=n_jobs)
    severity_model.fit(X_train, y_severity_train)
    
    # Evaluate Severity Model
    severity_probs_val = severity_model.predict_proba(X_val)[:, 1]
    wll = log_loss(y_severity_val, severity_probs_val, sample_weight=[1.5 if c == 0 else 1 for c in y_severity_val])
    logging.info(f"Validation Weighted Log Loss (Severity): {wll:.4f}")

    # --- MACE Model ---
    logging.info("Training MACE Model...")
    mace_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, class_weight='balanced', n_jobs=n_jobs)
    mace_model.fit(X_train, y_mace_train)

    # Evaluate MACE Model
    mace_preds_val = mace_model.predict(X_val)
    qwk = cohen_kappa_score(y_mace_val, mace_preds_val, weights='quadratic')
    logging.info(f"Validation Quadratic Weighted Kappa (MACE): {qwk:.4f}")

    # --- Save Models ---
    logging.info("Saving models...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    severity_model_path = os.path.join(MODEL_DIR, f'severity_model_{timestamp}.joblib')
    mace_model_path = os.path.join(MODEL_DIR, f'mace_model_{timestamp}.joblib')

    joblib.dump(severity_model, severity_model_path)
    joblib.dump(mace_model, mace_model_path)
    logging.info(f"Models saved to {severity_model_path} and {mace_model_path}")
    
    return severity_model_path, mace_model_path

def predict_and_generate_submission(test_df, severity_model_path, mace_model_path, timestamp):
    """Loads trained models and generates predictions for the test set."""
    logging.info("--- Generating Submission File ---")
    
    if not os.path.exists(severity_model_path) or not os.path.exists(mace_model_path):
        logging.error("Models not found. Please train the models first.")
        return

    # Load Models
    logging.info("Loading models...")
    severity_model = joblib.load(severity_model_path)
    mace_model = joblib.load(mace_model_path)

    # Prepare test data
    X_test = get_features(test_df)

    # Make predictions
    logging.info("Making predictions on the test set...")
    severity_probabilities = severity_model.predict_proba(X_test)[:, 1]
    mace_predictions = mace_model.predict(X_test)

    # Create submission file
    submission_df = pd.DataFrame({'trustii_id': test_df['trustii_id']})
    submission_df['OUTCOME_SEVERITY'] = severity_probabilities
    submission_df['OUTCOME_MACE'] = mace_predictions
    submission_df['OUTCOME_MACE'] = submission_df['OUTCOME_MACE'].astype(int)
    
    submission_path = os.path.join(OUTPUT_DIR, f'submission_{timestamp}.csv')
    submission_df.to_csv(submission_path, index=False)
    logging.info(f"Submission file created at: {submission_path}")
    logging.info(f"Submission head:\n{submission_df.head()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate models.")
    parser.add_argument("--n_estimators", type=int, default=1000, help="Number of estimators for RandomForest.")
    parser.add_argument("--n_jobs", type=int, default=2, help="Number of CPU cores to use.")
    args = parser.parse_args()

    run_training(n_estimators=args.n_estimators, n_jobs=args.n_jobs)