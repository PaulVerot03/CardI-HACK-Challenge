import pandas as pd

# Load the datasets
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'test.csv' are in the same directory.")
    exit()

print("\n--- Train DataFrame Info ---")
train_df.info()

print("\n--- Train DataFrame Description ---")
print(train_df.describe())

print("\n--- Train DataFrame Missing Values ---")
print(train_df.isnull().sum()[train_df.isnull().sum() > 0])

print("\n--- Test DataFrame Info ---")
test_df.info()

print("\n--- Test DataFrame Description ---")
print(test_df.describe())

print("\n--- Test DataFrame Missing Values ---")
print(test_df.isnull().sum()[test_df.isnull().sum() > 0])

# Identify categorical and numerical columns (initial guess)
# Clinical columns (excluding ID and outcomes)
clinical_cols = ['Age_Baseline', 'Age_Diag', 'BMI', 'BSA', 'Genre', 'Epaiss_max', 
                 'Gradient', 'TVNS', 'FEVG', 'ATCD_MS', 'SYNCOPE', 'Diam_OG']

# Genetic data (SNPs and Variant.Pathogene)
snp_cols = [col for col in train_df.columns if col.startswith('SNP')]
genetic_cols = ['Variant.Pathogene'] + snp_cols

# Target variables
target_cols = ['OUTCOME SEVERITY', 'OUTCOME MACE']

# Separate features from target for training data
X_train = train_df[clinical_cols + genetic_cols]
y_train = train_df[target_cols]
X_test = test_df[clinical_cols + genetic_cols]

print("\n--- X_train Sample (first 5 rows) ---")
print(X_train.head())

print("\n--- y_train Sample (first 5 rows) ---")
print(y_train.head())

print("\n--- X_test Sample (first 5 rows) ---")
print(X_test.head())

# Check unique values for 'Genre' and 'Variant.Pathogene'
print("\nUnique values in 'Genre' (train_df):", train_df['Genre'].unique())
print("Unique values in 'Variant.Pathogene' (train_df):", train_df['Variant.Pathogene'].unique())
print("Unique values in 'Genre' (test_df):", test_df['Genre'].unique())
print("Unique values in 'Variant.Pathogene' (test_df):", test_df['Variant.Pathogene'].unique())
