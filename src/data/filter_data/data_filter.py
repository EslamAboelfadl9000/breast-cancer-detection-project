import pandas as pd
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
# 1. Path to your raw data directory
input_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"

# 2. Path to where you want to save the new, filtered CSV files
output_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 3. Validation set size for the training data
VALIDATION_SIZE = 0.20
# 4. Random state for reproducibility of the split
RANDOM_STATE = 42

# --- Create the output directory if it doesn't exist ---
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

# --- File Paths ---
mass_train_path = os.path.join(input_directory, "mass_case_description_train_set.csv")
calc_train_path = os.path.join(input_directory, "calc_case_description_train_set.csv")
mass_test_path = os.path.join(input_directory, "mass_case_description_test_set.csv")
calc_test_path = os.path.join(input_directory, "calc_case_description_test_set.csv")

# --- Data Processing Functions ---

def process_data(file_path, abnormality_name, is_test_set=False):
    """
    Loads a dataset, applies filtering, and either splits it (for training data)
    or just cleans it (for test data).
    """
    print(f"\nProcessing {abnormality_name} {'Test' if is_test_set else 'Train/Val'} dataset...")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check your input_directory path.")
        return None

    # 1. Clean column names
    df.columns = df.columns.str.strip()
    
    # 2. Clean 'assessment' column and convert to numeric, handling errors
    df['assessment'] = pd.to_numeric(df['assessment'].astype(str).str.strip(), errors='coerce')
    original_rows = len(df)
    print(f"Original number of rows: {original_rows}")

    # 3. Filter by assessment scores (2, 4, 5)
    assessment_filter = [2, 4, 5]
    df_filtered = df[df['assessment'].isin(assessment_filter)].copy()
    print(f"Rows after filtering by assessment [2, 4, 5]: {len(df_filtered)}")

    # 4. Simplify the pathology label
    pathology_mapping = {
        'MALIGNANT': 'MALIGNANT',
        'BENIGN': 'BENIGN',
        'BENIGN_WITHOUT_CALLBACK': 'BENIGN'
    }
    df_filtered['pathology'] = df_filtered['pathology'].astype(str).str.strip()
    df_filtered['simple_pathology'] = df_filtered['pathology'].map(pathology_mapping)
    
    # 5. Add abnormality type column
    df_filtered['abnormality_type'] = abnormality_name
    
    # 6. Add a 'split' column based on whether it's test data or not
    if is_test_set:
        df_filtered['split'] = 'test'
    else:
        # Perform Stratified Train-Validation Split for training data
        X = df_filtered
        y = df_filtered['simple_pathology']
        _, val_indices = train_test_split(X.index, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_STATE)
        df_filtered['split'] = 'train'
        df_filtered.loc[val_indices, 'split'] = 'validation'
    
    print("Processing complete.")
    return df_filtered

# --- Main Execution ---

# Process Training/Validation Sets
mass_train_val_df = process_data(mass_train_path, "Mass", is_test_set=False)
calc_train_val_df = process_data(calc_train_path, "Calcification", is_test_set=False)

# Process Test Sets
mass_test_df = process_data(mass_test_path, "Mass", is_test_set=True)
calc_test_df = process_data(calc_test_path, "Calcification", is_test_set=True)

# Combine and Save the final datasets
if all(df is not None for df in [mass_train_val_df, calc_train_val_df, mass_test_df, calc_test_df]):
    # Combine all mass data (train/val/test) and save
    mass_final_df = pd.concat([mass_train_val_df, mass_test_df], ignore_index=True)
    mass_output_path = os.path.join(output_directory, "mass_specialist_set.csv")
    mass_final_df.to_csv(mass_output_path, index=False)
    print(f"\nSaved final Mass set to: {mass_output_path}")
    print("Final Mass split distribution:\n", mass_final_df['split'].value_counts())

    # Combine all calcification data (train/val/test) and save
    calc_final_df = pd.concat([calc_train_val_df, calc_test_df], ignore_index=True)
    calc_output_path = os.path.join(output_directory, "calc_specialist_set.csv")
    calc_final_df.to_csv(calc_output_path, index=False)
    print(f"\nSaved final Calcification set to: {calc_output_path}")
    print("Final Calcification split distribution:\n", calc_final_df['split'].value_counts())

    # Combine all data from both types and save
    final_combined_df = pd.concat([mass_final_df, calc_final_df], ignore_index=True)
    combined_output_path = os.path.join(output_directory, "combined_specialist_set.csv")
    final_combined_df.to_csv(combined_output_path, index=False)
    print(f"\nSaved final Combined set to: {combined_output_path}")
    print("Overall final split distribution:\n", final_combined_df['split'].value_counts())


print("\n--- Script Finished ---")
