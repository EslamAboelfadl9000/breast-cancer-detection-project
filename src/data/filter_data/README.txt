# CBIS-DDSM Data Preprocessing and Subsetting Script

## 1. Overview

This directory contains a Python script (`cbis_ddsm_data_filtering_script.py`) designed to preprocess the raw metadata from the Curated Breast Imaging Subset of DDSM (CBIS-DDSM).

The primary purpose of this script is to transform the original, complex CSV files into clean, well-structured, and analysis-ready datasets. It implements a specific, clinically-informed strategy to create high-quality training, validation, and test sets, which are ideal for training robust machine learning models for breast cancer detection.

The script generates three primary output files in the `processed/` directory:
- `mass_specialist_set.csv`: A clean dataset for training a model specialized in mass abnormalities.
- `calc_specialist_set.csv`: A clean dataset for training a model specialized in calcification abnormalities.
- `combined_specialist_set.csv`: A convenience file containing all processed data.

---

## 2. The Preprocessing Strategy (The "Why")

Raw medical imaging datasets are rarely ready for direct use. Our strategy is designed to address key challenges like data ambiguity, class imbalance, and the need for reproducible results.

### a. Specialist Model Approach

Instead of building one "generalist" model, our strategy is to build two "specialist" models: one for masses and one for calcifications.

* **Rationale:** Radiologists use different visual criteria to evaluate masses (shape, margins) versus calcifications (morphology, distribution). By training two separate models, each can become an expert on its specific abnormality type. This leads to higher potential accuracy and more interpretable results (e.g., heatmaps that highlight clinically relevant features).

### b. Filtering by Clinical Assessment (BI-RADS)

The script filters the data based on the `assessment` column, which corresponds to the BI-RADS score given by a radiologist.

* **Action:** We **keep only cases with assessment scores of 2, 4, and 5.**
* **Rationale:**
    * **Scores 2 (Benign) & 5 (Highly Suggestive of Malignancy):** These are high-confidence cases that provide the model with clear, unambiguous examples of what is and isn't cancer.
    * **Score 4 (Suspicious):** These are the most challenging and important cases, where a radiologist is uncertain. Including these is crucial for building a model that can perform well in real-world clinical scenarios.
    * **Scores 0 (Incomplete), 1 (Negative), & 3 (Probably Benign):** These are excluded because they represent incomplete data, are statistically insignificant, or introduce a strong bias towards benign cases without adding significant diagnostic challenge.

### c. Reproducible, Stratified Splits

To train and evaluate a model reliably, we need separate datasets for training, validation, and testing.

* **Action:** The script creates a `split` column, dividing the data into `train`, `validation`, and `test` sets.
* **Rationale:**
    * **Reproducibility:** By saving the split into the CSV file (using a fixed `RANDOM_STATE`), we ensure that every training run uses the exact same data, making our results reproducible and verifiable.
    * **Stratified Split:** The split is *stratified* based on the `simple_pathology` label. This guarantees that the proportion of benign and malignant cases is the same across the training and validation sets, preventing bias and ensuring the validation accuracy is a true reflection of the model's performance.
    * **Test Set Integrity:** The original test set provided by CBIS-DDSM is processed with the *exact same filtering logic* as the training set. This is critical for a fair "apples-to-apples" evaluation of the final trained model.

---

3. Using the Processed Data in a Model
The output CSVs are now ready to be used in your model training pipeline (e.g., in PyTorch or TensorFlow). The key is to use the split column to load the correct data for each phase.

Example: Loading Data with Pandas
Here is how you would load the training, validation, and test sets for the mass specialist model:

import pandas as pd

# Load the processed data
processed_mass_path = 'processed/mass_specialist_set.csv'
df = pd.read_csv(processed_mass_path)

# Create dataframes for each split
train_df = df[df['split'] == 'train'].copy()
validation_df = df[df['split'] == 'validation'].copy()
test_df = df[df['split'] == 'test'].copy()

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(validation_df)}")
print(f"Test samples: {len(test_df)}")

# You can now pass these dataframes to your data loader/generator.
