import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# 1. Path to the ROOT of your raw data (where the 'CBIS-DDSM' folder is)
#    This is the most important path to set correctly.
BASE_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw"

# 2. Path to your PROCESSED CSV directory
PROCESSED_CSV_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 3. Target size for resizing
IMG_SIZE = 512

# --- File Paths ---
# Use one of the new, final CSV files. Let's use the mass set as an example.
#csv_path = os.path.join(PROCESSED_CSV_DIR, "mass_specialist_set_final.csv")
# we can also use the calcification set similarly if needed.
#csv_path = os.path.join(PROCESSED_CSV_DIR, "calc_specialist_set_final.csv")
# we can also use the combined set if needed.
csv_path = os.path.join(PROCESSED_CSV_DIR, "combined_specialist_set.csv")


# --- Preprocessing and Visualization Function ---
def preprocess_and_visualize(row):
    """
    Loads, preprocesses, and visualizes a single image-mask pair from a dataframe row.
    This version correctly handles relative paths from the new CSVs.
    """
    try:
        # --- Step 1: Load the Correct Images using Relative Paths ---
        # Get the relative paths from the correct columns
        cropped_relative_path = row['full_image_path']
        mask_relative_path = row['roi_path']

        # Construct the full, absolute path to the image files
        cropped_full_path = os.path.join(BASE_DATA_DIR, cropped_relative_path)
        mask_full_path = os.path.join(BASE_DATA_DIR, mask_relative_path)

        # Load images using OpenCV
        original_image = cv2.imread(cropped_full_path, cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if images were loaded successfully
        if original_image is None or original_mask is None:
            print(f"Error loading images for patient {row['patient_id']}.")
            print(f"  - Tried to load image from: {cropped_full_path}")
            print(f"  - Tried to load mask from: {mask_full_path}")
            return

    except Exception as e:
        print(f"An error occurred during file loading for patient {row['patient_id']}: {e}")
        return

    # --- Step 2: Resize Image and Mask ---
    resized_image = cv2.resize(original_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(original_mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    # --- Step 3: Enhance Image Contrast with CLAHE ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(resized_image)

    # --- Step 4: Normalize and Binarize ---
    normalized_image = clahe_image / 255.0
    _, binarized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Patient: {row['patient_id']} | Type: {row['abnormality_type']} | Pathology: {row['simple_pathology']}", fontsize=16)
    
    axes[0].imshow(original_image, cmap='gray'); axes[0].set_title('1. Original Cropped Image'); axes[0].axis('off')
    axes[1].imshow(original_mask, cmap='gray'); axes[1].set_title('2. Original Mask'); axes[1].axis('off')
    axes[2].imshow(normalized_image, cmap='gray'); axes[2].set_title('3. Processed Image (CLAHE)'); axes[2].axis('off')
    axes[3].imshow(binarized_mask, cmap='gray'); axes[3].set_title('4. Processed Mask'); axes[3].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"visualization_{row['patient_id']}.png")

# --- Main Execution ---
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded '{csv_path}'. Found {len(df)} records.")

    # --- Select diverse samples for visualization ---
    # Taking 1 benign and 1 malignant from the training set, and 1 from the test set.
    samples = [
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'BENIGN')].iloc[0],
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'MALIGNANT')].iloc[0],
        df[df['split'] == 'test'].iloc[0]
    ]

    # --- Loop through samples and visualize ---
    for i, sample_row in enumerate(samples):
        print(f"\n--- Visualizing Sample {i+1} ---")
        preprocess_and_visualize(sample_row)

except FileNotFoundError:
    print(f"Error: Processed CSV file not found. Please ensure '{csv_path}' exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")