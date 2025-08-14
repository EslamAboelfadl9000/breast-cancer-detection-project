import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# 1. Path to your PROCESSED data directory
processed_data_dir = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 2. Target size for resizing
IMG_SIZE = 512

# --- File Paths ---
# The script now uses the combined CSV which contains full paths to the images.
combined_csv_path = os.path.join(processed_data_dir, "combined_specialist_set.csv")

# --- Preprocessing and Visualization Function ---
def preprocess_and_visualize(row):
    """
    Loads, preprocesses, and visualizes a single image-mask pair from a dataframe row.
    """
    try:
        # --- Step 1: Load the Correct Images ---
        # Get the full paths directly from the CSV
        cropped_img_full_path = row['full_cropped_image_path']
        mask_full_path = row['full_roi_mask_path']

        # Load images using OpenCV
        original_image = cv2.imread(cropped_img_full_path, cv2.IMREAD_GRAYSCALE)
        original_mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if images were loaded successfully
        if original_image is None or original_mask is None:
            print(f"Error loading images for patient {row['patient_id']}.")
            print(f"  - Tried to load image from: {cropped_img_full_path}")
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
    # Normalizing the image for visualization (not strictly necessary but good practice)
    normalized_image = clahe_image / 255.0
    # Binarizing the mask for visualization
    _, binarized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    fig.suptitle(f"Patient: {row['patient_id']} | Type: {row['abnormality_type']} | Pathology: {row['simple_pathology']}", fontsize=16)
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('1. Original Cropped Image')
    axes[0].axis('off')
    
    axes[1].imshow(original_mask, cmap='gray')
    axes[1].set_title('2. Original Mask')
    axes[1].axis('off')

    axes[2].imshow(normalized_image, cmap='gray')
    axes[2].set_title('3. Processed Image (Resized + CLAHE)')
    axes[2].axis('off')
    
    axes[3].imshow(binarized_mask, cmap='gray')
    axes[3].set_title('4. Processed Mask (Resized)')
    axes[3].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("Preprocessing_sample.jpg")
    plt.show()

# --- Main Execution ---
try:
    # Load the combined dataframe
    df = pd.read_csv(combined_csv_path)

    # --- Select 6 diverse samples for visualization ---
    # We'll take 1 benign and 1 malignant from each abnormality type in the training set,
    # and 1 from each abnormality type in the test set.
    samples = [
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'BENIGN') & (df['abnormality_type'] == 'Mass')].iloc[0],
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'MALIGNANT') & (df['abnormality_type'] == 'Mass')].iloc[0],
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'BENIGN') & (df['abnormality_type'] == 'Calcification')].iloc[0],
        df[(df['split'] == 'train') & (df['simple_pathology'] == 'MALIGNANT') & (df['abnormality_type'] == 'Calcification')].iloc[0],
        df[(df['split'] == 'test') & (df['abnormality_type'] == 'Mass')].iloc[0],
        df[(df['split'] == 'test') & (df['abnormality_type'] == 'Calcification')].iloc[0]
    ]

    # --- Loop through samples and visualize ---
    for sample_row in samples:
        preprocess_and_visualize(sample_row)

except FileNotFoundError:
    print("Error: Processed CSV file not found.")
    print(f"Please ensure '{combined_csv_path}' exists.")
    print("You may need to run the data_filter1.py script first.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
