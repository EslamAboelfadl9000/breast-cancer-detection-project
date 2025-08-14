import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
BASE_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw"
PROCESSED_CSV_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"
IMG_SIZE = 512

# Use one of the new CSVs. Let's test the calcification set this time.
# csv_path = os.path.join(PROCESSED_CSV_DIR, "calc_final_dataset.csv")
# csv_path = os.path.join(PROCESSED_CSV_DIR, "mass_final_dataset.csv")
csv_path = os.path.join(PROCESSED_CSV_DIR, "combined_final_dataset.csv")

def create_placeholder_image(text, size=(IMG_SIZE, IMG_SIZE)):
    """Creates a black image with centered white text."""
    placeholder = np.zeros(size, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (size[1] - text_size[0]) // 2
    text_y = (size[0] + text_size[1]) // 2
    cv2.putText(placeholder, text, (text_x, text_y), font, 1, (255, 255, 255), 2)
    return placeholder

def preprocess_and_visualize_all(row):
    """
    Loads, preprocesses, and visualizes a full, cropped (if available), and mask image.
    """
    # --- Load Images ---
    full_img, cropped_img, mask_img = None, None, None
    
    # Load Full Image
    try:
        full_img_path = os.path.join(BASE_DATA_DIR, row['full_image_path'])
        full_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
    except Exception: pass
        
    # Load Mask Image
    try:
        mask_path = os.path.join(BASE_DATA_DIR, row['roi_path'])
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    except Exception: pass

    # Load Cropped Image (if path exists)
    if pd.notna(row['cropped_image_path']):
        try:
            cropped_path = os.path.join(BASE_DATA_DIR, row['cropped_image_path'])
            cropped_img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
        except Exception: pass
    
    # --- Preprocess Images ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Process Full Image
    processed_full = clahe.apply(cv2.resize(full_img, (IMG_SIZE, IMG_SIZE))) if full_img is not None else create_placeholder_image("Full Image Not Found")
    
    # Process Cropped Image
    if cropped_img is not None:
        processed_cropped = clahe.apply(cv2.resize(cropped_img, (IMG_SIZE, IMG_SIZE)))
    else:
        processed_cropped = create_placeholder_image("Cropped Not Available")

    # Process Mask
    if mask_img is not None:
        _, processed_mask = cv2.threshold(cv2.resize(mask_img, (IMG_SIZE, IMG_SIZE)), 127, 255, cv2.THRESH_BINARY)
    else:
        processed_mask = create_placeholder_image("Mask Not Found")

    # --- Plotting in a 2x3 Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Patient: {row['patient_id']} | Type: {row['abnormality_type']} | Pathology: {row['simple_pathology']}", fontsize=18)
    
    # Top Row: Original Images
    axes[0, 0].imshow(full_img if full_img is not None else create_placeholder_image("Full Image Not Found"), cmap='gray')
    axes[0, 0].set_title('1. Original Full Image', fontsize=14)

    axes[0, 1].imshow(cropped_img if cropped_img is not None else create_placeholder_image("Cropped Not Available"), cmap='gray')
    axes[0, 1].set_title('2. Original Cropped Image', fontsize=14)
    
    axes[0, 2].imshow(mask_img if mask_img is not None else create_placeholder_image("Mask Not Found"), cmap='gray')
    axes[0, 2].set_title('3. Original ROI Mask', fontsize=14)
    
    # Bottom Row: Processed Images
    axes[1, 0].imshow(processed_full, cmap='gray')
    axes[1, 0].set_title('4. Processed Full (CLAHE)', fontsize=14)

    axes[1, 1].imshow(processed_cropped, cmap='gray')
    axes[1, 1].set_title('5. Processed Cropped (CLAHE)', fontsize=14)
    
    axes[1, 2].imshow(processed_mask, cmap='gray')
    axes[1, 2].set_title('6. Processed Mask (Binarized)', fontsize=14)
    
    for ax in axes.flat:
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"visualization_{row['patient_id']}.png")

# --- Main Execution ---
try:
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded '{os.path.basename(csv_path)}'. Found {len(df)} records.")

    # Select a random sample from the dataframe to visualize
    if not df.empty:
        random_sample = df.sample(n=1).iloc[0]
        print("\n--- Visualizing a Random Sample ---")
        preprocess_and_visualize_all(random_sample)
    else:
        print("CSV file is empty. No sample to visualize.")

except FileNotFoundError:
    print(f"Error: Processed CSV file not found. Please ensure '{csv_path}' exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")