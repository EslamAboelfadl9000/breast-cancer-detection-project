import pandas as pd
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
# 1. Path to the ROOT of your raw data (where the 'CBIS-DDSM' folder is)
BASE_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw"

# 2. Path to your PROCESSED CSV directory
PROCESSED_CSV_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 3. Where to save the new, augmented dataset
AUGMENTED_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\augmented_dataset"

# 4. The final, clean CSV file to use as a source
SOURCE_CSV_PATH = os.path.join(PROCESSED_CSV_DIR, "combined_final_dataset.csv")

# --- Augmentation Parameters ---
# Total number of original images you want to process to create the augmented set.
# Set to a smaller number (e.g., 100) for testing, or a larger number for the full run.
TOTAL_IMAGES_TO_PROCESS = 100 

# Number of random crops to generate per original image.
# This will be doubled because we also create flipped versions.
CROPS_PER_IMAGE = 10

# The fixed size for all new images and masks
IMG_SIZE = (512, 512)

# --- Helper Functions ---
def get_mask_bbox(mask_img):
    """Finds the bounding box of the non-zero regions in a mask."""
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # Combine all contours to get the overall bounding box
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, w, h)

def create_crop(image, center_x, center_y, size):
    """Creates a crop of a specific size centered at a given coordinate."""
    h, w = image.shape
    crop_w, crop_h = size
    
    # Calculate crop boundaries, ensuring they are within the image dimensions
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    
    # Create the crop
    crop = image[y1:y2, x1:x2]
    
    # Pad the crop if it's smaller than the target size (due to being near an edge)
    padded_crop = np.zeros(size, dtype=image.dtype)
    padded_crop[:crop.shape[0], :crop.shape[1]] = crop
    
    return padded_crop

# --- Main Preprocessing Logic ---
def create_augmented_dataset():
    """
    Reads the source CSV, generates augmented offline crops, and saves a new CSV.
    """
    # 1. Create output directories if they don't exist
    augmented_images_dir = os.path.join(AUGMENTED_DATA_DIR, 'images')
    augmented_masks_dir = os.path.join(AUGMENTED_DATA_DIR, 'masks')
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_masks_dir, exist_ok=True)
    
    print("--- Starting Offline Augmentation Script ---")

    # 2. Load the source CSV and take a balanced sample
    try:
        df = pd.read_csv(SOURCE_CSV_PATH)
        # Stratify by abnormality type and pathology to get a well-balanced subset
        _, sample_df = train_test_split(
            df, 
            test_size=min(TOTAL_IMAGES_TO_PROCESS, len(df)),
            stratify=df[['abnormality_type', 'simple_pathology']],
            random_state=42
        )
        print(f"Selected a balanced subset of {len(sample_df)} images to process.")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: Could not load or sample the source CSV. Reason: {e}")
        return

    # 3. Loop through samples and generate augmented crops
    new_dataset_records = []
    
    for index, row in sample_df.iterrows():
        try:
            # Load the original full-resolution image and mask
            full_img_path = os.path.join(BASE_DATA_DIR, row['full_image_path'])
            mask_img_path = os.path.join(BASE_DATA_DIR, row['roi_path'])
            full_img = cv2.imread(full_img_path, cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

            if full_img is None or mask_img is None: continue

            # Get the bounding box of the abnormality
            bbox = get_mask_bbox(mask_img)
            if bbox is None: continue
            
            x, y, w, h = bbox
            
            # Generate multiple crops for this one image
            for i in range(CROPS_PER_IMAGE):
                # --- Smart Random Cropping ---
                # 70% chance to center the crop inside the abnormality, 30% anywhere
                if np.random.rand() < 0.7:
                    # Pick a random point inside the bounding box
                    center_x = np.random.randint(x, x + w)
                    center_y = np.random.randint(y, y + h)
                else:
                    # Pick a random point from the entire image
                    center_x = np.random.randint(0, full_img.shape[1])
                    center_y = np.random.randint(0, full_img.shape[0])

                # Create the crops
                cropped_image = create_crop(full_img, center_x, center_y, IMG_SIZE)
                cropped_mask = create_crop(mask_img, center_x, center_y, IMG_SIZE)

                # --- Augmentation: Horizontal Flip ---
                flipped_image = cv2.flip(cropped_image, 1)
                flipped_mask = cv2.flip(cropped_mask, 1)

                # --- Save the new images and log them ---
                base_filename = f"{row['patient_id']}_{index}_crop{i}"

                # Save normal version
                cv2.imwrite(os.path.join(augmented_images_dir, f"{base_filename}_normal.png"), cropped_image)
                cv2.imwrite(os.path.join(augmented_masks_dir, f"{base_filename}_normal_mask.png"), cropped_mask)
                new_dataset_records.append({'image_path': os.path.join('images', f"{base_filename}_normal.png"), 
                                            'mask_path': os.path.join('masks', f"{base_filename}_normal_mask.png")})
                
                # Save flipped version
                cv2.imwrite(os.path.join(augmented_images_dir, f"{base_filename}_flipped.png"), flipped_image)
                cv2.imwrite(os.path.join(augmented_masks_dir, f"{base_filename}_flipped_mask.png"), flipped_mask)
                new_dataset_records.append({'image_path': os.path.join('images', f"{base_filename}_flipped.png"), 
                                            'mask_path': os.path.join('masks', f"{base_filename}_flipped_mask.png")})

        except Exception as e:
            print(f"Warning: Could not process patient {row['patient_id']}. Reason: {e}")

    # 4. Save the new CSV file with paths to the augmented data
    new_df = pd.DataFrame(new_dataset_records)
    output_csv_path = os.path.join(AUGMENTED_DATA_DIR, "augmented_dataset.csv")
    new_df.to_csv(output_csv_path, index=False)
    
    print("\n--- Script Finished ---")
    print(f"Successfully generated {len(new_df)} augmented samples.")
    print(f"Final training-ready CSV saved to: {output_csv_path}")


if __name__ == "__main__":
    create_augmented_dataset()