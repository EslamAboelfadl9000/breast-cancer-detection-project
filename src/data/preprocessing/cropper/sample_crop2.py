import pandas as pd
import cv2
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

# --- Configuration ---
# 1. Path to the ROOT of your raw data (where the 'CBIS-DDSM' folder is)
BASE_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw"

# 2. Path to your PROCESSED CSV directory
PROCESSED_CSV_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 3. Where to save the new, augmented dataset
AUGMENTED_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\augmented_dataset_v2"

# 4. The final, clean CSV file to use as a source
SOURCE_CSV_PATH = os.path.join(PROCESSED_CSV_DIR, "combined_final_dataset.csv")

# --- Augmentation Parameters ---
# Total number of original train/val images you want to process to create the augmented set.
TOTAL_IMAGES_TO_PROCESS = 120

# Number of random crops to generate per original image.
CROPS_PER_IMAGE = 10

# The fixed size for all new images and masks
IMG_SIZE = (512, 512)

# --- Test Set Parameters ---
# The number of pre-labeled test images to copy to the test set (max= 259)
NUM_TEST_IMAGES = 20

# --- Helper Functions ---
def get_mask_bbox(mask_img):
    """Finds the bounding box of the non-zero regions in a mask."""
    contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, w, h)

def create_crop(image, center_x, center_y, size):
    """Creates a crop of a specific size centered at a given coordinate."""
    h, w = image.shape
    crop_w, crop_h = size
    
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    
    crop = image[y1:y2, x1:x2]
    
    padded_crop = np.zeros(size, dtype=image.dtype)
    padded_crop[:crop.shape[0], :crop.shape[1]] = crop
    return padded_crop

def balance_data(df, total_images):
    """
    Balances the dataset by sampling from each combination of
    assessment, pathology, and abnormality type.
    """
    balance_on_cols = ['assessment', 'simple_pathology', 'abnormality_type']

    if not all(col in df.columns for col in balance_on_cols):
        raise ValueError(f"One or more balancing columns not found in the dataframe. Required columns: {balance_on_cols}")

    num_groups = df[balance_on_cols].drop_duplicates().shape[0]

    if num_groups == 0:
        print("Warning: No groups found to balance on. Returning an empty dataframe.")
        return pd.DataFrame()

    samples_per_group = total_images // num_groups
    
    print(f"Balancing data across {num_groups} unique groups.")
    print(f"Aiming for approximately {samples_per_group} samples per group.")

    balanced_df = df.groupby(balance_on_cols).apply(
        lambda x: x.sample(n=min(len(x), samples_per_group), replace=False)
    ).reset_index(drop=True)

    print(f"Created a balanced train/val dataset with {len(balanced_df)} samples.")
    return balanced_df


# --- Main Preprocessing Logic ---
def create_augmented_dataset():
    """
    Reads the source CSV, generates augmented offline crops, 
    and saves a new CSV with train, validation, and test sets.
    """
    # 1. Create output directories
    for split in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(AUGMENTED_DATA_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(AUGMENTED_DATA_DIR, split, 'masks'), exist_ok=True)

    # 2. Load the source data
    source_df = pd.read_csv(SOURCE_CSV_PATH)
    
    new_dataset_records = []

    # --- NEW: Process the pre-defined test set ---
    predefined_test_df = source_df[source_df['split'] == 'test'].copy()
    
    # Take a sample of the predefined test images
    if len(predefined_test_df) > 0:
        test_df = predefined_test_df.sample(n=min(NUM_TEST_IMAGES, len(predefined_test_df)), random_state=42)
        print(f"\nFound {len(predefined_test_df)} pre-labeled test images. Selecting {len(test_df)} to process.")
        
        for index, row in test_df.iterrows():
            try:
                # Copy the original full image and ROI mask without changes
                shutil.copy(os.path.join(BASE_DATA_DIR, row['full_image_path']), os.path.join(AUGMENTED_DATA_DIR, 'test', 'images'))
                shutil.copy(os.path.join(BASE_DATA_DIR, row['roi_path']), os.path.join(AUGMENTED_DATA_DIR, 'test', 'masks'))
                
                record = row.to_dict()
                record['image_path'] = os.path.join('test', 'images', os.path.basename(row['full_image_path']))
                record['mask_path'] = os.path.join('test', 'masks', os.path.basename(row['roi_path']))
                new_dataset_records.append(record)
            except Exception as e:
                print(f"Warning: Could not process patient {row.get('patient_id', 'Unknown')} for test set. Reason: {e}")
    else:
        print("Warning: No pre-labeled test images found in the source CSV.")

    # --- Process Train/Validation data ---
    train_val_source_df = source_df[source_df['split'] != 'test'].copy()

    # Filter for the assessments we are interested in for training/validation
    train_val_source_df = train_val_source_df[train_val_source_df['assessment'].isin([2, 4, 5])]
    
    # Balance the train/validation data
    train_val_df = balance_data(train_val_source_df, TOTAL_IMAGES_TO_PROCESS)

    # Process the balanced train/validation set for augmentation
    for index, row in train_val_df.iterrows():
        try:
            image_path = os.path.join(BASE_DATA_DIR, row['full_image_path'])
            mask_path = os.path.join(BASE_DATA_DIR, row['roi_path'])

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            bbox = get_mask_bbox(mask)
            if not bbox:
                continue

            (x, y, w, h) = bbox
            center_x, center_y = x + w // 2, y + h // 2
            
            # Use the original split ('train' or 'validation') from the source CSV
            split_folder = row['split']

            for i in range(CROPS_PER_IMAGE):
                if np.random.rand() < 0.7:
                    cx = int(np.random.normal(center_x, w * 0.25))
                    cy = int(np.random.normal(center_y, h * 0.25))
                else:
                    cx = np.random.randint(0, image.shape[1])
                    cy = np.random.randint(0, image.shape[0])

                cropped_image = create_crop(image, cx, cy, IMG_SIZE)
                cropped_mask = create_crop(mask, cx, cy, IMG_SIZE)
                
                flipped_image = cv2.flip(cropped_image, 1)
                flipped_mask = cv2.flip(cropped_mask, 1)

                base_filename = f"{row['patient_id']}_{index}_crop{i}"
                
                # Process and save normal/flipped versions
                for aug_type, aug_img, aug_mask in [('normal', cropped_image, cropped_mask), ('flipped', flipped_image, flipped_mask)]:
                    img_name = f"{base_filename}_{aug_type}.png"
                    mask_name = f"{base_filename}_{aug_type}_mask.png"
                    
                    cv2.imwrite(os.path.join(AUGMENTED_DATA_DIR, split_folder, 'images', img_name), aug_img)
                    cv2.imwrite(os.path.join(AUGMENTED_DATA_DIR, split_folder, 'masks', mask_name), aug_mask)
                    
                    record = row.to_dict()
                    record['split'] = split_folder
                    record['image_path'] = os.path.join(split_folder, 'images', img_name)
                    record['mask_path'] = os.path.join(split_folder, 'masks', mask_name)
                    new_dataset_records.append(record)

        except Exception as e:
            print(f"Warning: Could not process patient {row.get('patient_id', 'Unknown')}. Reason: {e}")

    # Save the final CSV file
    new_df = pd.DataFrame(new_dataset_records)
    output_csv_path = os.path.join(AUGMENTED_DATA_DIR, "augmented_dataset.csv")
    new_df.to_csv(output_csv_path, index=False)
    
    print("\n--- Script Finished ---")
    print(f"Augmented dataset created at: {AUGMENTED_DATA_DIR}")
    if not new_df.empty:
        print("New dataset distribution:\n", new_df['split'].value_counts())

if __name__ == "__main__":
    create_augmented_dataset()