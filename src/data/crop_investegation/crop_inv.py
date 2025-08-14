import pandas as pd
import cv2
import numpy as np
import os

# --- Configuration ---
# 1. Path to the ROOT of your raw data (where the 'CBIS-DDSM' folder is)
BASE_DATA_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw"

# 2. Path to your PROCESSED CSV directory
PROCESSED_CSV_DIR = r"c:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"

# 3. The final, clean CSV you want to analyze
# You can change this to "mass_final_dataset.csv" as well
CSV_TO_ANALYZE = "calc_final_dataset.csv"

# --- Main Investigation Logic ---

def analyze_image_properties():
    """
    Analyzes the geometric properties of images listed in a dataset CSV.
    """
    csv_path = os.path.join(PROCESSED_CSV_DIR, CSV_TO_ANALYZE)
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Analysis failed. Could not find '{csv_path}'.")
        return

    print(f"--- Starting Analysis of '{CSV_TO_ANALYZE}' ---")
    
    analysis_results = []
    
    # Iterate through each row of the dataframe to gather image data
    for index, row in df.iterrows():
        record = {'patient_id': row['patient_id']}
        
        # Analyze Full Image
        try:
            full_path = os.path.join(BASE_DATA_DIR, row['full_image_path'])
            full_img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            record['full_dims'] = full_img.shape # (height, width)
        except Exception:
            record['full_dims'] = None
            
        # Analyze Mask Image
        try:
            mask_path = os.path.join(BASE_DATA_DIR, row['roi_path'])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            record['mask_dims'] = mask_img.shape
        except Exception:
            record['mask_dims'] = None

        # Analyze Cropped Image (if it exists)
        if pd.notna(row['cropped_image_path']):
            try:
                cropped_path = os.path.join(BASE_DATA_DIR, row['cropped_image_path'])
                cropped_img = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
                record['cropped_dims'] = cropped_img.shape
            except Exception:
                record['cropped_dims'] = None
        else:
            record['cropped_dims'] = None
            
        analysis_results.append(record)

    # Convert results to a new dataframe for analysis
    analysis_df = pd.DataFrame(analysis_results)
    
    # Save the detailed log to a new CSV file
    log_path = os.path.join(PROCESSED_CSV_DIR, "image_analysis_log.csv")
    analysis_df.to_csv(log_path, index=False)
    print(f"\n[INFO] Detailed analysis log saved to: {log_path}")

    # --- Print Summary Report ---
    print("\n--- DATASET COMPLETENESS REPORT ---")
    total_records = len(analysis_df)
    with_cropped = analysis_df['cropped_dims'].notna().sum()
    without_cropped = total_records - with_cropped
    print(f"Total Records Analyzed: {total_records}")
    print(f"  - Records with Full + Mask + Cropped: {with_cropped} ({with_cropped/total_records:.1%})")
    print(f"  - Records with Full + Mask only:      {without_cropped} ({without_cropped/total_records:.1%})")

    print("\n--- IMAGE SCALE REPORT (Top 5 most common sizes) ---")
    if with_cropped > 0:
        print("\nCropped Image Dimensions:")
        print(analysis_df['cropped_dims'].value_counts().nlargest(5).to_string())
    print("\nFull Image Dimensions:")
    print(analysis_df['full_dims'].value_counts().nlargest(5).to_string())
    print("\nMask Dimensions:")
    print(analysis_df['mask_dims'].value_counts().nlargest(5).to_string())

    # --- Test "Centered Mask" Hypothesis on a few samples ---
    print("\n--- 'CENTERED MASK' HYPOTHESIS TEST ---")
    samples_to_test = analysis_df[analysis_df['cropped_dims'].notna()].sample(min(3, with_cropped))
    for index, row in samples_to_test.iterrows():
        try:
            # We need to re-load the images for this test
            mask_path = os.path.join(BASE_DATA_DIR, df.loc[index, 'roi_path'])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Find contours and bounding box of the mask
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            mask_center = (x + w//2, y + h//2)
            
            # Get cropped image center
            cropped_dims = row['cropped_dims']
            cropped_center = (cropped_dims[1]//2, cropped_dims[0]//2) # (width/2, height/2)
            
            distance = np.sqrt((mask_center[0] - cropped_center[0])**2 + (mask_center[1] - cropped_center[1])**2)
            
            print(f"\nTesting Patient {row['patient_id']}:")
            print(f"  - Cropped Image Center: {cropped_center}")
            print(f"  - Mask Center: {mask_center}")
            print(f"  - Distance between centers: {distance:.2f} pixels")
        except Exception as e:
            print(f"\nCould not test patient {row['patient_id']}. Reason: {e}")


if __name__ == "__main__":
    analyze_image_properties()