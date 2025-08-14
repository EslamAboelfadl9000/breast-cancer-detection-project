import pandas as pd
import os

# --- Configuration ---
CSV_INPUT_DIR = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"

# --- Main Debugging Logic ---

def debug_cropped_image_link(description_filename: str):
    """
    This function debugs the link between the full mammogram and the cropped image
    using the proven multi-merge strategy.
    
    Args:
        description_filename: The name of the description file to test.
    """
    print(f"\n======================================================================")
    print(f"--- Running Cropped Image Debug Test for: {description_filename} ---")
    print(f"======================================================================\n")

    # 1. Load files
    try:
        dicom_info_df = pd.read_csv(os.path.join(CSV_INPUT_DIR, "dicom_info.csv"))
        description_df = pd.read_csv(os.path.join(CSV_INPUT_DIR, description_filename))
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a necessary file: {e.filename}")
        return

    # 2. Create the universal Path Finder map from dicom_info.csv
    path_finder_map = dicom_info_df[['SeriesInstanceUID', 'image_path']].copy()
    print(f"Step 1: Loaded 'dicom_info.csv' with {len(path_finder_map)} total path records.")

    # 3. Prepare the description file by extracting UIDs for each image type
    def extract_uid(path):
        try:
            return path.split('/')[-2]
        except (TypeError, IndexError):
            return None

    description_df['full_image_SeriesUID'] = description_df['image file path'].apply(extract_uid)
    description_df['cropped_image_SeriesUID'] = description_df['cropped image file path'].apply(extract_uid)
    print("Step 2: Extracted separate Series UIDs for full and cropped images.")
    
    # 4. Perform TWO separate MERGES to get the final paths
    temp_df = description_df.copy()

    # Merge 1: Link the full image path
    merged_df = pd.merge(
        temp_df,
        path_finder_map.rename(columns={'image_path': 'linked_full_image_path'}),
        left_on='full_image_SeriesUID',
        right_on='SeriesInstanceUID',
        how='left'
    )

    # Merge 2: Link the cropped image path
    final_df = pd.merge(
        merged_df,
        path_finder_map.rename(columns={'image_path': 'linked_cropped_path'}),
        left_on='cropped_image_SeriesUID',
        right_on='SeriesInstanceUID',
        how='left',
        suffixes=('_full_merge', '_cropped_merge')
    )
    print("Step 3: Correctly linked paths using two separate pd.merge() operations.")

    # 5. Report results
    total_cases = len(final_df)
    successful_full_links = final_df['linked_full_image_path'].notna().sum()
    successful_cropped_links = final_df['linked_cropped_path'].notna().sum()

    print("\n--- CROPPED IMAGE DEBUGGING RESULTS ---")
    print(f"Total cases analyzed: {total_cases}")
    print(f"Successfully linked FULL IMAGE paths:   {successful_full_links} / {total_cases} ({successful_full_links/total_cases:.1%})")
    print(f"Successfully linked CROPPED IMAGE paths: {successful_cropped_links} / {total_cases} ({successful_cropped_links/total_cases:.1%})")
    
    print("\n--- Sample of Successfully Linked Data ---")
    fully_linked_df = final_df[final_df['linked_full_image_path'].notna() & final_df['linked_cropped_path'].notna()]
    print(fully_linked_df[['patient_id', 'linked_full_image_path', 'linked_cropped_path']].head())


if __name__ == "__main__":
    # Run the test for both a Mass file and a Calcification file
    debug_cropped_image_link("mass_case_description_train_set.csv")
    debug_cropped_image_link("calc_case_description_train_set.csv")