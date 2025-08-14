import pandas as pd
import os

# --- Configuration ---
CSV_INPUT_DIR = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"

# --- Main Debugging Logic ---

def ultimate_debug_test(description_filename: str):
    """
    This function uses the correct pd.merge() method to link description files
    to their final JPEG paths from dicom_info.csv.
    """
    print(f"\n======================================================================")
    print(f"--- Running Ultimate Debug Test for: {description_filename} ---")
    print(f"======================================================================\n")

    # 1. Load files
    try:
        dicom_info_df = pd.read_csv(os.path.join(CSV_INPUT_DIR, "dicom_info.csv"))
        description_df = pd.read_csv(os.path.join(CSV_INPUT_DIR, description_filename))
    except FileNotFoundError as e:
        print(f"ERROR: Could not find a necessary file: {e.filename}")
        return

    # 2. Prepare the Path Finder map. We no longer set an index.
    # This dataframe contains all possible paths. We don't drop any duplicates.
    path_finder_map = dicom_info_df[['SeriesInstanceUID', 'image_path']].copy()
    print(f"Step 1: Loaded 'dicom_info.csv' with {len(path_finder_map)} total path records.")

    # 3. Prepare the description file by extracting the specific UIDs
    def extract_uid(path):
        try:
            return path.split('/')[-2]
        except (TypeError, IndexError):
            return None

    description_df['full_image_SeriesUID'] = description_df['image file path'].apply(extract_uid)
    description_df['roi_mask_SeriesUID'] = description_df['ROI mask file path'].apply(extract_uid)
    print("Step 2: Extracted separate Series UIDs from the description file.")
    
    # 4. Perform TWO separate MERGES to get the final paths
    
    # Create a temporary dataframe for merging to avoid column name conflicts
    temp_df = description_df.copy()

    # Merge 1: Link the full image path
    # We rename the column to avoid conflicts in the next merge
    merged_df = pd.merge(
        temp_df,
        path_finder_map.rename(columns={'image_path': 'linked_full_image_path'}),
        left_on='full_image_SeriesUID',
        right_on='SeriesInstanceUID',
        how='left'
    )

    # Merge 2: Link the ROI mask path
    # Now merge the result with the path finder map again for the ROI
    final_df = pd.merge(
        merged_df,
        path_finder_map.rename(columns={'image_path': 'linked_roi_path'}),
        left_on='roi_mask_SeriesUID',
        right_on='SeriesInstanceUID',
        how='left',
        suffixes=('_full_merge', '_roi_merge') # Handles redundant 'SeriesInstanceUID' columns
    )
    print("Step 3: Correctly linked paths using two separate pd.merge() operations.")

    # 5. Report results
    total_cases = len(final_df)
    # Check for non-empty strings in the linked path columns
    successful_full_links = final_df['linked_full_image_path'].notna().sum()
    successful_roi_links = final_df['linked_roi_path'].notna().sum()

    print("\n--- ULTIMATE DEBUGGING RESULTS ---")
    print(f"Total cases analyzed: {total_cases}")
    print(f"Successfully linked FULL IMAGE paths: {successful_full_links} / {total_cases} ({successful_full_links/total_cases:.1%})")
    print(f"Successfully linked ROI MASK paths:   {successful_roi_links} / {total_cases} ({successful_roi_links/total_cases:.1%})")
    
    print("\n--- Sample of Successfully Linked Data ---")
    # Show cases where both links were successful
    fully_linked_df = final_df[final_df['linked_full_image_path'].notna() & final_df['linked_roi_path'].notna()]
    print(fully_linked_df[['patient_id', 'linked_full_image_path', 'linked_roi_path']].head())


if __name__ == "__main__":
    ultimate_debug_test("mass_case_description_train_set.csv")
    ultimate_debug_test("calc_case_description_train_set.csv")