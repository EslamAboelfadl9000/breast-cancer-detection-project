import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataFilter:
    def __init__(self, input_dir, output_dir, validation_size=0.20, random_state=42):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.validation_size = validation_size
        self.random_state = random_state

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        self.file_map = {
            "mass_train": "mass_case_description_train_set.csv",
            "calc_train": "calc_case_description_train_set.csv",
            "mass_test": "mass_case_description_test_set.csv",
            "calc_test": "calc_case_description_test_set.csv",
            "dicom_info": "dicom_info.csv"
        }

    def get_file_path(self, key):
        filename = self.file_map.get(key)
        if not filename:
            raise ValueError(f"No file mapping for key: {key}")
        return os.path.join(self.input_dir, filename)

    def _build_universal_path_finder(self):
        """
        Creates a single, universal map from SeriesInstanceUID to the final JPEG image path.
        This map is created once from dicom_info.csv.
        """
        print("Building universal path finder map from dicom_info.csv...")
        dicom_info_path = self.get_file_path("dicom_info")
        try:
            dicom_df = pd.read_csv(dicom_info_path)
            path_finder_map = dicom_df[['SeriesInstanceUID', 'image_path']].copy()
            print("Universal path finder map built successfully.")
            return path_finder_map
        except FileNotFoundError:
            print(f"Error: Crucial file 'dicom_info.csv' not found at {dicom_info_path}.")
            return None

    def process_data(self, file_path, abnormality_name, path_finder_map, is_test_set=False):
        """
        Loads, filters, and processes a dataset, using the robust multi-merge logic to
        link all image paths correctly before splitting the data.
        """
        print(f"\nProcessing {abnormality_name} {'Test' if is_test_set else 'Train/Val'} dataset...")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}.")
            return None

        # --- Standard Filtering from original script ---
        df.columns = df.columns.str.strip()
        df['assessment'] = pd.to_numeric(df['assessment'].astype(str).str.strip(), errors='coerce')
        original_rows = len(df)
        print(f"Original number of rows: {original_rows}")

        assessment_filter = [2, 4, 5]
        df_filtered = df[df['assessment'].isin(assessment_filter)].copy()
        print(f"Rows after filtering by assessment [2, 4, 5]: {len(df_filtered)}")

        pathology_mapping = {
            'MALIGNANT': 'MALIGNANT',
            'BENIGN': 'BENIGN',
            'BENIGN_WITHOUT_CALLBACK': 'BENIGN'
        }
        df_filtered['simple_pathology'] = df_filtered['pathology'].map(pathology_mapping)
        df_filtered['abnormality_type'] = abnormality_name

        # --- The New, Proven Linking Logic ---
        def extract_uid(path):
            try:
                return path.split('/')[-2]
            except (TypeError, IndexError):
                return None
        
        df_filtered['full_image_SeriesUID'] = df_filtered['image file path'].apply(extract_uid)
        df_filtered['cropped_image_SeriesUID'] = df_filtered['cropped image file path'].apply(extract_uid)
        df_filtered['roi_mask_SeriesUID'] = df_filtered['ROI mask file path'].apply(extract_uid)

        # Perform three separate merges for each image type
        merged_df = pd.merge(df_filtered, path_finder_map.rename(columns={'image_path': 'full_image_path'}), left_on='full_image_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])
        merged_df = pd.merge(merged_df, path_finder_map.rename(columns={'image_path': 'cropped_image_path'}), left_on='cropped_image_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])
        merged_df = pd.merge(merged_df, path_finder_map.rename(columns={'image_path': 'roi_path'}), left_on='roi_mask_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])

        # --- Train/Val/Test Split from original script ---
        if is_test_set:
            merged_df['split'] = 'test'
        else:
            if not merged_df.empty:
                y = merged_df['simple_pathology']
                # Check for cases where stratification is possible
                if y.nunique() > 1:
                    _, val_indices = train_test_split(merged_df.index, test_size=self.validation_size, stratify=y, random_state=self.random_state)
                    merged_df['split'] = 'train'
                    merged_df.loc[val_indices, 'split'] = 'validation'
                else:
                    # If only one class is present, just assign all to train
                    merged_df['split'] = 'train'
        
        print(f"Processing complete. Found {len(merged_df)} records.")
        return merged_df

    def run(self):
        # Create the path finder map once at the beginning
        path_finder_map = self._build_universal_path_finder()
        if path_finder_map is None:
            print("\n--- Script Aborted: Could not build the path finder map. ---")
            return

        # Process all four datasets
        mass_train_val_df = self.process_data(self.get_file_path("mass_train"), "Mass", path_finder_map, is_test_set=False)
        calc_train_val_df = self.process_data(self.get_file_path("calc_train"), "Calcification", path_finder_map, is_test_set=False)
        mass_test_df = self.process_data(self.get_file_path("mass_test"), "Mass", path_finder_map, is_test_set=True)
        calc_test_df = self.process_data(self.get_file_path("calc_test"), "Calcification", path_finder_map, is_test_set=True)

        # --- Combination and Saving Logic from original script ---
        if all(df is not None for df in [mass_train_val_df, calc_train_val_df, mass_test_df, calc_test_df]):
            # Combine and save mass
            mass_final_df = pd.concat([mass_train_val_df, mass_test_df], ignore_index=True)
            mass_output_path = os.path.join(self.output_dir, "mass_specialist_set.csv")
            mass_final_df.to_csv(mass_output_path, index=False)
            print(f"\nSaved final Mass set to: {mass_output_path}")
            print("Final Mass split distribution:\n", mass_final_df['split'].value_counts())

            # Combine and save calcification
            calc_final_df = pd.concat([calc_train_val_df, calc_test_df], ignore_index=True)
            calc_output_path = os.path.join(self.output_dir, "calc_specialist_set.csv")
            calc_final_df.to_csv(calc_output_path, index=False)
            print(f"\nSaved final Calcification set to: {calc_output_path}")
            print("Final Calcification split distribution:\n", calc_final_df['split'].value_counts())

            # Combine all and save
            final_combined_df = pd.concat([mass_final_df, calc_final_df], ignore_index=True)
            combined_output_path = os.path.join(self.output_dir, "combined_specialist_set.csv")
            final_combined_df.to_csv(combined_output_path, index=False)
            print(f"\nSaved final Combined set to: {combined_output_path}")
            print("Overall final split distribution:\n", final_combined_df['split'].value_counts())

        print("\n--- Script Finished ---")


if __name__ == "__main__":
    # Define your input and output directories
    input_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"
    output_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"
    
    # Create an instance of the class and run the process
    filterer = DataFilter(input_directory, output_directory)
    filterer.run()