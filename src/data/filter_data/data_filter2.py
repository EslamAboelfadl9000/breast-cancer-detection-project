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

    def _build_typed_path_finders(self):
        print("Building type-specific path finder maps from dicom_info.csv...")
        dicom_info_path = self.get_file_path("dicom_info")
        try:
            dicom_df = pd.read_csv(dicom_info_path)
            full_map = dicom_df[dicom_df['SeriesDescription'] == 'full mammogram images'][['SeriesInstanceUID', 'image_path']].copy()
            cropped_map = dicom_df[dicom_df['SeriesDescription'] == 'cropped images'][['SeriesInstanceUID', 'image_path']].copy()
            roi_map = dicom_df[dicom_df['SeriesDescription'] == 'ROI mask images'][['SeriesInstanceUID', 'image_path']].copy()
            print("Type-specific path finder maps built successfully.")
            return {"full": full_map, "cropped": cropped_map, "roi": roi_map}
        except FileNotFoundError:
            print(f"Error: Crucial file 'dicom_info.csv' not found at {dicom_info_path}.")
            return None

    def process_data(self, file_path, abnormality_name, path_finder_maps, is_test_set=False):
        print(f"\nProcessing {abnormality_name} {'Test' if is_test_set else 'Train/Val'} dataset...")
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError: return None

        df.columns = df.columns.str.strip()
        df_filtered = df[df['assessment'].isin([2, 4, 5])].copy()
        pathology_mapping = {'MALIGNANT': 'MALIGNANT', 'BENIGN': 'BENIGN', 'BENIGN_WITHOUT_CALLBACK': 'BENIGN'}
        df_filtered['simple_pathology'] = df_filtered['pathology'].map(pathology_mapping)
        df_filtered['abnormality_type'] = abnormality_name

        def extract_uid(path):
            try: return path.split('/')[-2]
            except: return None
        
        df_filtered['full_image_SeriesUID'] = df_filtered['image file path'].apply(extract_uid)
        df_filtered['cropped_image_SeriesUID'] = df_filtered['cropped image file path'].apply(extract_uid)
        df_filtered['roi_mask_SeriesUID'] = df_filtered['ROI mask file path'].apply(extract_uid)

        merged_df = pd.merge(df_filtered, path_finder_maps['full'].rename(columns={'image_path': 'full_image_path'}), left_on='full_image_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])
        merged_df = pd.merge(merged_df, path_finder_maps['cropped'].rename(columns={'image_path': 'cropped_image_path'}), left_on='cropped_image_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])
        merged_df = pd.merge(merged_df, path_finder_maps['roi'].rename(columns={'image_path': 'roi_path'}), left_on='roi_mask_SeriesUID', right_on='SeriesInstanceUID', how='left').drop(columns=['SeriesInstanceUID'])

        final_cols = ['patient_id', 'abnormality_type', 'simple_pathology', 'full_image_path', 'cropped_image_path', 'roi_path']
        merged_df.dropna(subset=['full_image_path', 'roi_path'], inplace=True)
        final_df = merged_df[final_cols].copy().drop_duplicates()

        if is_test_set:
            final_df['split'] = 'test'
        else:
            if not final_df.empty:
                y = final_df['simple_pathology']
                if y.nunique() > 1:
                    _, val_indices = train_test_split(final_df.index, test_size=self.validation_size, stratify=y, random_state=self.random_state)
                    final_df.loc[:, 'split'] = 'train'
                    final_df.loc[val_indices, 'split'] = 'validation'
                else:
                    final_df.loc[:, 'split'] = 'train'
        
        print(f"Processing complete. Found {len(final_df)} records with mandatory full image and ROI mask.")
        return final_df

    def run(self):
        path_finder_maps = self._build_typed_path_finders()
        if path_finder_maps is None: return

        mass_train_val_df = self.process_data(self.get_file_path("mass_train"), "Mass", path_finder_maps, is_test_set=False)
        calc_train_val_df = self.process_data(self.get_file_path("calc_train"), "Calcification", path_finder_maps, is_test_set=False)
        mass_test_df = self.process_data(self.get_file_path("mass_test"), "Mass", path_finder_maps, is_test_set=True)
        calc_test_df = self.process_data(self.get_file_path("calc_test"), "Calcification", path_finder_maps, is_test_set=True)

        mass_final_df, calc_final_df = None, None

        if all(df is not None and not df.empty for df in [mass_train_val_df, mass_test_df]):
            mass_final_df = pd.concat([mass_train_val_df, mass_test_df], ignore_index=True)
            mass_output_path = os.path.join(self.output_dir, "mass_final_dataset.csv")
            mass_final_df.to_csv(mass_output_path, index=False)
            print(f"\nSaved final Mass set to: {mass_output_path}")

        if all(df is not None and not df.empty for df in [calc_train_val_df, calc_test_df]):
            calc_final_df = pd.concat([calc_train_val_df, calc_test_df], ignore_index=True)
            calc_output_path = os.path.join(self.output_dir, "calc_final_dataset.csv")
            calc_final_df.to_csv(calc_output_path, index=False)
            print(f"\nSaved final Calcification set to: {calc_output_path}")
        
        # --- ADDED THIS SECTION BACK ---
        if mass_final_df is not None and calc_final_df is not None:
            final_combined_df = pd.concat([mass_final_df, calc_final_df], ignore_index=True)
            combined_output_path = os.path.join(self.output_dir, "combined_final_dataset.csv")
            final_combined_df.to_csv(combined_output_path, index=False)
            print(f"\nSaved final Combined set to: {combined_output_path}")
            print("Overall final split distribution:\n", final_combined_df['split'].value_counts())
        
        print("\n--- Script Finished ---")


if __name__ == "__main__":
    input_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"
    output_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"
    
    filterer = DataFilter(input_directory, output_directory)
    filterer.run()