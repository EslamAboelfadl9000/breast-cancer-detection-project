import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataFilter:
    def __init__(self, input_dir, output_dir, image_root_dir=None, validation_size=0.20, random_state=42):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.image_root_dir = os.path.abspath(image_root_dir) if image_root_dir else None
        self.validation_size = validation_size
        self.random_state = random_state

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        self.file_map = {
            "mass_train": "mass_case_description_train_set.csv",
            "calc_train": "calc_case_description_train_set.csv",
            "mass_test": "mass_case_description_test_set.csv",
            "calc_test": "calc_case_description_test_set.csv"
        }

    def get_file_path(self, key):
        filename = self.file_map.get(key)
        if not filename:
            raise ValueError(f"No file mapping for key: {key}")
        return os.path.join(self.input_dir, filename)

    def find_image_file(self, folder, prefix):
        """Finds the first file in folder whose name starts with prefix (e.g., '1' for image, '2' for ROI mask)."""
        try:
            for name in os.listdir(folder):
                if name.split("-")[0] == prefix:
                    return os.path.join(folder, name)
            return None
        except Exception:
            return None

    def enrich_paths(self, df):
        """Adds full image, cropped image, and ROI mask paths to the dataframe."""
        if not self.image_root_dir:
            df['full_image_path'] = None
            df['full_cropped_image_path'] = None
            df['full_roi_mask_path'] = None
            return df

        def get_folder_path(rel_path):
            # Assumes rel_path is like ".../some_folder/filename" and you want the folder
            return os.path.join(self.image_root_dir, "jpeg", rel_path.split("/")[-2])

        df['image_folder'] = df['image file path'].apply(get_folder_path)
        df['cropped_folder'] = df['cropped image file path'].apply(get_folder_path)
        df['roi_folder'] = df['ROI mask file path'].apply(get_folder_path)
        df['full_image_path'] = df['image_folder'].apply(lambda folder: self.find_image_file(folder, "1"))
        df['full_cropped_image_path'] = df['cropped_folder'].apply(lambda folder: self.find_image_file(folder, "1"))
        df['full_roi_mask_path'] = df['roi_folder'].apply(lambda folder: self.find_image_file(folder, "2"))
        df = df.drop(columns=['image_folder', 'cropped_folder', 'roi_folder'])
        return df

    def process_data(self, file_path, abnormality_name, is_test_set=False):
        print(f"\nProcessing {abnormality_name} {'Test' if is_test_set else 'Train/Val'} dataset...")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}. Please check your input_directory path.")
            return None

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
        df_filtered['pathology'] = df_filtered['pathology'].astype(str).str.strip()
        df_filtered['simple_pathology'] = df_filtered['pathology'].map(pathology_mapping)
        df_filtered['abnormality_type'] = abnormality_name

        # Add image and ROI mask full paths
        df_filtered = self.enrich_paths(df_filtered)

        if is_test_set:
            df_filtered['split'] = 'test'
        else:
            X = df_filtered
            y = df_filtered['simple_pathology']
            _, val_indices = train_test_split(
                X.index, test_size=self.validation_size, stratify=y, random_state=self.random_state
            )
            df_filtered['split'] = 'train'
            df_filtered.loc[val_indices, 'split'] = 'validation'

        print("Processing complete.")
        return df_filtered

    def run(self):
        # Process all datasets
        mass_train_val_df = self.process_data(self.get_file_path("mass_train"), "Mass", is_test_set=False)
        calc_train_val_df = self.process_data(self.get_file_path("calc_train"), "Calcification", is_test_set=False)
        mass_test_df = self.process_data(self.get_file_path("mass_test"), "Mass", is_test_set=True)
        calc_test_df = self.process_data(self.get_file_path("calc_test"), "Calcification", is_test_set=True)

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
    # Example usage (edit these paths as needed)
    input_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM\csv"
    output_directory = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\processed\csv"
    image_root_dir = r"C:\Users\New User\Documents\GitHub\breast-cancer-detection-project\data\raw\CBIS-DDSM"
    filterer = DataFilter(input_directory, output_directory, image_root_dir=image_root_dir)
    filterer.run()