""" 
Preprocessing library to generate dataframes 
useable for model training

Author: Rohan
Stand: 22/07/2025
"""






#import libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from PIL import Image
import os


class Extractor:
    
    def __init__(self,directory: str):
        """
        assign the global directory for local csv files 
        """
        self.directory = directory
        self.filenames = self.get_file_names()
        self.dataframes = {}
        self.modified_dataframe = {}

    def get_file_names(self):
        return [x for x in os.listdir(self.directory) if x.endswith(".csv")]
    
    def load_df(self,name=None):
        if name:
            path = os.path.join(self.directory, name)
            self.dataframes[name] = pd.read_csv(path)
        else:
            for names in self.filenames:
                if names not in self.dataframes:
                    path = os.path.join(self.directory, names)
                    self.dataframes[names] = pd.read_csv(path)
    
    def rename(self, old_name, new_name):
        if old_name not in self.dataframes:
            raise ValueError(f"{old_name} not loaded or doesn't exist.")
        if new_name in self.dataframes:
            raise ValueError(f"{new_name} already exists in dataframes.")
        
        self.dataframes[new_name] = self.dataframes.pop(old_name)

    def summary(self):
        summary = {}
        for name, df in self.dataframes.items():
            summary[name] = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'missing_values': df.isnull().sum().to_dict()
            }
        return summary
    
    def column_names(self, filename):
        if filename not in self.dataframes:
            raise ValueError(f"{filename} not loaded. Call `load_data('{filename}')` first.")
        return self.dataframes[filename].columns.tolist()
    
    def keep_col(self,filename,col_names=None):
        if not col_names:
            col_names = ["patient_id","assessment","pathology","image file path",
                         "ROI mask file path"]

        df =  self.dataframes[filename]
        missing_cols = [col for col in col_names if col not in df.columns]
        #print(missing_cols)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in {filename}.")
        df_mod = df.drop(df.columns.difference(col_names),axis=1)
        return df_mod
    
    def ROI_image_finder(self,folder):
        try:
            for name in os.listdir(folder):
                if name.split("-")[0]=="2":
                    return os.path.join(folder,name)
            return None
        except FileNotFoundError:
            return None

    def image_finder(self,folder):
        try:
            for name in os.listdir(folder):
                if name.split("-")[0]=="1":
                    return os.path.join(folder,name)
            return None
        except FileNotFoundError:
            return None

    def path_modifier(self,df,image_path:str):
        df["image file path"]=df["image file path"].apply(lambda x: image_path+"jpeg/"+x.split("/")[-2])
        df["ROI mask file path"]=df["ROI mask file path"].apply(lambda y: image_path+"jpeg/"+y.split("/")[-2])
        df["ROI image"] = df["ROI mask file path"].apply(self.ROI_image_finder)
        df["full image"] = df["image file path"].apply(self.image_finder)
        df = df.dropna()
        return df


    def preprocess_steps(self, filename=None,image_path=None, output_dir="New_Files"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)        

        if not filename:
            filename = list(self.dataframes.keys())
        for name in filename:
            modified = self.keep_col(name)
            modified = self.path_modifier(modified,image_path)

            self.modified_dataframe[name]=modified.copy()

            output_path = os.path.join(output_dir, name)
            modified.to_csv(output_path, index=False)