import os
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class ClinicalBiomarkerAgent:
    """
    Expert-grade Tabular Data Parser.
    Ingests clinical metadata, normalizes continuous variables, 
    and outputs dense PyTorch tensors for multimodal fusion.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.features_fitted = False
        self.continuous_cols = ['Age', 'Educ', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
        self.categorical_cols = ['M/F']
        
    def _generate_synthetic_clinical_data(self, output_path: str) -> None:
        print("[!] Clinical CSV not found. Generating a mathematically synthetic OASIS dataset for testing...")
        np.random.seed(42)
        data = {
            'Subject_ID': [f'OAS1_{str(i).zfill(4)}' for i in range(1, 101)],
            'M/F': np.random.choice(['M', 'F'], 100),
            'Age': np.random.normal(75, 8, 100).clip(60, 95),
            'Educ': np.random.normal(14, 3, 100).clip(6, 20),
            'SES': np.random.choice([1, 2, 3, 4, 5], 100, p=[0.1, 0.2, 0.4, 0.2, 0.1]).astype(float),
            'MMSE': np.random.normal(25, 4, 100).clip(0, 30),
            'eTIV': np.random.normal(1450, 150, 100),
            'nWBV': np.random.normal(0.73, 0.04, 100),
            'ASF': np.random.normal(1.2, 0.1, 100)
        }
        data['MMSE'][5] = np.nan
        data['SES'][12] = np.nan
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    def ingest_and_process(self, csv_path: str) -> Tuple[torch.Tensor, pd.DataFrame]:
        if not os.path.exists(csv_path):
            self._generate_synthetic_clinical_data(csv_path)
            
        df = pd.read_csv(csv_path)
        
        # 1. Feature Imputation
        for col in self.continuous_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        # 2. Categorical Encoding
        if 'M/F' in df.columns:
            df['Gender_Male'] = (df['M/F'] == 'M').astype(int)
            df = df.drop(columns=['M/F'])
            
        # FIXED: Save the human-readable dataframe for the final report BEFORE scaling
        human_readable_df = df.copy()
            
        # 3. Z-Score Normalization for the Neural Network Tensor
        continuous_data = df[self.continuous_cols].values
        if not self.features_fitted:
            normalized_continuous = self.scaler.fit_transform(continuous_data)
            self.features_fitted = True
        else:
            normalized_continuous = self.scaler.transform(continuous_data)
            
        clean_features = np.hstack((normalized_continuous, df[['Gender_Male']].values))
        feature_tensor = torch.tensor(clean_features, dtype=torch.float32)
        
        # Return the mathematical tensor AND the human-readable dataframe
        return feature_tensor, human_readable_df

if __name__ == "__main__":
    CSV_PATH = os.path.join("data", "oasis_raw", "oasis_clinical_data.csv")
    agent = ClinicalBiomarkerAgent()
    feature_tensor, clean_dataframe = agent.ingest_and_process(CSV_PATH)