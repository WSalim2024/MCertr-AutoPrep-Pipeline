import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Step 2: Load the dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError:
            print("❌ File not found. Please check the path.")

    def handle_missing_values(self, strategy='drop'):
        """Step 3: Handle missing values (Drop or Fill)."""
        if self.df is None: return

        # Optional: Visualize missing data (Step 1 requirement)
        # msno.matrix(self.df)
        # plt.show()

        if strategy == 'drop':
            self.df.dropna(inplace=True)
            print("Values dropped.")
        elif strategy == 'fill_mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            print("Missing values filled with mean.")

    def handle_outliers(self, threshold=3):
        """Step 4: Remove outliers using Z-score."""
        if self.df is None: return

        # Select only numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(numeric_df))

        # Filter data keeping only rows with Z-score < threshold
        self.df = self.df[(z_scores < threshold).all(axis=1)]
        print(f"Outliers removed. New Shape: {self.df.shape}")

    def scale_data(self, method='minmax'):
        """Step 5: Normalize/Scale data."""
        if self.df is None: return

        scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()

        # Identify numeric columns (excluding potential target variables if needed)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        print(f"Data scaled using {method}.")

    def encode_categorical(self):
        """Step 6: One-Hot Encoding for categorical variables."""
        if self.df is None: return

        self.df = pd.get_dummies(self.df, drop_first=True)
        print("Categorical variables encoded.")

    def save_data(self, output_name='cleaned_data.csv'):
        """Step 7: Save to file."""
        if self.df is None: return

        self.df.to_csv(output_name, index=False)
        print(f"✅ Process complete. Data saved to {output_name}")


# --- Automation / Workflow Execution ---
if __name__ == "__main__":
        print("\n--- Starting AutoPrep Pipeline ---")

        # Initialize with the file we just generated
        pipeline = DataPreprocessor('raw_data.csv')

        # Execute the workflow step-by-step
        pipeline.load_data()

        # Handle Missing Values (using mean for numeric columns)
        pipeline.handle_missing_values(strategy='fill_mean')

        # Remove Outliers (Standard Deviation > 3)
        pipeline.handle_outliers()

        # Normalize Data (0 to 1 scale)
        pipeline.scale_data(method='minmax')

        # Encode Categories (Paris/Berlin/etc. -> 0/1)
        pipeline.encode_categorical()

        # Save the final result
        pipeline.save_data('cleaned_master_data.csv')