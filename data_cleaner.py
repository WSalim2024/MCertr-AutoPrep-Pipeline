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
        """Step 1: Load the dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("❌ File not found. Please check the path.")
            return None

    def handle_missing_values(self, strategy='drop'):
        """Step 2.1: Handle missing values (Mean for numeric, Mode for categorical)."""
        if self.df is None: return self.df

        if strategy == 'drop':
            self.df.dropna(inplace=True)
            print("Values dropped.")
        elif strategy == 'fill_mean':
            # 1. Fill Numeric with Mean
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

            # 2. Fill Categorical with Mode (Most Frequent Value)
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                # We use [0] to get the first mode if there are multiple
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)

            print("Missing values filled (Mean for numeric, Mode for categorical).")

        return self.df

    def handle_outliers(self, threshold=3):
        """Step 2.2: Remove outliers using Z-score."""
        if self.df is None: return self.df

        # Calculate Z-scores for numeric columns only
        numeric_df = self.df.select_dtypes(include=[np.number])
        z_scores = np.abs(stats.zscore(numeric_df))

        # Keep rows where ALL z-scores are less than the threshold (3)
        self.df = self.df[(z_scores < threshold).all(axis=1)]

        print(f"Outliers removed. New Shape: {self.df.shape}")
        return self.df

    def scale_data(self, method='standard'):
        """Step 2.3: Scale data (Default to StandardScaler/Z-score per instructions)."""
        if self.df is None: return self.df

        # Instruction specifies StandardScaler (Z-score normalization)
        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        # Apply scaling
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        print(f"Data scaled using {method} scaler.")
        return self.df

    def encode_categorical(self):
        """Step 2.4: One-hot encode categorical variables."""
        if self.df is None: return self.df

        # Instruction requires all categories to be present (e.g., Category_A)
        # Therefore, we set drop_first=False (default behavior)
        self.df = pd.get_dummies(self.df, columns=['Category'], drop_first=False)

        print("Categorical variables encoded.")
        return self.df

    def save_data(self, output_name='preprocessed_dummy_data.csv'):
        """Step 2.5: Save to file."""
        if self.df is None: return

        self.df.to_csv(output_name, index=False)
        print(f"✅ Preprocessing complete. Data saved as {output_name}")


# --- Step 3: Verify the data and perform a quality check ---
if __name__ == "__main__":
    print("\n--- Starting ML Preprocessing Pipeline (Walkthrough Mode) ---")

    # 1. Initialize
    pipeline = DataPreprocessor('raw_data.csv')

    # 2. Run Steps (Matching the Walkthrough exactly)
    df = pipeline.load_data()
    pipeline.handle_missing_values(strategy='fill_mean')
    pipeline.handle_outliers()
    pipeline.scale_data(method='standard')  # Updated to Standard
    pipeline.encode_categorical()
    pipeline.save_data('preprocessed_dummy_data.csv')

    # 3. VERIFICATION (Required by Walkthrough Text)
    print("\n--- 🔍 Verification Report ---")

    print("\n1. Check for missing values (Should be 0):")
    print(pipeline.df.isnull().sum())

    print("\n2. Verify outlier removal (Check Count/Max):")
    print(pipeline.df.describe())

    print("\n3. Inspect scaled data (Values should be Standardized/Z-scores):")
    print(pipeline.df.head())

    print("\n4. Check categorical encoding (Should include Category_A, B, C, D):")
    print(pipeline.df.columns)