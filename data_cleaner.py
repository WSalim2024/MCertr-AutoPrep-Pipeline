import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats


class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Step 1: Load and inspect the dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print("❌ File not found.")
            return None

    # --- NEW: Step 5 Validation ---
    def remove_duplicates(self):
        """Step 5: Identify and remove duplicate rows."""
        if self.df is None: return

        initial_count = len(self.df)
        self.df.drop_duplicates(inplace=True)
        dropped_count = initial_count - len(self.df)

        print(f"Removed {dropped_count} duplicate rows.")
        return self.df

    # --- NEW: Step 4 Data Entry Errors ---
    def validate_integrity(self):
        """Step 4: Check for impossible values (e.g., negative ages)."""
        if self.df is None: return

        # Replace negative ages with NaN (Logic check)
        if 'Age' in self.df.columns:
            neg_count = (self.df['Age'] < 0).sum()
            if neg_count > 0:
                self.df['Age'] = np.where(self.df['Age'] < 0, np.nan, self.df['Age'])
                print(f"Fixed {neg_count} negative Age values (set to NaN).")
        return self.df

    def standardize_text(self):
        """Step 4: Standardize categories (strip whitespace, title case)."""
        if self.df is None: return

        # Select object (text) columns
        cat_cols = self.df.select_dtypes(include=['object']).columns

        for col in cat_cols:
            # Strip whitespace and convert to Title Case (e.g., "paris " -> "Paris")
            self.df[col] = self.df[col].astype(str).str.strip().str.title()

        print("Text columns standardized (Trimmed & Title Cased).")
        return self.df

    def fix_consistency(self):
        """Step 5: Validate consistency (Recalculate Totals)."""
        if self.df is None: return

        # Logic: Total should equal Part1 + Part2
        if {'Total', 'Part1', 'Part2'}.issubset(self.df.columns):
            expected = self.df['Part1'] + self.df['Part2']
            mismatches = (self.df['Total'] != expected).sum()

            if mismatches > 0:
                self.df['Total'] = expected
                print(f"Corrected {mismatches} calculation errors in 'Total' column.")
        return self.df

    # --- Existing Steps 2 & 3 ---
    def handle_missing_values(self, strategy='fill_mean'):
        """Step 2: Handle missing values."""
        if self.df is None: return

        if strategy == 'fill_mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

            # Fill Categorical with Mode
            cat_cols = self.df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if not self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            print("Missing values filled.")
        return self.df

    def handle_outliers(self, threshold=3):
        """Step 3: Remove outliers using Z-score."""
        if self.df is None: return

        numeric_df = self.df.select_dtypes(include=[np.number])
        # Only compute Z-scores on columns with NO NaNs (important!)
        numeric_df = numeric_df.dropna()

        if not numeric_df.empty:
            z_scores = np.abs(stats.zscore(numeric_df))
            self.df = self.df[(z_scores < threshold).all(axis=1)]
            print(f"Outliers removed. New Shape: {self.df.shape}")
        return self.df

    def scale_data(self, method='standard'):
        """Step 3: Normalize/Scale data."""
        if self.df is None: return

        scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        if not numeric_cols.empty:
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
            print(f"Data scaled using {method}.")
        return self.df

    def save_data(self, output_name='cleaned_data.csv'):
        if self.df is None: return
        self.df.to_csv(output_name, index=False)
        print(f"✅ Saved to {output_name}")


if __name__ == "__main__":
    # Test Run
    pipeline = DataPreprocessor('raw_data.csv')
    pipeline.load_data()
    pipeline.remove_duplicates()  # New Step 5
    pipeline.validate_integrity()  # New Step 4
    pipeline.standardize_text()  # New Step 4
    pipeline.fix_consistency()  # New Step 5
    pipeline.handle_missing_values()
    pipeline.handle_outliers()
    pipeline.scale_data()
    pipeline.save_data()