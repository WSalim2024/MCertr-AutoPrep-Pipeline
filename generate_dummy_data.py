import pandas as pd
import numpy as np


def create_messy_dataset():
    # Set seed for reproducibility as requested in the text
    np.random.seed(0)

    # Generate the specific dummy data structure from the activity
    # 100 normal rows + 2 outliers/missing = 102 rows total
    feature1 = np.random.normal(100, 10, 100).tolist() + [np.nan, 200]
    feature2 = np.random.randint(0, 100, 102).tolist()
    # Pattern A,B,C,D repeated 25 times = 100, plus NaN and 'A' = 102
    category = ['A', 'B', 'C', 'D'] * 25 + [np.nan, 'A']
    target = np.random.choice([0, 1], 102).tolist()

    dummy_data = {
        'Feature1': feature1,
        'Feature2': feature2,
        'Category': category,
        'Target': target
    }

    df = pd.DataFrame(dummy_data)

    # Save to CSV
    output_file = 'raw_data.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Dummy data generated matches Activity specs: {output_file}")
    print(df.tail())  # Show the outliers at the end


if __name__ == "__main__":
    create_messy_dataset()