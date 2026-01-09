import pandas as pd
import numpy as np


def create_messy_dataset():
    # Create a dictionary with intentional 'dirtiness'
    data = {
        'EmployeeID': [101, 102, 103, 104, 105, 106, 107, 108],
        'Age': [25, 30, np.nan, 35, 100, 28, 45, np.nan],  # NaN and Outlier (100)
        'Salary': [50000, 60000, 55000, np.nan, 200000, 58000, 62000, 59000],  # NaN and Outlier (200k)
        'City': ['New York', 'Paris', 'New York', 'Paris', 'Berlin', 'Berlin', 'Tokyo', 'Tokyo'],
        'Department': ['IT', 'HR', 'IT', 'Finance', 'IT', 'HR', 'Finance', 'IT']
    }

    df = pd.DataFrame(data)

    # Save to CSV
    output_file = 'raw_data.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Dummy data generated: {output_file}")
    print(df)


if __name__ == "__main__":
    create_messy_dataset()