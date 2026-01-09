import pandas as pd
import numpy as np
import random


def create_messy_dataset():
    # Set seed for reproducibility
    np.random.seed(42)

    # 1. Create Base Data (100 rows)
    n = 100
    ids = np.arange(1001, 1001 + n)

    # 2. Logic Errors (Negative Ages) - Step 4 in text
    ages = np.random.randint(18, 65, n)
    ages[0:5] = ages[0:5] * -1  # Make first 5 ages negative

    # 3. Inconsistent Text (Typos/Case) - Step 4 in text
    cities = np.random.choice(['New York', 'Paris', 'Berlin', 'Tokyo'], n)
    # Introduce inconsistency
    cities[10] = "paris "  # Lowercase + space
    cities[11] = "NEW YORK"  # All caps
    cities[12] = "Tokyoo"  # Typo

    # 4. Consistency Check (Math Errors) - Step 5 in text
    part1 = np.random.randint(10, 50, n)
    part2 = np.random.randint(10, 50, n)
    total = part1 + part2
    # Break the logic for a few rows
    total[20:25] = total[20:25] + 10

    # 5. Outliers (Salary) - Step 3 in text
    salary = np.random.normal(50000, 10000, n)
    salary[90:95] = [150000, 2000000, 500, 12, np.nan]  # Extreme outliers + NaN

    # Create DataFrame
    data = {
        'EmployeeID': ids,
        'Age': ages,
        'City': cities,
        'Part1': part1,
        'Part2': part2,
        'Total': total,  # Expected = Part1 + Part2
        'Salary': salary
    }
    df = pd.DataFrame(data)

    # 6. Create Duplicates - Step 5 in text
    # Append the first 5 rows to the end again
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

    # Save
    output_file = 'raw_data.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Generated 'raw_data.csv' with intentional errors:")
    print("- Negative Ages")
    print("- Inconsistent Cities (paris vs Paris)")
    print("- Bad Totals (Part1 + Part2 != Total)")
    print("- Duplicate Rows")
    print(f"Total Rows: {len(df)}")


if __name__ == "__main__":
    create_messy_dataset()