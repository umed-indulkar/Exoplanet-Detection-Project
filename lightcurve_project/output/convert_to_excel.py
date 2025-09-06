import pandas as pd

# Load the CSV file
csv_file = "features.csv"       # change this to your CSV filename
excel_file = "features.xlsx"    # output Excel file name

# Read CSV into a DataFrame
df = pd.read_csv(csv_file)

# Save as Excel
df.to_excel(excel_file, index=False)

print(f"Converted '{csv_file}' to '{excel_file}' successfully!")
