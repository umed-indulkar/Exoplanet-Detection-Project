import pandas as pd

# Define the 502 column names
cols = ['kepid', 'Label'] + [f'flux_{i}' for i in range(1, 501)]

# Load the merged file (which currently has no header)
file_path = r"D:\ppp\data\dataset_500\raw_curve_500_head.csv"
df = pd.read_csv(file_path, header=None, names=cols)

# Save it back WITH the header
df.to_csv(r"D:\ppp\data\dataset_500\raw_curve_500_head.csv", index=False)
print("Header added to Training Data!")