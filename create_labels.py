import pandas as pd
import re
from urllib.request import urlopen
from urllib.parse import quote

# Load features
df = pd.read_csv('outputs/features_combined.csv')
sources = df['source'].tolist()

# Extract KIC IDs
kic_ids = []
for s in sources:
    match = re.search(r'KIC_(\d+)', s)
    if match:
        kic_ids.append(int(match.group(1)))
    else:
        kic_ids.append(None)

print(f'Extracted {sum(1 for k in kic_ids if k is not None)} KIC IDs from {len(sources)} files')

# Query NASA Exoplanet Archive for confirmed Kepler planets
print('Querying NASA Exoplanet Archive...')
query = quote('select kepid from ps where default_flag=1 and hostname like \'Kepler%\'')
url = f'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={query}&format=csv'

try:
    with urlopen(url, timeout=30) as response:
        confirmed = pd.read_csv(response)
    confirmed_set = set(confirmed['kepid'].astype(int).tolist())
    print(f'Found {len(confirmed_set)} confirmed Kepler planets')
except Exception as e:
    print(f'NASA query failed: {e}. Using fallback (all label=0)')
    confirmed_set = set()

# Create labels
labels = []
for src, kic in zip(sources, kic_ids):
    if kic is None:
        label = 0  # no KIC ID found
    else:
        label = 1 if kic in confirmed_set else 0
    labels.append({'source': src, 'label': label})

labels_df = pd.DataFrame(labels)
labels_df.to_csv('labels.csv', index=False)
print(f'\nCreated labels.csv: {labels_df.shape}')
print(f'Positive labels (confirmed planets): {labels_df["label"].sum()}')
print(f'Negative labels: {(labels_df["label"]==0).sum()}')
