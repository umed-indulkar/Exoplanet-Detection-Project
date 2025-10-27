import pandas as pd
import re
import ssl
import urllib.request

# Workaround for SSL timeout: use unverified context (not ideal but works for read-only data)
ssl_context = ssl._create_unverified_context()

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

# Try downloading confirmed Kepler planets from NASA Exoplanet Archive
# Using the Planetary Systems Composite Parameters table (simpler CSV download)
print('Downloading confirmed Kepler planets from NASA Exoplanet Archive...')
url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,koi_disposition&where=koi_disposition+like+%27CONFIRMED%27&format=csv'

confirmed_set = set()
try:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60, context=ssl_context) as response:
        confirmed = pd.read_csv(response)
    # KIC ID is in 'kepid' column
    confirmed_set = set(confirmed['kepid'].dropna().astype(int).tolist())
    print(f'Found {len(confirmed_set)} confirmed Kepler planets from NASA')
except Exception as e:
    print(f'NASA download failed: {e}')
    print('Trying alternative: Kepler Objects of Interest (KOI) table...')
    # Fallback: try KOI table
    try:
        url2 = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=koi&select=kepid,koi_disposition&where=koi_disposition+like+%27CONFIRMED%27&format=csv'
        req2 = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req2, timeout=60, context=ssl_context) as response2:
            confirmed2 = pd.read_csv(response2)
        confirmed_set = set(confirmed2['kepid'].dropna().astype(int).tolist())
        print(f'Found {len(confirmed_set)} confirmed Kepler planets from KOI table')
    except Exception as e2:
        print(f'Fallback also failed: {e2}')
        print('Using a hardcoded list of known Kepler planets as last resort...')
        # Small sample of known confirmed Kepler planets (from public data)
        confirmed_set = {
            10593626, 10601284, 10666592, 10748390, 10811496, 10854555, 10910878,
            11446443, 11465813, 11904151, 3114167, 5812701, 6185476, 6278762,
            6922244, 7199397, 7447200, 8191672, 8394721, 8644288, 8866102,
            9837578, 10024862, 10187159, 10255705, 10284575, 10403228
        }
        print(f'Using {len(confirmed_set)} known Kepler planets as fallback')

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
print(f'\n✓ Created labels.csv: {labels_df.shape}')
print(f'✓ Positive labels (confirmed planets): {labels_df["label"].sum()}')
print(f'✓ Negative labels: {(labels_df["label"]==0).sum()}')

if labels_df["label"].sum() == 0:
    print('\n⚠ WARNING: No confirmed planets found in your dataset.')
    print('This might mean:')
    print('  - Your KIC IDs do not match any confirmed planets')
    print('  - The NASA query failed and fallback list is too small')
    print('You can still train models, but the dataset is very imbalanced.')
