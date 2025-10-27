import pandas as pd

file = 'sentinel_with_vectors.csv'
df = pd.read_csv(file)

occ = dict()
for idx, row in df.iterrows():
    if str(row['species_names']) == 'nan':
        continue

    for species in row['species_names'].split('|'):
        if species in occ:
            occ[species] += 1
        else:
            occ[species] = 1

pairs = sorted(occ.items(), key=lambda item: -item[1])
for name, count in pairs:
    print(f'{name}: \t{count}')