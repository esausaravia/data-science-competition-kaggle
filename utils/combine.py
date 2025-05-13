import pandas as pd

parquet_files = [
    './data/mobility_features-0.csv',
    './data/mobility_features-1.csv',
    './data/mobility_features-2.csv',
    './data/mobility_features-3.csv',
    './data/mobility_features-4.csv',
    './data/mobility_features-5.csv',
    './data/mobility_features-6.csv',
    './data/mobility_features-7.csv',
    './data/mobility_features-8.csv',
    './data/mobility_features-9.csv',
    './data/mobility_features-10.csv',
    './data/mobility_features-11.csv',
    './data/mobility_features-12.csv',
    './data/mobility_features-13.csv',
    './data/mobility_features-14.csv',
    './data/mobility_features-15.csv',
    './data/mobility_features-16.csv',
    './data/mobility_features-17.csv',
    './data/mobility_features-18.csv',
]

parquetArray = []
for batch_file in parquet_files:
    batch = pd.read_csv(batch_file)
    parquetArray.append(batch)

df_merged = pd.concat(parquetArray, ignore_index=True)

df_merged.to_csv('./data/mobility-featured.csv', index=False)
df_merged.to_parquet('./data/mobility-featured.parquet', index=False)