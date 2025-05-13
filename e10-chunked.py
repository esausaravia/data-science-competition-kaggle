import pandas as pd
import pyarrow.parquet as pq
import h3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

## pip install pandas scikit-learn h3 pyarrow
## python pandas read_parquet process killed


# PARAMETERS
H3_RESOLUTION = 8
BATCH_SIZE = 18000000

exported_files = [""]

# 1. Load data
print('1. Load data')

print('load mobility_data.parquet')

parquet_file = pq.ParquetFile('./data/mobility_data.parquet')

featuresBatches = []
for batch in parquet_file.iter_batches(BATCH_SIZE):
    mobility = batch.to_pandas()

    # Assign H3 hex to each row
    print('Assign H3 hex to each row')
    mobility['hex_id'] = mobility.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lon'], H3_RESOLUTION), axis=1
    )
    mobility = mobility.drop(['lat', 'lon'], axis=1)

    mobility['timestamp'] = pd.to_datetime(mobility['timestamp'], unit="s").dt.date

    # 3. Features Engineering: aggregate mobility data per hex and date
    print('Aggregate mobility data per hex and date')
    features = mobility.groupby(['hex_id','timestamp']).agg(
        device_count=('device_id', 'nunique'),
        record_count=('device_id', 'count'),
    ).reset_index()
    # # free ram test
    # del mobility

    featuresBatches.append( features )
    # # Test the full script with just a batch part
    # if len(featuresBatches)>1 :
    #     break

print(f"featuresBatches len: {len(featuresBatches)}" )

combined = pd.concat(featuresBatches, ignore_index=True)
# combined.to_parquet("./data/02-mobility-featured.parquet", index=False)
# # CSV for human evaluation
# combined.to_csv("./data/02-mobility-featured.csv", index=False)

# # Load previous combined parquet from mobility, already featured by hex_id and timestamp
# combined = pd.read_parquet("./data/02-mobility-featured.parquet")

# Features Engineering: aggregate mobility data per hex
print("3. Featured: groupby hex_id with device_count, records_count, unique_days");
features = combined.groupby('hex_id').agg(
    device_count=('device_count', 'sum'),
    record_count=('record_count', 'sum'),
    unique_days=('timestamp', 'count'),
).reset_index()
features.to_csv("./data/02-mobility-features-grpby-hex.csv", index=False)

# Merge with training data
print('4. Merge with training data')
print('load train.csv')
train = pd.read_csv('./data/train.csv')

train_data = pd.merge(train, features, on='hex_id', how='left').fillna(0)
train_data.to_csv('./data/03-merge-train-features.csv', index=False)

# Prepare data for modeling
print('5. Prepare data for modeling')
X = train_data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = train_data['cost_of_living']


# Train/test split
print('6. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)


# Train model
print('8. Train model')
# {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)


# Evaluate (optional)
print('6. Evaluate')
y_pred = model.predict(X_val)
print('Validation RMSE:', root_mean_squared_error(y_val, y_pred))


# Retraining
features['cost_of_living'] = model.predict(features[['device_count', 'record_count', 'unique_days']])
features['cost_of_living'] = features['cost_of_living'].clip(0, 1)

second_train_data = pd.merge(features, train, on='hex_id', how='left')
second_train_data['cost_of_living_x'] = second_train_data['cost_of_living_y'].where(
    second_train_data['cost_of_living_y'].notna() & (second_train_data['cost_of_living_y']!=""),
    second_train_data['cost_of_living_x']
)
second_train_data['cost_of_living'] = second_train_data['cost_of_living_x']
second_train_data = second_train_data.drop(columns=['cost_of_living_x','cost_of_living_y'])
# second_train_data.to_csv('./data/05-second_train_data.csv', index=False)

# Prepare data for modeling
print('5. Prepare data for modeling')
X = second_train_data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = second_train_data['cost_of_living']

# Train/test split
print('6. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)


# Train model
print('7. Train model')
# {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)


# 6. Evaluate (optional)
print('8. Evaluate')
y_pred = model.predict(X_val)
print('Validation RMSE:', root_mean_squared_error(y_val, y_pred))


# Predict only for hexes in test.csv
print('10. Predict only for hexes in test.csv')
print('load test.csv')
test_data = pd.read_csv('./data/test.csv')  # columns: hex_id

print('merge with test data')
hexes_to_predict = pd.merge(
    test_data[['hex_id']],
    features,
    on='hex_id',
    how='left'
)
hexes_to_predict.to_csv('./data/04-merge-test-features.csv', index=False)

print('predict')
hexes_to_predict['cost_of_living'] = model.predict(
    hexes_to_predict[[
        'device_count',
        'record_count',
        'unique_days'
    ]]
)
hexes_to_predict['cost_of_living'] = hexes_to_predict['cost_of_living'].clip(0, 1)


# Output CSV
print('11. Output CSV')
hexes_to_predict[['hex_id', 'cost_of_living']].to_csv('./data/cost_of_living_predictions.csv', index=False)
print('Predictions saved to cost_of_living_predictions.csv')