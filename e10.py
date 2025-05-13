import pandas as pd
import pyarrow.parquet as pq
import h3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## pip install pandas scikit-learn h3 pyarrow
## python pandas read_parquet process killed


# PARAMETERS
H3_RESOLUTION = 8

# 1. Load data
print('1. Load data')

print('load mobility_data.parquet')
mobility = pd.read_parquet('./data/mobility_data.parquet', engine='fastparquet')

print( f"number of rows: {len(mobility)}" ) # 340,411,133


# 2. Assign H3 hex to each mobility record
print('2. Assign H3 hex to each mobility record')
mobility['hex_id'] = mobility.apply(
    lambda row: h3.latlng_to_cell(row['lat'], row['lon'], H3_RESOLUTION), axis=1
)

mobility = mobility.drop(['lat', 'lon'], axis=1)

mobility['timestamp'] = pd.to_datetime(mobility['timestamp'], unit="s").dt.date


mobility = mobility.groupby(['hex_id','timestamp']).agg(
    device_count=('device_id', 'nunique'),
    record_count=('device_id', 'count'),
).reset_index()
mobility.to_csv('./data/mobility_groupby_hextime.csv')


# 3. Feature engineering: aggregate mobility data per hex
print('3. Feature engineering: aggregate mobility data per hex')
features = mobility.groupby('hex_id').agg(
    device_count=('device_count', 'sum'),
    record_count=('record_count', 'sum'),
    unique_days=('timestamp', 'nunique'),
).reset_index()
# print(features.head(10))
features.to_csv('./data/features-groupby-hex.csv', index=False)

# free ram test
del mobility


# 4. Merge with training data
print('4. Merge with training data')
print('load train.csv')
train = pd.read_csv('./data/train.csv')

data = pd.merge(features, train, on='hex_id', how='left')
data.to_csv('./data/merge-features-train.csv', index=False)


# 5. Prepare data for modeling
print('5. Prepare data for modeling')
X = data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = data['cost_of_living']


# 6. Train/test split
print('6. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)


# 7. Train model
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


# # 8. Evaluate (optional)
# print('8. Evaluate')
# y_pred = model.predict(X_val)
# print('Validation RMSE:', mean_squared_error(y_val, y_pred, squared=False))

# ## Predict for all hexes in features
# # features['cost_of_living'] = model.predict(features[['device_count', 'record_count', 'unique_days']])
# # features['cost_of_living'] = features['cost_of_living'].clip(0, 1)  # Ensure within [0, 1]


# 9. Predict only for hexes in test.csv
print('load test.csv')
test = pd.read_csv('./data/test.csv')  # columns: hex_id

print('hexes_to_predict merge')
hexes_to_predict = pd.merge(
    test[['hex_id']],
    features,
    on='hex_id',
    how='left'
)
hexes_to_predict.to_csv('./data/merge-test-features.csv', index=False)


print('predict')
hexes_to_predict['cost_of_living'] = model.predict(
    hexes_to_predict[[
        'device_count',
        'record_count',
        'unique_days'
    ]]
)
hexes_to_predict['cost_of_living'] = hexes_to_predict['cost_of_living'].clip(0, 1)


# 10. Output CSV
print('10. Output CSV')
hexes_to_predict[['hex_id', 'cost_of_living']].to_csv('./data/cost_of_living_predictions.csv', index=False)
print('Predictions saved to cost_of_living_predictions.csv')