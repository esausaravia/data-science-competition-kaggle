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

# Load data
print('1. Load data')

print('load mobility_data.parquet')

features = pd.read_csv("./data/02-mobility-features-grpby-hex.csv")


# Merge with training data
print('2. Merge with training data')
print('load train.csv')
train = pd.read_csv('./data/train.csv')

train_data = pd.merge(train, features, on='hex_id', how='left').fillna(0)


# Prepare data for modeling
print('3. Prepare data for modeling')
X = train_data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = train_data['cost_of_living']

# Train/test split
print('4. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)


# Train model
print('5. Train model')
# {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}
model = RandomForestRegressor(
    n_estimators=500,
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
print('3. Prepare data for modeling')
X = second_train_data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = second_train_data['cost_of_living']

# Train/test split
print('4. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)


# Train model
print('5. Train model')
# {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 300}
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)


# 6. Evaluate (optional)
print('6. Evaluate')
y_pred = model.predict(X_val)
print('Validation RMSE:', root_mean_squared_error(y_val, y_pred))

