import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Load data
print("1. Load data")
combined = pd.read_parquet("./data/02-mobility-featured.parquet")

# Features Engineering: aggregate mobility data per hex
print("3. Featured: groupby hex_id with device_count, records_count, unique_days");
features = combined.groupby('hex_id').agg(
    device_count=('device_count', 'sum'),
    record_count=('record_count', 'sum'),
    unique_days=('timestamp', 'count'),
).reset_index()

# Merge with training data
print('4. Merge with training data')
print('load train.csv')
train = pd.read_csv('./data/train.csv')

train_data = pd.merge(train, features, on='hex_id', how='left').fillna(0)

# 6. Prepare data for modeling
print('5. Prepare data for modeling')
X = train_data[[
    'device_count',
    'record_count',
    'unique_days'
]]
y = train_data['cost_of_living']


# 7. Train/test split
print('6. Train/test split')
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

# Define model
rf = RandomForestRegressor(random_state=42)

# Define hyperparameter grid
param_grid = {
    "n_estimators": [100, 300, 500,1000],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Grid search for best parameters
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
