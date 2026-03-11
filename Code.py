# ==========================================
# Turbofan Engine Remaining Useful Life (RUL)
# ==========================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ==========================================
# 2. Load Dataset
# ==========================================

# Define column names
columns = ['id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]

# Load training data
train = pd.read_csv('PM_train.txt', sep=r'\s+', header=None)

# Remove extra columns if dataset has them
train = train.iloc[:, :26]

# Assign column names
train.columns = columns

# Load test data
test = pd.read_csv('PM_test.txt', sep=r'\s+', header=None)
test = test.iloc[:, :26]
test.columns = columns

# Load truth data
truth = pd.read_csv('PM_truth.txt', header=None)

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)


# ==========================================
# 3. Calculate Remaining Useful Life (RUL)
# ==========================================

max_cycle = train.groupby('id')['cycle'].max().reset_index()
max_cycle.columns = ['id','max_cycle']

train = train.merge(max_cycle, on='id')

train['RUL'] = train['max_cycle'] - train['cycle']

train.drop('max_cycle', axis=1, inplace=True)

print("\nTraining Data with RUL:")
print(train.head())


# ==========================================
# 4. Feature Selection
# ==========================================

features = train.drop(['id','cycle','RUL'], axis=1)
target = train['RUL']


# ==========================================
# 5. Feature Scaling
# ==========================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(features)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled,
    target,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 6. Train Machine Learning Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)


# ==========================================
# 7. Predictions
# ==========================================

pred = model.predict(X_val)


# ==========================================
# 8. Model Evaluation
# ==========================================

rmse = np.sqrt(mean_squared_error(y_val, pred))
mae = mean_absolute_error(y_val, pred)
r2 = r2_score(y_val, pred)

print("\nModel Performance")
print("------------------")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)


# ==========================================
# 9. Feature Importance
# ==========================================

importance = model.feature_importances_

feature_names = features.columns

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Important Sensors:")
print(imp_df.head())


# ==========================================
# 10. Visualization
# ==========================================

plt.figure(figsize=(10,5))
plt.plot(y_val.values[:100], label="True RUL")
plt.plot(pred[:100], label="Predicted RUL")
plt.xlabel("Samples")
plt.ylabel("Remaining Useful Life")
plt.title("True vs Predicted RUL")
plt.legend()
plt.show()


# ==========================================
# 11. Predict on Test Data
# ==========================================

test_features = test.drop(['id','cycle'], axis=1)

test_scaled = scaler.transform(test_features)

test_predictions = model.predict(test_scaled)

print("\nTest predictions generated successfully.")
