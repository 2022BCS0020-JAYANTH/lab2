import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")
X = df.drop("quality", axis=1)
y = df["quality"]

prelim_model = RandomForestRegressor(n_estimators=100, random_state=42)
prelim_model.fit(X, y)

important_features = X.columns[prelim_model.feature_importances_.argsort()[-6:]]
X_imp = X[important_features]

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

joblib.dump(model, MODEL_PATH)

results = {
    "experiment_id": "EXP-06",
    "model": "Random Forest",
    "hyperparameters": "100 trees, depth=15",
    "preprocessing": "None",
    "feature_selection": "Importance-based",
    "split": "80/20",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)








# # task 2 code
# import os
# import json
# import joblib
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, r2_score


# DATA_PATH = "dataset/winequality-red.csv"
# OUTPUT_DIR = "outputs"
# MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
# RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# df = pd.read_csv(DATA_PATH, sep=";")

# X = df.drop("quality", axis=1)
# y = df["quality"]
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# pipeline = Pipeline(
#  steps=[
#  ("scaler", StandardScaler()),
#  ("feature_selection", SelectKBest(score_func=f_regression, k=8)),
#  ("model", LinearRegression())
#  ]
# )

# pipeline.fit(X_train, y_train)

# y_pred = pipeline.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"R2 Score: {r2}")

# joblib.dump(pipeline, MODEL_PATH)

# results = {
#  "mse": mse,
#  "r2_score": r2
# }

# with open(RESULTS_PATH, "w") as f:
#  json.dump(results, f, indent=4)
