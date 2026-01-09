import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = "dataset/winequality-red.csv"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.joblib")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

X = df.drop("quality", axis=1)
y = df["quality"]

y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y_bins, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

joblib.dump(model, MODEL_PATH)

results = {
    "experiment_id": "EXP-03",
    "model": "Linear Regression",
    "hyperparameters": "Default",
    "preprocessing": "None",
    "feature_selection": "All",
    "split": "75/25 (Stratified)",
    "mse": mse,
    "r2_score": r2
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=4)





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
