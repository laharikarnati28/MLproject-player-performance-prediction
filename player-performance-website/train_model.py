
import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "data/player_performance_ridge_lasso_dataset.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

features = [
    "Matches",
    "Runs",
    "Batting_Average",
    "Strike_Rate",
    "Fifties",
    "Hundreds",
    "Wickets",
]
target = "Performance_Score"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0)),
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Lasso(alpha=0.1, max_iter=10000)),
    ]),
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics[name.lower()] = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "r2": round(float(r2_score(y_test, predictions)), 3),
    }

    print(f"\n{name} Regression")
    print(f"MAE: {metrics[name.lower()]['mae']}")
    print(f"R²: {metrics[name.lower()]['r2']}")

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, f"{name.lower()}_model.pkl"),
    )

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as file:
    json.dump(metrics, file, indent=4)

print("\nModels and evaluation metrics saved successfully!")
