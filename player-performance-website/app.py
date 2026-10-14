
from flask import Flask, render_template, request
import pandas as pd
import joblib
import json

app = Flask(__name__)

ridge_model = joblib.load("models/ridge_model.pkl")
lasso_model = joblib.load("models/lasso_model.pkl")

with open("models/metrics.json", "r") as file:
    metrics = json.load(file)

FEATURES = [
    "Matches",
    "Runs",
    "Batting_Average",
    "Strike_Rate",
    "Fifties",
    "Hundreds",
    "Wickets",
]


@app.route("/", methods=["GET", "POST"])
def home():
    predictions = None
    error = None

    if request.method == "POST":
        try:
            values = {
                feature: float(request.form[feature])
                for feature in FEATURES
            }

            if any(value < 0 for value in values.values()):
                raise ValueError("Values cannot be negative.")

            player_data = pd.DataFrame([values], columns=FEATURES)

            predictions = {
                "ridge": round(float(ridge_model.predict(player_data)[0]), 2),
                "lasso": round(float(lasso_model.predict(player_data)[0]), 2),
            }

        except (ValueError, KeyError):
            error = "Please enter valid, non-negative numbers in every field."

    return render_template(
        "index.html",
        predictions=predictions,
        error=error,
        metrics=metrics,
    )


if __name__ == "__main__":
    app.run(debug=True)
