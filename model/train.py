"""Train a tiny LinearRegression model and save to model.joblib.

Run from the project root:
    uv run python model/train.py

Outputs:
    model/model.joblib

NOTE: This training script intentionally does NOT scale features. See ISSUES.md #6.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


RNG = np.random.default_rng(42)


def synthesize_houses(n: int = 1000) -> pd.DataFrame:
    sqft = RNG.integers(400, 4500, size=n)
    bedrooms = RNG.integers(1, 7, size=n)
    age = RNG.integers(0, 80, size=n)
    location_score = RNG.uniform(1.0, 9.5, size=n).round(1)

    base_price = (
        sqft * 6500
        + bedrooms * 250000
        + (80 - age) * 12000
        + location_score * 600000
    )
    noise = RNG.normal(0, 250000, size=n)
    price = (base_price + noise).clip(min=500000).round(-3)

    return pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "age": age,
        "location_score": location_score,
        "price": price,
    })


def main() -> None:
    df = synthesize_houses(1000)

    X = df[["sqft", "bedrooms", "age", "location_score"]].values
    y = df["price"].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"R^2 on training data: {model.score(X, y):.3f}")
    print(f"Coefficients: {dict(zip(['sqft','bedrooms','age','location_score'], model.coef_.round(2)))}")

    joblib.dump(model, "model/model.joblib")
    print("Saved model/model.joblib")

    sample = df.sample(n=8, random_state=7).reset_index(drop=True)
    sample.to_csv("data/sample_houses.csv", index=False)
    print("Saved data/sample_houses.csv")


if __name__ == "__main__":
    main()
