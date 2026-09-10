import joblib
from pathlib import Path
import pandas as pd


MODEL_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "best_model_and_threshold.joblib"
)


def load_model():
    """
    Load the trained model, threshold and expected feature names.
    """

    artifact = joblib.load(MODEL_PATH)

    model_pipeline = artifact["model"]
    threshold = artifact["threshold"]
    expected_features = artifact["feature_names"]

    return model_pipeline, threshold, expected_features


def prediction(data):
    """
    Make a prediction for a single patient.
    """

    model_pipeline, threshold, expected_features = load_model()

    new_data = pd.DataFrame(
        [data],
        columns=expected_features
    )

    probability = model_pipeline.predict_proba(
        new_data
    )[0, 1]

    pred = int(probability >= threshold)

    return probability, pred


if __name__ == "__main__":

    data_path = (
        Path(__file__).parent.parent
        / "data"
        / "diabetes.csv"
    )

    data = pd.read_csv(data_path)

    if "Outcome" in data.columns:
        data = data.drop(columns="Outcome")

    model_pipeline, threshold, expected_features = load_model()

    probabilities = model_pipeline.predict_proba(
        data[expected_features]
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results = pd.DataFrame({
        "Predictions": predictions,
        "Probabilities": probabilities
    })

    print(results.head())

