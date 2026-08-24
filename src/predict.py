import joblib
from pathlib import Path
import pandas as pd

def prediction():
    artifact_path = Path('models/best_model_and_threshold.joblib')
    data_path = Path('data/diabetes.csv')

    artifact = joblib.load(artifact_path)

    model_pipeline = artifact['model']
    threshold = artifact['threshold']
    expected_features = artifact['feature_names']

    data = pd.read_csv(data_path)

    if 'Outcome' in data.columns:
        data = data.drop(columns='Outcome')

    new_data = data[expected_features]

    y_pred_proba_positive = model_pipeline.predict_proba(new_data)[:,1]

    y_preds = (y_pred_proba_positive >= threshold).astype(int)

    return y_pred_proba_positive, y_preds

if __name__ == '__main__':
    probabilities, predictions = prediction()

    results = pd.DataFrame({'Predictions': [predictions], 'Probabilities': [probabilities]})
    print(results)