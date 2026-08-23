from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from preprocessing import preproccessing
import joblib


def main():
    data_path = Path('data/diabetes.csv')
    model_path = Path('models')
    artifact_path = model_path / 'best_model_and_threshold.joblib'

    data = pd.read_csv(data_path)

    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = preproccessing()
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=0.1, class_weight='balanced', max_iter=200, l1_ratio=1, solver='liblinear'))
    ])

    model_pipeline.fit(X_train, y_train)

    artifact = {
        'model': model_pipeline,
        'threshold': 0.36,
        'feature_names': X_train.columns.tolist()
    }
    
    joblib.dump(artifact, artifact_path)

if __name__ == '__main__':
    main()