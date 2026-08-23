from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preproccessing():

    zero_categories = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    others = ['Pregnancies', 'Age', 'DiabetesPedigreeFunction']

    zeros_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=0, strategy='median')),
        ('scaler', StandardScaler())
    ])

    others_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preproccessor = ColumnTransformer(transformers=[
        ('zeros', zeros_pipeline, zero_categories),
        ('others', others_pipeline, others)
    ])

    return preproccessor