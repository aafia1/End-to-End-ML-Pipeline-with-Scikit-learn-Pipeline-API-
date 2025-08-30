import argparse
import pandas as pd
import json
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data(csv_path):
    return pd.read_csv(csv_path)

def build_pipeline(model):
    # Separate numeric and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                            'PhoneService', 'MultipleLines', 'InternetService',
                            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies',
                            'Contract', 'PaperlessBilling', 'PaymentMethod']

    # Preprocessors
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('clf', model)])
    return pipeline

def main(args):
    # Load dataset
    df = load_data(args.csv)

    # Target variable
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models to try
    models = {
        "log_reg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(random_state=42)
    }

    # Hyperparameter grids
    param_grids = {
        "log_reg": {
            "clf__C": [0.1, 1.0, 10]
        },
        "rf": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [8, 16],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2]
        }
    }

    best_model = None
    best_score = 0
    best_params = {}

    # GridSearch for each model
    for name, model in models.items():
        pipeline = build_pipeline(model)
        grid_search = GridSearchCV(pipeline, param_grids[name],
                                   cv=3, scoring='accuracy', n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        score = grid_search.best_score_
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save metrics (only JSON-safe objects)
    metrics = {
        "best_params": best_params,
        "cv_best_accuracy": best_score,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "classification_report": classification_report(y_test, y_pred, output_dict=True)
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save trained model
    joblib.dump(best_model, "models/churn_model.pkl")

    print("✅ Training complete!")
    print("📂 Metrics saved to models/metrics.json")
    print("📂 Model saved to models/churn_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV file")
    args = parser.parse_args()
    main(args)
