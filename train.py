import joblib
from sklearn.model_selection import train_test_split
from src.data.loader import load_data, clean_data
from src.features.preprocessing import get_preprocessor
from src.models.train_model import build_pipeline
from src.models.evaluate import evaluate_model

# CONFIG
DATA_PATH = 'sesame-jci-stroke-prediction/train.csv'
MODEL_PATH = 'model.pkl'
RANDOM_STATE = 42

def main():
    print("1. Loading Data...")
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        print(e)
        return
        
    df = clean_data(df)

    # Split
    X = df.drop('stroke', axis=1)
    y = df['stroke']
    
    # Identify columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print("2. Building Pipeline...")
    preprocessor = get_preprocessor(numerical_cols, categorical_cols)
    model = build_pipeline(preprocessor, RANDOM_STATE)

    print("3. Training (with SMOTE)...")
    model.fit(X_train, y_train)

    print("4. Evaluating...")
    auc, threshold, report = evaluate_model(model, X_test, y_test)
    
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Optimal Threshold: {threshold:.4f}")
    print("\n" + report)

    print("5. Saving Artifacts...")
    artifact = {
        'model': model,
        'base_threshold': threshold,
        'features_num': list(numerical_cols),
        'features_cat': list(categorical_cols)
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
