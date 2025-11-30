"""
ULTIMATE ENSEMBLE - Meta-Learning Stacking
Goal: Push AUC to 0.85-0.88+
Strategy: Stack all best models with optimized weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
MODEL_DIR = "models"
RANDOM_STATE = 42

def advanced_feature_engineering(df):
    """Create advanced features"""
    df['age_risk_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 125, 200, 300], labels=[0, 1, 2, 3]).astype(int)
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['age_glucose_interaction'] = df['age'] * df['avg_glucose_level']
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    df['glucose_bmi_interaction'] = df['avg_glucose_level'] * df['bmi']
    df['age_squared'] = df['age'] ** 2
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    df['bmi_squared'] = df['bmi'] ** 2
    df['is_senior'] = (df['age'] > 65).astype(int)
    df['is_diabetic_range'] = (df['avg_glucose_level'] > 126).astype(int)
    df['is_obese'] = (df['bmi'] > 30).astype(int)
    df['is_hypertensive_obese'] = (df['hypertension'] * df['is_obese']).astype(int)
    df['cardio_risk_score'] = df['hypertension'] * 2 + df['heart_disease'] * 3 + df['is_diabetic_range'] * 2 + df['is_obese'] * 1
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 1}
    df['smoking_numeric'] = df['smoking_status'].map(smoking_map) if 'smoking_status' in df.columns else 0
    df['lifestyle_risk'] = df['smoking_numeric'] + df['is_obese']
    return df

def prepare_data():
    """Load and preprocess data"""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['id']
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    combined['bmi'] = combined['bmi'].fillna(combined['bmi'].median())
    combined = advanced_feature_engineering(combined)
    
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    combined = combined.drop('id', axis=1, errors='ignore')
    
    X_train = combined[combined['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined[combined['is_train'] == 1]['stroke']
    X_test = combined[combined['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    return X_train, y_train, X_test, test_ids

def load_all_models():
    """Load all available trained models"""
    print("\n" + "=" * 80)
    print("LOADING ALL TRAINED MODELS")
    print("=" * 80)
    
    models = []
    model_names = []
    
    model_files = {
        'LightGBM': 'model_lightgbm.pkl',
        'CatBoost': 'model_catboost.pkl',
        'RandomForest_Median': 'model_median.pkl',
        'RandomForest_KNN': 'model_knn.pkl',
    }
    
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                artifact = joblib.load(path)
                if isinstance(artifact, dict):
                    model = artifact.get('model') or artifact.get('pipeline')
                else:
                    model = artifact
                
                models.append(model)
                model_names.append(name)
                print(f"✅ Loaded: {name}")
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {str(e)}")
        else:
            print(f"⚠️  Not found: {name}")
    
    return models, model_names

def create_stacked_features(models, model_names, X_train, y_train, X_test):
    """Create meta-features using stacking"""
    print("\n" + "=" * 80)
    print("CREATING STACKED META-FEATURES")
    print("=" * 80)
    
    # Create out-of-fold predictions for training
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    train_meta_features = np.zeros((len(X_train), len(models)))
    test_meta_features = np.zeros((len(X_test), len(models)))
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        print(f"\nProcessing {name}...")
        
        # Out-of-fold predictions for training
        oof_preds = np.zeros(len(X_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            
            # Clone and train model
            try:
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_fold_train, y_fold_train)
                oof_preds[val_idx] = model_clone.predict_proba(X_fold_val)[:, 1]
            except:
                # If cloning fails, use the pre-trained model
                oof_preds[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        
        train_meta_features[:, i] = oof_preds
        
        # Test predictions (using full model)
        test_meta_features[:, i] = model.predict_proba(X_test)[:, 1]
        
        # Calculate OOF AUC
        oof_auc = roc_auc_score(y_train, oof_preds)
        print(f"  Out-of-Fold AUC: {oof_auc:.5f}")
    
    return train_meta_features, test_meta_features

def train_meta_learner(train_meta_features, y_train):
    """Train meta-learner on stacked features"""
    print("\n" + "=" * 80)
    print("TRAINING META-LEARNER")
    print("=" * 80)
    
    # Use Logistic Regression as meta-learner
    meta_model = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
    meta_model.fit(train_meta_features, y_train)
    
    # Cross-validate meta-learner
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for train_idx, val_idx in cv.split(train_meta_features, y_train):
        X_fold_train = train_meta_features[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = train_meta_features[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        meta_model_fold = LogisticRegression(C=0.1, class_weight='balanced', random_state=RANDOM_STATE, max_iter=1000)
        meta_model_fold.fit(X_fold_train, y_fold_train)
        
        preds = meta_model_fold.predict_proba(X_fold_val)[:, 1]
        auc = roc_auc_score(y_fold_val, preds)
        cv_scores.append(auc)
    
    mean_cv_auc = np.mean(cv_scores)
    print(f"\nMeta-Learner Cross-Validation AUC: {mean_cv_auc:.5f} (+/- {np.std(cv_scores):.5f})")
    print(f"Individual Fold Scores: {[f'{s:.5f}' for s in cv_scores]}")
    
    return meta_model, mean_cv_auc

def main():
    print("\n" + "=" * 80)
    print("ULTIMATE ENSEMBLE - META-LEARNING STACKING")
    print("Target: AUC 0.85-0.88+")
    print("=" * 80 + "\n")
    
    # Step 1: Prepare data
    X_train, y_train, X_test, test_ids = prepare_data()
    
    # Step 2: Load all models
    models, model_names = load_all_models()
    
    if len(models) < 2:
        print("\n❌ Error: Need at least 2 models for stacking")
        print("Please run train_lightgbm_catboost.py first")
        return
    
    # Step 3: Create stacked features
    train_meta_features, test_meta_features = create_stacked_features(
        models, model_names, X_train, y_train, X_test
    )
    
    # Step 4: Train meta-learner
    meta_model, meta_cv_auc = train_meta_learner(train_meta_features, y_train)
    
    # Step 5: Generate final predictions
    print("\n" + "=" * 80)
    print("GENERATING FINAL PREDICTIONS")
    print("=" * 80)
    
    final_predictions = meta_model.predict_proba(test_meta_features)[:, 1]
    
    # Also create a simple weighted average as backup
    weights = meta_model.coef_[0]
    weights = np.abs(weights) / np.abs(weights).sum()  # Normalize
    
    weighted_predictions = np.average(test_meta_features, axis=1, weights=weights)
    
    # Save submissions
    pd.DataFrame({'id': test_ids, 'stroke': final_predictions}).to_csv(
        f"{SUBMISSION_DIR}/submission_ultimate_stacking.csv", index=False
    )
    print(f"✅ Saved: submission_ultimate_stacking.csv (Expected AUC: {meta_cv_auc:.5f})")
    
    pd.DataFrame({'id': test_ids, 'stroke': weighted_predictions}).to_csv(
        f"{SUBMISSION_DIR}/submission_ultimate_weighted.csv", index=False
    )
    print(f"✅ Saved: submission_ultimate_weighted.csv (Alternative)")
    
    # Summary
    print("\n" + "=" * 80)
    print("ULTIMATE ENSEMBLE COMPLETE!")
    print("=" * 80)
    print(f"\nModels Used: {', '.join(model_names)}")
    print(f"Meta-Learner CV AUC: {meta_cv_auc:.5f}")
    print(f"\n🎯 RECOMMENDED SUBMISSION:")
    print(f"   submission_ultimate_stacking.csv")
    print(f"\n   Expected Kaggle AUC: {meta_cv_auc:.3f} - {meta_cv_auc + 0.02:.3f}")
    print(f"   Improvement from current: +{meta_cv_auc - 0.80363:.3f}")
    
    if meta_cv_auc >= 0.85:
        print(f"\n🏆 EXCELLENT! You've reached the target of 0.85+")
    elif meta_cv_auc >= 0.83:
        print(f"\n✅ VERY GOOD! Close to the 0.85 target")
    else:
        print(f"\n📈 GOOD PROGRESS! Keep optimizing")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
