"""
ULTIMATE APPROACH - AutoML with H2O
Automatically finds the best model and hyperparameters
Target: 0.85-0.90+ AUC
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
TEST_PATH = "sesame-jci-stroke-prediction/test.csv"
SUBMISSION_DIR = "submissions"
RANDOM_STATE = 42

def extreme_feature_engineering(df):
    """Create 60+ advanced features"""
    
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    # Age features
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['age_sqrt'] = np.sqrt(df['age'])
    df['age_log'] = np.log1p(df['age'])
    df['is_senior'] = (df['age'] > 65).astype(int)
    df['is_very_senior'] = (df['age'] > 75).astype(int)
    df['is_young'] = (df['age'] < 40).astype(int)
    
    # Glucose features
    df['glucose_category'] = pd.cut(df['avg_glucose_level'], bins=[0, 100, 125, 200, 300], labels=[0, 1, 2, 3]).astype(int)
    df['glucose_squared'] = df['avg_glucose_level'] ** 2
    df['glucose_log'] = np.log1p(df['avg_glucose_level'])
    df['is_diabetic'] = (df['avg_glucose_level'] > 126).astype(int)
    df['is_prediabetic'] = ((df['avg_glucose_level'] > 100) & (df['avg_glucose_level'] <= 126)).astype(int)
    df['is_very_high_glucose'] = (df['avg_glucose_level'] > 200).astype(int)
    
    # BMI features
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    df['bmi_squared'] = df['bmi'] ** 2
    df['bmi_log'] = np.log1p(df['bmi'])
    df['is_obese'] = (df['bmi'] >= 30).astype(int)
    df['is_very_obese'] = (df['bmi'] >= 40).astype(int)
    df['is_overweight'] = ((df['bmi'] >= 25) & (df['bmi'] < 30)).astype(int)
    
    # Interaction features
    df['age_x_glucose'] = df['age'] * df['avg_glucose_level']
    df['age_x_bmi'] = df['age'] * df['bmi']
    df['glucose_x_bmi'] = df['avg_glucose_level'] * df['bmi']
    df['age_x_glucose_x_bmi'] = df['age'] * df['avg_glucose_level'] * df['bmi']
    df['senior_x_diabetic'] = df['is_senior'] * df['is_diabetic']
    df['senior_x_obese'] = df['is_senior'] * df['is_obese']
    df['diabetic_x_obese'] = df['is_diabetic'] * df['is_obese']
    df['hypertension_x_diabetic'] = df['hypertension'] * df['is_diabetic']
    df['hypertension_x_obese'] = df['hypertension'] * df['is_obese']
    df['heart_disease_x_hypertension'] = df['heart_disease'] * df['hypertension']
    
    # Risk scores
    df['cardio_risk_score'] = (
        df['hypertension'] * 3 + 
        df['heart_disease'] * 4 + 
        df['is_diabetic'] * 2 +
        df['is_obese'] * 1 +
        df['is_senior'] * 2
    )
    df['metabolic_risk_score'] = df['is_diabetic'] * 2 + df['is_obese'] * 2 + df['is_overweight'] * 1
    
    # Smoking features
    smoking_map = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 3, 'Unknown': 1}
    df['smoking_numeric'] = df['smoking_status'].map(smoking_map) if 'smoking_status' in df.columns else 0
    df['lifestyle_risk'] = df['smoking_numeric'] + df['is_obese'] * 2
    df['smoking_x_age'] = df['smoking_numeric'] * df['age']
    
    # Statistical features
    df['all_vitals_sum'] = df['age'] + df['avg_glucose_level'] + df['bmi']
    df['all_vitals_mean'] = (df['age'] + df['avg_glucose_level'] + df['bmi']) / 3
    
    return df

def train_super_ensemble(X_train, y_train, X_test):
    """Train multiple models and ensemble them"""
    print("\n" + "=" * 80)
    print("TRAINING SUPER ENSEMBLE")
    print("=" * 80)
    
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    import xgboost as xgb
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    
    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_weight = neg / pos
    
    # Define models with aggressive hyperparameters
    models = {
        'LightGBM_v1': LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.005,
            num_leaves=31,
            max_depth=8,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        'LightGBM_v2': LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            num_leaves=50,
            max_depth=10,
            min_child_samples=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=0.5,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE + 1,
            n_jobs=-1,
            verbose=-1
        ),
        'CatBoost_v1': CatBoostClassifier(
            iterations=2000,
            learning_rate=0.005,
            depth=8,
            l2_leaf_reg=5,
            subsample=0.8,
            colsample_bylevel=0.8,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=-1
        ),
        'CatBoost_v2': CatBoostClassifier(
            iterations=1500,
            learning_rate=0.01,
            depth=10,
            l2_leaf_reg=3,
            subsample=0.7,
            colsample_bylevel=0.7,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE + 1,
            verbose=False,
            thread_count=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.005,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='auc'
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=1000,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        )
    }
    
    # Train and evaluate each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_scores = {}
    test_predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict_proba(X_fold_val)[:, 1]
            auc = roc_auc_score(y_fold_val, preds)
            cv_scores.append(auc)
        
        mean_cv_auc = np.mean(cv_scores)
        model_scores[name] = mean_cv_auc
        print(f"  CV AUC: {mean_cv_auc:.5f} (+/- {np.std(cv_scores):.5f})")
        
        # Train on full data and predict
        model.fit(X_train, y_train)
        test_predictions[name] = model.predict_proba(X_test)[:, 1]
    
    # Create weighted ensemble
    print("\n" + "=" * 80)
    print("CREATING WEIGHTED ENSEMBLE")
    print("=" * 80)
    
    # Weight by CV scores
    total_score = sum(model_scores.values())
    weights = {name: score / total_score for name, score in model_scores.items()}
    
    print("\nModel Weights:")
    for name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name}: {weight:.4f} (CV AUC: {model_scores[name]:.5f})")
    
    # Weighted average
    ensemble_pred = np.zeros(len(X_test))
    for name, pred in test_predictions.items():
        ensemble_pred += pred * weights[name]
    
    best_model_name = max(model_scores, key=model_scores.get)
    best_cv_auc = model_scores[best_model_name]
    
    return ensemble_pred, test_predictions, model_scores, best_cv_auc

def main():
    print("\n" + "=" * 80)
    print("ULTIMATE SUPER ENSEMBLE")
    print("7 Models with Aggressive Hyperparameters")
    print("Target: 0.85-0.90+ AUC")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    test_ids = test_df['id']
    
    # Combine
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['stroke'] = -1
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # Extreme feature engineering
    print("Creating 50+ features...")
    combined = extreme_feature_engineering(combined)
    
    # One-hot encoding
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    combined = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    combined = combined.drop('id', axis=1, errors='ignore')
    
    # Split
    X_train = combined[combined['is_train'] == 1].drop(['stroke', 'is_train'], axis=1)
    y_train = combined[combined['is_train'] == 1]['stroke']
    X_test = combined[combined['is_train'] == 0].drop(['stroke', 'is_train'], axis=1)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Train super ensemble
    ensemble_pred, individual_preds, model_scores, best_cv_auc = train_super_ensemble(X_train, y_train, X_test)
    
    # Save submissions
    print("\n" + "=" * 80)
    print("SAVING SUBMISSIONS")
    print("=" * 80)
    
    # Ensemble
    pd.DataFrame({
        'id': test_ids,
        'stroke': ensemble_pred
    }).to_csv(f"{SUBMISSION_DIR}/submission_super_ensemble_7models.csv", index=False)
    print(f"\n✅ Saved: submission_super_ensemble_7models.csv")
    
    # Best individual model
    best_model_name = max(model_scores, key=model_scores.get)
    pd.DataFrame({
        'id': test_ids,
        'stroke': individual_preds[best_model_name]
    }).to_csv(f"{SUBMISSION_DIR}/submission_best_single_{best_model_name.lower()}.csv", index=False)
    print(f"✅ Saved: submission_best_single_{best_model_name.lower()}.csv")
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nBest Single Model: {best_model_name}")
    print(f"Best CV AUC: {best_cv_auc:.5f}")
    print(f"\n🎯 RECOMMENDED SUBMISSION:")
    print(f"   submission_super_ensemble_7models.csv")
    print(f"   Expected Kaggle AUC: {best_cv_auc:.3f} - {best_cv_auc + 0.02:.3f}")
    
    if best_cv_auc >= 0.85:
        print(f"\n🏆 EXCELLENT! You've reached 0.85+ AUC!")
    elif best_cv_auc >= 0.83:
        print(f"\n✅ VERY GOOD! Close to the 0.85 target!")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
