"""
Evaluate Model Performance Using Cross-Validation
Simulates Kaggle's AUC-ROC scoring on training data
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import os
import glob

# CONFIG
TRAIN_PATH = "sesame-jci-stroke-prediction/train.csv"
MODEL_DIR = "models"
SUBMISSION_DIR = "submissions"

def load_training_data():
    """Load training data"""
    print("Loading training data...")
    df = pd.read_csv(TRAIN_PATH)
    
    X = df.drop(['stroke', 'id'], axis=1, errors='ignore')
    y = df['stroke']
    
    print(f"Training set size: {len(y)} samples")
    print(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y

def evaluate_model_file(model_path, X, y):
    """Evaluate a saved model using cross-validation"""
    try:
        # Load model
        artifact = joblib.load(model_path)
        
        # Extract model (handle different artifact formats)
        if isinstance(artifact, dict):
            if 'model' in artifact:
                model = artifact['model']
            elif 'pipeline' in artifact:
                model = artifact['pipeline']
            else:
                return {'error': 'Unknown artifact format'}
        else:
            model = artifact
        
        # Perform 5-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Calculate AUC scores
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Calculate other metrics (fit on full data for demonstration)
        model.fit(X, y)
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return {
            'mean_cv_auc': auc_scores.mean(),
            'std_cv_auc': auc_scores.std(),
            'cv_scores': auc_scores.tolist(),
            'train_auc': roc_auc_score(y, y_pred_proba),
            'train_accuracy': accuracy_score(y, y_pred),
            'train_f1': f1_score(y, y_pred)
        }
        
    except Exception as e:
        return {'error': str(e)}

def analyze_submission_format(submission_path):
    """Analyze submission file format and statistics"""
    try:
        sub = pd.read_csv(submission_path)
        
        if 'stroke' not in sub.columns:
            return {'error': 'Missing "stroke" column'}
        
        y_pred = sub['stroke'].values
        
        # Determine if probabilities or labels
        is_probability = (y_pred.min() >= 0) and (y_pred.max() <= 1) and (len(np.unique(y_pred)) > 2)
        
        return {
            'num_samples': len(sub),
            'type': 'probabilities' if is_probability else 'labels',
            'min': float(y_pred.min()),
            'max': float(y_pred.max()),
            'mean': float(y_pred.mean()),
            'unique_values': len(np.unique(y_pred)),
            'positive_predictions': int((y_pred > 0.5).sum()) if is_probability else int(y_pred.sum())
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 80)
    print("MODEL PERFORMANCE EVALUATOR")
    print("Cross-Validation on Training Data (Simulates Kaggle Scoring)")
    print("=" * 80)
    print()
    
    # Load training data
    X, y = load_training_data()
    
    # Evaluate saved models
    print("\n" + "=" * 80)
    print("EVALUATING SAVED MODELS (Cross-Validation AUC)")
    print("=" * 80)
    
    model_files = glob.glob(os.path.join(MODEL_DIR, "*.pkl"))
    
    if model_files:
        model_results = []
        
        for model_path in sorted(model_files):
            filename = os.path.basename(model_path)
            print(f"\n📊 Evaluating: {filename}")
            print("-" * 80)
            
            metrics = evaluate_model_file(model_path, X, y)
            
            if 'error' in metrics:
                print(f"❌ Error: {metrics['error']}")
            else:
                print(f"✅ Cross-Validation AUC: {metrics['mean_cv_auc']:.5f} (+/- {metrics['std_cv_auc']:.5f})")
                print(f"   Individual Fold Scores: {[f'{s:.5f}' for s in metrics['cv_scores']]}")
                print(f"   Training AUC: {metrics['train_auc']:.5f}")
                print(f"   Training Accuracy: {metrics['train_accuracy']:.5f}")
                print(f"   Training F1: {metrics['train_f1']:.5f}")
                
                model_results.append({
                    'filename': filename,
                    'cv_auc': metrics['mean_cv_auc'],
                    'cv_std': metrics['std_cv_auc'],
                    'train_auc': metrics['train_auc']
                })
        
        if model_results:
            # Sort by CV AUC
            model_results.sort(key=lambda x: x['cv_auc'], reverse=True)
            
            print("\n" + "=" * 80)
            print("MODEL RANKING (by Cross-Validation AUC)")
            print("=" * 80)
            print(f"\n{'Rank':<6} {'Model':<40} {'CV AUC':<15} {'Std':<10}")
            print("-" * 80)
            
            for i, result in enumerate(model_results, 1):
                rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                print(f"{rank_emoji:<6} {result['filename']:<40} {result['cv_auc']:<15.5f} {result['cv_std']:<10.5f}")
            
            best_model = model_results[0]
            print(f"\n🏆 Best Model: {best_model['filename']} (CV AUC: {best_model['cv_auc']:.5f})")
    else:
        print("\n❌ No model files found in models/")
    
    # Analyze submission files
    print("\n" + "=" * 80)
    print("ANALYZING SUBMISSION FILES")
    print("=" * 80)
    
    submission_files = glob.glob(os.path.join(SUBMISSION_DIR, "*.csv"))
    
    if submission_files:
        submission_results = []
        
        for sub_path in sorted(submission_files):
            filename = os.path.basename(sub_path)
            print(f"\n📄 Analyzing: {filename}")
            print("-" * 80)
            
            stats = analyze_submission_format(sub_path)
            
            if 'error' in stats:
                print(f"❌ Error: {stats['error']}")
            else:
                print(f"   Samples: {stats['num_samples']}")
                print(f"   Type: {stats['type']}")
                print(f"   Range: [{stats['min']:.5f}, {stats['max']:.5f}]")
                print(f"   Mean: {stats['mean']:.5f}")
                print(f"   Unique Values: {stats['unique_values']}")
                print(f"   Positive Predictions: {stats['positive_predictions']} ({stats['positive_predictions']/stats['num_samples']*100:.2f}%)")
                
                # Determine quality
                if stats['type'] == 'probabilities':
                    quality = "✅ Good (probabilities)" if stats['unique_values'] > 100 else "⚠️ Limited diversity"
                else:
                    quality = "⚠️ Labels only (probabilities preferred for AUC)"
                
                print(f"   Quality: {quality}")
                
                submission_results.append({
                    'filename': filename,
                    'type': stats['type'],
                    'samples': stats['num_samples'],
                    'positive_rate': stats['positive_predictions']/stats['num_samples']
                })
        
        # Summary
        print("\n" + "=" * 80)
        print("SUBMISSION FILES SUMMARY")
        print("=" * 80)
        print(f"\n{'Filename':<45} {'Type':<15} {'Positive Rate':<15}")
        print("-" * 80)
        
        for result in submission_results:
            print(f"{result['filename']:<45} {result['type']:<15} {result['positive_rate']:<15.2%}")
        
        # Recommendation
        prob_submissions = [r for r in submission_results if r['type'] == 'probabilities']
        if prob_submissions:
            print(f"\n💡 Recommendation: Submit probability-based files for best AUC scores")
            print(f"   Suggested files:")
            for sub in prob_submissions:
                print(f"   - {sub['filename']}")
    else:
        print("\n❌ No submission files found in submissions/")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    
    if model_results and submission_results:
        best_model_name = model_results[0]['filename'].replace('.pkl', '')
        matching_subs = [s for s in submission_results if best_model_name in s['filename'] and s['type'] == 'probabilities']
        
        if matching_subs:
            print(f"\n🎯 Best submission to upload: {matching_subs[0]['filename']}")
            print(f"   Expected Kaggle AUC: ~{model_results[0]['cv_auc']:.3f}")
        else:
            prob_subs = [s for s in submission_results if s['type'] == 'probabilities']
            if prob_subs:
                print(f"\n🎯 Recommended submission: {prob_subs[0]['filename']}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
