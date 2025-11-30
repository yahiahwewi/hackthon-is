# 📊 Preprocessing Methods Comparison

This report compares different preprocessing strategies for handling missing BMI values.

## 🎯 Objective
Train separate models using different imputation methods and evaluate:
- **Performance** (AUC-ROC, F1-Score)
- **Overfitting** (Train vs Validation gap)
- **Submission quality** (predictions on test set)

## 📈 Results Summary

| Method | Train AUC | Val AUC | Train F1 | Val F1 | Overfit Δ AUC | Overfit Δ F1 |
|--------|-----------|---------|----------|--------|---------------|-------------|
| **KNN** | 0.9984 | 0.7828 | 0.8674 | 0.1408 | 0.2156 | 0.7266 |
| **MEDIAN** | 0.9986 | 0.7885 | 0.8650 | 0.1039 | 0.2101 | 0.7611 |

## 📊 Visual Comparison

![Preprocessing Comparison](plots\preprocessing_comparison.png)

## 🔍 Analysis

### Overfitting Detection
- **Δ AUC/F1 < 0.02**: Minimal overfitting ✅
- **Δ AUC/F1 0.02-0.05**: Moderate overfitting ⚠️
- **Δ AUC/F1 > 0.05**: Significant overfitting ❌

### 🏆 Best Method: **MEDIAN**
- Validation AUC: **0.7885**
- Submission file: `submissions/submission_median.csv`

## 📁 Generated Files

### Models
- `models/model_knn.pkl`
- `models/model_median.pkl`

### Submissions
- `submissions/submission_knn.csv` (format: id, stroke)
- `submissions/submission_median.csv` (format: id, stroke)

## 💡 Recommendations

1. **For Kaggle submission**: Use `submission_median.csv`
2. **For production API**: Load `model_median.pkl` in `main.py`
3. **For further improvement**: Consider ensemble methods or hyperparameter tuning
