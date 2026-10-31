# VIMARSH Milestone 1: Baseline Experiment Report

## 1. Research Objective
This report presents the results of VIMARSH's first milestone: establishing baseline machine learning models and investigating the relationship between prediction confidence and actual prediction correctness on the Wine Quality dataset.

## 2. Dataset Description and Provenance
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/186/wine+quality
- **Download Date**: 2026-10-30T19:59:55.149903
- **Total combined samples**: 6497
-  - Red wines: 1599
-  - White wines: 4898
- **Features**: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, quality, wine_type
- **Missing values**: 0 total
- **Duplicate records**: 1177

## 3. Data Preprocessing
- Binary target created: quality >= 6 = 'good quality' (experimental threshold choice)
- Train/validation/test split: 70/15/15 (stratified)
- Random seed: 42
- Standardization applied to all numerical features (fit only on training data)

## 4. Experimental Setup
Three baseline models were trained and evaluated:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
Probability calibration using isotonic regression (fit on validation data only)

## 5. Evaluation Methodology
Models evaluated on held-out test set with comprehensive metrics:
- Discrimination: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix
- Calibration: Expected Calibration Error (ECE), Brier Score, Log Loss

## 6. Model Performance Results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | ECE | Brier Score |
|-------|----------|-----------|--------|----|---------|-----|-------------|
| Logistic Regression | 0.7436 | 0.7671 | 0.8541 | 0.8083 | 0.8096 | 0.0330 | 0.1713 |
| Random Forest | 0.8287 | 0.8430 | 0.8963 | 0.8688 | 0.8983 | 0.0661 | 0.1264 |
| Gradient Boosting | 0.7754 | 0.7935 | 0.8720 | 0.8309 | 0.8448 | 0.0441 | 0.1549 |

## 7. Confidence vs Correctness Findings

### Logistic Regression
- Total test samples: 975
- High confidence (>=90%) predictions: 155
- Incorrect high-confidence predictions: 6
- Error rate in high-confidence predictions: 3.87%

Confidence bucket analysis:
| Bucket | Count | Avg Confidence | Actual Accuracy | Incorrect |
|--------|-------|----------------|-----------------|-----------|
| 50-60% | 205 | 0.5490 | 0.5805 | 86 |
| 60-70% | 190 | 0.6479 | 0.6158 | 73 |
| 70-80% | 216 | 0.7483 | 0.7407 | 56 |
| 80-90% | 209 | 0.8488 | 0.8612 | 29 |
| 90-100% | 155 | 0.9385 | 0.9613 | 6 |

### Random Forest
- Total test samples: 975
- High confidence (>=90%) predictions: 249
- Incorrect high-confidence predictions: 7
- Error rate in high-confidence predictions: 2.81%

Confidence bucket analysis:
| Bucket | Count | Avg Confidence | Actual Accuracy | Incorrect |
|--------|-------|----------------|-----------------|-----------|
| 50-60% | 140 | 0.5454 | 0.5571 | 62 |
| 60-70% | 185 | 0.6450 | 0.7189 | 52 |
| 70-80% | 185 | 0.7474 | 0.8541 | 27 |
| 80-90% | 216 | 0.8436 | 0.9120 | 19 |
| 90-100% | 220 | 0.9483 | 0.9682 | 7 |

### Gradient Boosting
- Total test samples: 975
- High confidence (>=90%) predictions: 165
- Incorrect high-confidence predictions: 4
- Error rate in high-confidence predictions: 2.42%

Confidence bucket analysis:
| Bucket | Count | Avg Confidence | Actual Accuracy | Incorrect |
|--------|-------|----------------|-----------------|-----------|
| 50-60% | 157 | 0.5505 | 0.5541 | 70 |
| 60-70% | 170 | 0.6510 | 0.6529 | 59 |
| 70-80% | 252 | 0.7513 | 0.7698 | 58 |
| 80-90% | 231 | 0.8464 | 0.8788 | 28 |
| 90-100% | 165 | 0.9414 | 0.9758 | 4 |

## 8. Calibration Findings
Isotonic regression calibration was applied to all models using only validation data. Reliability diagrams were generated for both original and calibrated probabilities.

## 9. Visual Observations
All required visualizations have been generated:
- Class distribution chart
- Model accuracy comparison
- Model ROC curves
- Confusion matrices for all models
- Confidence distributions (correct/incorrect predictions)
- Confidence vs empirical accuracy plots
- Reliability diagrams (before/after calibration)
- Brier Score comparison
- ECE comparison

## 10. Limitations
- This is a baseline experiment using a single dataset (Wine Quality)
- Minimal hyperparameter tuning was performed
- Binary classification only (multi-class not explored)
- Calibration evaluated on a single test partition

## 11. Conclusions
Baseline models have been successfully established. The relationship between confidence and correctness has been characterized, including the existence of high-confidence incorrect predictions that demonstrate the need for verification mechanisms. Calibration improved probability reliability for some models.

## 12. Next Research Steps
1. Extend analysis to additional datasets (Breast Cancer Wisconsin, Covertype)
2. Explore additional reliability signals (model disagreement, OOD detection)
3. Begin development of the verification action selection framework
4. Evaluate risk-aware decision policies on benchmark datasets