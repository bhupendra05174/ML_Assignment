# Adult Income Classification - ML Assignment

**Student:** Bhupendra Kumar  
**BITS ID:** 2026CT05174  
**BITS Email:** 2026ct05174@wilp.bits-pilani.ac.in  
**Course:** Machine Learning (M.Tech AIML/DSE, WILP)

## Problem statement
Build and compare multiple classification models on the same public dataset to predict whether a person's annual income is **`>50K`** or **`<=50K`** (binary classification). The assignment implements five required classifiers, evaluates them with Accuracy, AUC, Precision, Recall, F1, and MCC, and includes preprocessing, leakage-safe pipelines, model saving, visualizations, and generated artifacts.

## Dataset description
- **Dataset:** Adult Census Income (Adult) dataset from the UCI Machine Learning Repository, loaded programmatically via OpenML (`fetch_openml(name='adult', version=2, as_frame=True, parser='auto')`).
- **Why selected:** Public, well-known benchmark for income classification with mixed numerical/categorical attributes, missing values, and class imbalance; suitable for comparing linear, tree-based, distance-based, probabilistic, and ensemble models.
- **Size (before cleaning):** 48,842 instances and 15 columns total (14 input features + 1 target).
- **Size (after duplicate removal):** 48,790 instances and 15 columns (52 duplicate rows removed).
- **Target variable:** `class` (income), normalized to `<=50K` and `>50K`; encoded for modeling as `income_gt_50k` (1 = `>50K`, 0 = `<=50K`).
- **Input features (14):**
  - Numerical (6): `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
  - Categorical (8): `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Missing values (handled):** `occupation` = 2809, `workclass` = 2799, `native-country` = 857 (imputed in preprocessing pipelines).
- **Class distribution (after cleaning, examined imbalance):** `<=50K` = 37,109 (76.06%), `>50K` = 11,681 (23.94%), minority-class share = 23.94%.

## GitHub repository link
- https://github.com/bhupendra05174/ML_Assignment

> Ensure the repository contains the complete source notebook/code, `requirements.txt`, this `README.md`, `test_data.csv`, and the saved model files in `model/`.

## Preprocessing
- Missing values identified and handled: median imputation for numerical features, most-frequent imputation for categorical features.
- Duplicate rows identified and removed.
- Categorical encoding: one-hot encoding (`handle_unknown='ignore'`).
- Feature scaling: standardization for numerical features (`StandardScaler`).
- Train-test split: **80:20** stratified split with fixed random seed **42** (`stratify=y`), preserving class distribution (~76.06% / ~23.94% in train and test).
- Leakage prevention: all preprocessing is performed inside scikit-learn `Pipeline`/`ColumnTransformer` and fitted only on training data during model training.
- Test data exported for submission/evaluation: `test_data.csv` (9,758 rows, test features + `income_gt_50k`).

## Models used and configurations
All five required models were trained on the same dataset using the same preprocessing pipeline:
| ML Model Name | Configuration (key parameters) |
|---|---|
| Logistic Regression | `LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)` |
| Decision Tree | `DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=42)` |
| kNN | `KNeighborsClassifier(n_neighbors=7, weights='distance')` |
| Naive Bayes | `GaussianNB()` |
| Random Forest (Ensemble) | `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)` |

## Results (comparison table)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (Ensemble) | 0.8193 | 0.9148 | 0.5847 | 0.8467 | 0.6917 | 0.5895 |
| Logistic Regression | 0.8045 | 0.9022 | 0.5602 | 0.8527 | 0.6762 | 0.5692 |
| Decision Tree | 0.7982 | 0.8942 | 0.5500 | 0.8643 | 0.6722 | 0.5653 |
| kNN | 0.8305 | 0.8669 | 0.6607 | 0.6002 | 0.6290 | 0.5205 |
| Naive Bayes | 0.6173 | 0.8329 | 0.3777 | 0.9242 | 0.5363 | 0.3855 |


## Visualizations (graph representation)
### Model metrics comparison
![Model metrics comparison](model_metrics_comparison.png)

### ROC curves
![ROC curves](roc_curves.png)

### Confusion matrices
![Confusion matrices](confusion_matrices.png)

## Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with balanced class weighting; high recall (0.853) and AUC (0.902) with good MCC (0.569), though precision (0.560) is moderate due to more false positives on the >50K class. |
| Decision Tree | Captures non-linear patterns with high recall (0.864) and competitive AUC (0.894), but precision (0.550) and MCC (0.565) are slightly lower than Logistic Regression/Random Forest, indicating limited generalization compared with the ensemble. |
| kNN | Achieves the highest accuracy (0.830) and precision (0.661), but lower recall (0.600), F1 (0.629), and MCC (0.520) show it misses many >50K cases; distance-based performance depends on scaling and neighborhood choice. |
| Naive Bayes | Very high recall (0.924) but the lowest accuracy (0.617), precision (0.378), F1 (0.536), and MCC (0.385), indicating many false positives because the conditional-independence assumption is restrictive for correlated census features. |
| Random Forest (Ensemble) | Best overall model by MCC (0.589), F1 (0.692), and AUC (0.915), with strong recall (0.847) and better balance of errors than single models, benefiting from ensemble averaging and non-linear feature interactions. |


## Overall winner for this dataset
**Random Forest (Ensemble)** is the overall winner based on the highest **MCC (0.5895)**, **F1 (0.6917)**, and **AUC (0.9148)**, indicating the best balanced performance on the imbalanced income classes. kNN achieved the highest accuracy (0.8305) and Naive Bayes achieved the highest recall (0.9242), but with weaker precision/MCC trade-offs.

## Execution instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open and run all cells in `2026CT05174_ml_assignment.ipynb` (internet access required on first run to download the Adult dataset via OpenML).
3. Verify generated artifacts:
   - `README.md`, `test_data.csv`, `model_comparison.csv`, `model_observations.txt`
   - `confusion_matrices.png`, `model_metrics_comparison.png`, `roc_curves.png`
   - `model/logistic_regression.joblib`, `model/decision_tree.joblib`, `model/knn.joblib`, `model/naive_bayes.joblib`, `model/random_forest_ensemble.joblib`

## Submission checklist (manual items)
- [ ] Run the notebook in BITS Virtual Lab and capture the execution screenshot.
- [ ] Save the notebook with outputs.
- [ ] Push/update the GitHub repository and verify the link works.
- [ ] Prepare a single PDF containing: GitHub repo link, required files/evidence, test data, BITS Virtual Lab screenshot, and this README content (in the required order).
