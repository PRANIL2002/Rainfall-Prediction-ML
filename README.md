# Rainfall Prediction Using Machine Learning

A beginner-to-intermediate machine learning project that predicts whether rainfall is likely (`Yes` / `No`) from recorded weather measurements. The project uses Python, pandas, scikit-learn, imbalanced-learn, and Jupyter Notebook.

> **Project scope:** This is an educational classification project built from a small historical dataset. It is not a live weather service and should not be used for operational weather decisions.

## 1. Problem Statement

Rainfall is a weather outcome influenced by multiple atmospheric conditions. This project explores whether a supervised machine learning classifier can identify rainfall and no-rainfall observations from the available weather features.

## 2. Objectives

- Inspect and clean the weather dataset.
- Handle missing feature values.
- Explore the relationship between weather measurements and rainfall.
- Train and compare classification algorithms.
- Tune a Random Forest model using cross-validation.
- Evaluate classification performance using multiple metrics.
- Interpret feature importance.
- Save the complete preprocessing-and-model pipeline with Joblib.

## 3. Dataset

The notebook uses `Rainfall.csv`.

The dataset contains **366 rows and 12 columns** in the original notebook output. The target column is `rainfall`, with `yes` and `no` labels. The original target distribution recorded in the notebook was:

| Target | Records |
|---|---:|
| Yes (rainfall) | 249 |
| No (no rainfall) | 117 |

The model uses these eight weather features after removing the non-selected columns:

- `pressure`
- `temparature` (spelling retained from the source CSV)
- `dewpoint`
- `humidity`
- `cloud`
- `sunshine`
- `winddirection`
- `windspeed`

The notebook removes `day`, `maxtemp`, and `mintemp` from the feature matrix. Column headers are stripped of leading and trailing whitespace before feature selection.

## 4. Technologies

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- Joblib

## 5. Machine Learning Workflow

1. Load the CSV and inspect its shape, columns, and missing values.
2. Clean column names and encode the target (`yes` → `1`, `no` → `0`).
3. Split the data into training and test subsets using a stratified split.
4. Build preprocessing/model pipelines with median imputation, feature scaling, and RandomOverSampler.
5. Compare Logistic Regression, Decision Tree, Random Forest, KNN, and SVM.
6. Tune Random Forest hyperparameters with `GridSearchCV` and 5-fold `StratifiedKFold`.
7. Evaluate the tuned model with accuracy, precision, recall, F1-score, ROC-AUC, a classification report, a confusion matrix, and an ROC curve.
8. Examine built-in and permutation feature importance.
9. Save the full fitted pipeline as `models/rainfall_model.pkl`.

Oversampling is included inside the imbalanced-learn pipeline so it is applied during model fitting rather than to the complete dataset before splitting.

## 6. Algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM with RBF kernel)

Random Forest is additionally tuned using GridSearchCV.

## 7. Recorded Results

The original notebook run recorded the following model-comparison results. Re-run the corrected notebook before treating these values as the final reproducible results for a public release.

| Algorithm | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8243 | 0.8776 | 0.8600 | 0.8687 | 0.9233 |
| SVM | 0.7973 | 0.8571 | 0.8400 | 0.8485 | 0.9188 |
| Random Forest | 0.8108 | 0.8600 | 0.8600 | 0.8600 | 0.8725 |
| KNN | 0.7432 | 0.8444 | 0.7600 | 0.8000 | 0.7946 |
| Decision Tree | 0.7027 | 0.7692 | 0.8000 | 0.7843 | 0.6500 |

The recorded tuned Random Forest test metrics were:

| Metric | Recorded value |
|---|---:|
| Accuracy | 0.8108 |
| Precision | 0.8913 |
| Recall | 0.8200 |
| F1-score | 0.8542 |
| ROC-AUC | 0.8892 |

The tuned Random Forest's recorded 5-fold cross-validation ROC-AUC was **0.8464**.

**Metric note:** ROC-AUC measures ranking/discrimination across classification thresholds; it is not the same as accuracy.

## 8. Repository Structure

Organize the GitHub repository like this:

```text
Rainfall-Prediction-ML/
├── dataset/
│   └── Rainfall.csv
├── notebooks/
│   └── Rainfall_Prediction.ipynb
├── models/
│   └── rainfall_model.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

Place your corrected notebook in `notebooks/`, the CSV in `dataset/`, and the generated model in `models/`. If your notebook currently reads `Rainfall.csv` from the project root, either keep the CSV in the root or update the notebook path to `../dataset/Rainfall.csv` after moving it.

## 9. Setup and Run

### Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run the notebook

```powershell
jupyter notebook
```

Open `notebooks/Rainfall_Prediction.ipynb` and run all cells from top to bottom. The notebook should create the `models/` directory and save `rainfall_model.pkl`.

## 10. Model Artifact

The saved file `models/rainfall_model.pkl` contains the fitted pipeline, including preprocessing and the classifier. Load it only from a trusted source, because Joblib uses Python object serialization.

Example:

```python
import joblib

model = joblib.load("models/rainfall_model.pkl")
```

The model expects the same eight feature names and order used during training.

## 11. Important Limitations and Future Improvements

- The dataset is small (366 rows), so performance estimates may vary and generalization is uncertain.
- This is a demonstration model, not a substitute for official meteorological forecasts.
- The notebook's earlier workflow used the test set for model comparison and then for later evaluation. This means the test set is no longer a completely untouched final holdout. For a stronger portfolio evaluation, compare and tune models using cross-validation on training data, then evaluate the selected model once on a fresh holdout set.
- Feature importance indicates model reliance or predictive association; it does not prove that a feature causes rainfall.
- Future work could use a larger, time-aware dataset, external validation, calibration, and a documented decision threshold.

## 12. Author

Add your name, GitHub profile, and LinkedIn profile here before publishing.

- GitHub: [Add your GitHub profile]
- LinkedIn: [Add your LinkedIn profile]

---

**Disclaimer:** This repository is for learning and demonstration purposes.
