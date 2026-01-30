# Student Performance Classification with AI Usage Analysis

## a. Problem Statement
The goal of this project is to analyze student performance based on demographic, academic, and AI tool usage data. We aim to build a classification model that can predict whether a student will **Pass** or **Fail** (based on the `passed` binary indicator). This helps in identifying at-risk students and understanding how AI usage correlates with academic success.

## b. Dataset Description
The dataset contains information about students, including:
- **Demographics**: Age, Gender, Grade Level.
- **Academic Info**: Study hours, Attendance, Previous scores, etc.
- **AI Usage**: Usage of AI tools, dependency scores, ethics scores, and purpose.
- **Target**: `passed` (1 = Yes, 0 = No).

**Key Features used**: `study_hours_per_day`, `attendance_percentage`, `last_exam_score`, `ai_dependency_score`, `ai_usage_time_minutes`, etc.

## c. Models Used
We implemented and evaluated the following 6 Machine Learning models:
1.  **Logistic Regression**
2.  **Decision Tree Classifier**
3.  **K-Nearest Neighbor (kNN)**
4.  **Naive Bayes (Gaussian)**
5.  **Random Forest Classifier** (Ensemble)
6.  **XGBoost Classifier** (Ensemble)

### Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9975 | 0.9998 | 0.9972 | 1.0000 | 0.9986 | 0.9871 |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **kNN** | 0.9231 | 0.9116 | 0.9272 | 0.9916 | 0.9583 | 0.5209 |
| **Naive Bayes** | 0.9719 | 0.9963 | 0.9943 | 0.9741 | 0.9841 | 0.8681 |
| **Random Forest** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost** | 0.9975 | 1.0000 | 0.9979 | 0.9993 | 0.9986 | 0.9871 |

### Observations about Model Performance

1.  **Logistic Regression**: Excellent performance with near-perfect accuracy (99.75%). It suggests the data is linearly separable or features are highly predictive.
2.  **Decision Tree**: Achieved a perfect score (1.0). While this indicates the model captured the patterns perfectly, it might be overfitted to the training data, though performance on the test set is also perfect here.
3.  **kNN**: The lowest performer (92.3%) with a significantly lower MCC (0.52). Ideally, kNN struggles with higher dimensionality or irrelevant features without extensive feature selection.
4.  **Naive Bayes**: Performs well (97.2%) but slightly worse than tree-based methods, likely due to the independence assumption not holding fully true for all features.
5.  **Random Forest (Ensemble)**: Matches Decision Tree with perfect scores (1.0). As an ensemble method, it is generally more robust than a single tree, confirming the strong signal in the dataset.
6.  **XGBoost (Ensemble)**: Exceptional performance (99.75%), effectively matching Logistic Regression. It handles complex relationships well and is very robust.

**Conclusion**: The dataset features are highly predictive of the `passed` status. Tree-based models (Decision Tree, Random Forest) and XGBoost/Logistic Regression are all suitable choices. For deployability and interpretability, Logistic Regression or Random Forest are recommended.
