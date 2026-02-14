# Student Performance Classification with AI Usage Analysis

## a. Problem Statement
The goal of this project is to analyze student performance based on demographic, academic, and AI usage data. We build classification models to predict whether a student will **Pass** or **Fail** using a binary target variable (`passed`).  

This system helps:
- Identify students at academic risk
- Understand how study behavior and AI usage relate to performance
- Support early academic intervention strategies

---

## b. Dataset Features Description

The dataset contains student study behavior, AI usage patterns, academic indicators, and lifestyle factors.

### Key Feature Groups
- **Study & Academic Behavior:** study_hours_per_day, study_consistency_index, concept_understanding_score, improvement_rate, last_exam_score, assignment_scores_avg, attendance_percentage, class_participation_score, tutoring_hours  
- **AI Usage:** uses_ai, ai_usage_time_minutes, ai_tools_used, ai_usage_purpose, ai_dependency_score, ai_generated_content_percentage, ai_prompts_per_week, ai_ethics_score  
- **Lifestyle:** sleep_hours, social_media_hours  

### Target Variables
- **passed** – Binary outcome (1 = Pass, 0 = Fail)  

_Note: The feature `final_score` was removed because it directly determines pass/fail status and would lead to unrealistic model performance. Also removed two unrelated features named as `student_id` and `performance_category`_

### Class Distribution
The dataset is imbalanced:
- ~89% Pass
- ~11% Fail

Because of this imbalance, **F1 Score** was selected as the primary metric for model comparison.

---

## c. Models Used

Six classification algorithms were implemented and compared:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

### Handling Class Imbalance
- Logistic Regression, Decision Tree, Random Forest:
  - `class_weight='balanced'`
---

## d. Metrics Comparison (Final Results)

**Best model is selected based on F1 Score** since it balances Precision and Recall for an imbalanced dataset.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8997 | 0.9720 | 0.9891 | 0.8969 | 0.9407 | **0.6522** |
| Decision Tree | 0.9154 | 0.7766 | 0.9495 | 0.9555 | 0.9525 | 0.5659 |
| kNN | 0.9003 | 0.8448 | 0.9089 | 0.9866 | 0.9462 | 0.3448 |
| **Naive Bayes🏆** | **0.9335** | 0.9637 | 0.9396 | 0.9887 | **0.9635** | 0.6185 |
| Random Forest | 0.9216 | 0.9594 | 0.9211 | **0.9972** | 0.9576 | 0.5245 |
| XGBoost | 0.9292 | 0.9622 | 0.9490 | 0.9725 | 0.9606 | 0.6157 |

---

## e. Observations About Model Performance

Since the dataset is imbalanced, **F1 Score** was used to determine the best model.

### 🏆 Naive Bayes (Best Model – Highest F1 Score: 0.9635)
- Highest accuracy and F1 score
- Strong balance between precision and recall
- Very stable performance across predictions
- Detects most passing students while maintaining good fail detection

### XGBoost
- Very close to Naive Bayes in performance
- Strong F1, AUC, and MCC
- Well-balanced predictions
- Captures complex patterns effectively

### Random Forest
- Extremely high recall (~0.997)
- Detects almost all passing students
- Slight bias toward predicting Pass reduces MCC

### Logistic Regression
- Highest MCC (0.65) and AUC (~0.97)
- Best at separating Pass vs Fail classes
- Very interpretable and stable
- Slightly lower F1 compared to Naive Bayes

### Decision Tree
- Good F1 and accuracy
- Reasonable balance in predictions
- Lower AUC suggests weaker generalization

### kNN
- High recall but struggles to identify Fail cases
- Lowest MCC
- More biased toward majority class predictions

---

## f. Key Insights

- Removing `final_score` prevented data leakage and produced realistic model results.
- Class imbalance strongly influenced prediction behavior.
- F1 Score proved to be the most suitable metric for model comparison.
- Behavioral and academic history features are strong predictors of performance.
- Ensemble models improved recall, while Naive Bayes and Logistic Regression showed strong overall balance.

---

## g. Final Conclusion

Based on the chosen evaluation criterion (**F1 Score**):

### 🏆 Best Model: Naive Bayes

Reasons:
- Highest F1 Score
- Highest accuracy
- Strong balance between precision and recall
- Consistent and stable performance

### Additional Insights
- Logistic Regression achieved the highest MCC and AUC, showing excellent class separation.
- XGBoost delivered highly balanced performance across all metrics.
- Random Forest showed exceptional recall.
- kNN struggled with detecting failing students due to class imbalance.

