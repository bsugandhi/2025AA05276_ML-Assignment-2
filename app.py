import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef


# Page Config
st.set_page_config(
    page_title="Student Performance Classifier",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Performance Classification")
st.markdown("Predict student pass/fail outcomes using academic performance metrics, behavioral factors, and AI usage insights")

# Sidebar - Model Selection
st.sidebar.header("Model Configuration")
model_options = [
    'Logistic Regression',
    'Decision Tree',
    'kNN',
    'Naive Bayes',
    'Random Forest',
    'XGBoost'
]
selected_model_name = st.sidebar.selectbox("Select Classification Model", model_options)

# Load Artifacts
@st.cache_resource
def load_artifacts():
    encoders = joblib.load('model/encoders.pkl')
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    metrics_df = pd.read_csv('model/metrics.csv', encoding='utf-8')

    metrics_df['ML Model Name'] = metrics_df['ML Model Name'].astype(str)


    models = {}
    for name in model_options:
        filename = f"model/{name.replace(' ', '_').lower()}.pkl"
        try:
            models[name] = joblib.load(filename)
        except Exception as e:
            st.error(f"Error loading {name}: {e}")
            
    return models, encoders, scaler, feature_names, metrics_df

try:
    models, encoders, scaler, feature_names, metrics_df = load_artifacts()
    current_model = models[selected_model_name]
except Exception as e:
    st.error(f"Failed to load models/artifacts. Please ensure 'train.py' has been run. Error: {e}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📂 Batch Prediction", "📝 Single Prediction", "📊 Overal Evaluation"])


# --- TAB 1: Batch Prediction ---

with tab1:
    st.subheader("Batch Prediction via CSV")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    with open("test.csv", "rb") as f:
        st.download_button(
            label="📥 Download Sample Test CSV",
            data=f,
            file_name="test.csv",
            mime="text/csv"
        )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", batch_df.head())

            original_batch_df = batch_df.copy()

            # Save target column (if present) for evaluation
            true_target = None
            if 'passed' in batch_df.columns:
                true_target = batch_df['passed'].copy()

            # ---------------- PREPROCESSING ----------------
            if 'student_id' in batch_df.columns:
                batch_df = batch_df.drop(columns=['student_id'])

            # Drop target-like columns for prediction
            cols_to_drop = ['passed', 'final_score', 'performance_category']
            batch_df = batch_df.drop(columns=[c for c in cols_to_drop if c in batch_df.columns])

            # Apply encoders
            for col, le in encoders.items():
                if col in batch_df.columns:
                    batch_df[col] = batch_df[col].astype(str).map(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    batch_df[col] = le.transform(batch_df[col])

            # Ensure all model features exist
            for col in feature_names:
                if col not in batch_df.columns:
                    batch_df[col] = 0

            batch_df = batch_df[feature_names]

            # Scale
            batch_scaled = scaler.transform(batch_df)

            # ---------------- PREDICTION ----------------
            if st.button("Generate Predictions"):
                preds = current_model.predict(batch_scaled)

                original_batch_df['Predicted_Pass'] = preds
                original_batch_df['Predicted_Status'] = original_batch_df['Predicted_Pass'].map({
                    1: 'Passed',
                    0: 'Failed'
                })
                

                st.write("### Prediction Results")
                st.dataframe(original_batch_df.astype(str))

                # Download predictions
                csv = original_batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Predictions",
                    csv,
                    "predictions.csv",
                    "text/csv",
                    key='download-csv'
                )

                # ---------------- EVALUATION (ONLY IF TARGET EXISTS) ----------------
                if true_target is not None:
                    y_true = true_target
                    y_pred = preds

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred)
                    rec = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)

                    # AUC (if model supports probabilities)
                    try:
                        y_prob = current_model.predict_proba(batch_scaled)[:, 1]
                        auc = roc_auc_score(y_true, y_prob)
                    except:
                        auc = None

                    st.markdown("---")


                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("Evaluation Metrics")

                        eval_metrics_df = pd.DataFrame({
                            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score"],
                            "Value": [acc, auc if auc else None, prec, rec, f1]
                        })

                        st.markdown(
                            eval_metrics_df.style
                            .format({"Value": "{:.4f}"})
                            .hide(axis="index")
                            .set_table_attributes('class="compact-table" style="width:80%;height:320px;"')
                            .to_html(classes="compact-table"),
                            unsafe_allow_html=True
                        )

                    with col2:
                        st.subheader("Confusion Matrix")

                        cm = confusion_matrix(y_true, y_pred)

                        fig, ax = plt.subplots(figsize=(2.6, 2.2))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt='d',
                            cmap="Blues",
                            xticklabels=["Failed", "Passed"],
                            yticklabels=["Failed", "Passed"],
                            cbar=False,
                            ax=ax
                        )

                        ax.set_xlabel("Predicted", fontsize=9)
                        ax.set_ylabel("Actual", fontsize=9)
                        ax.tick_params(labelsize=8)

                        plt.tight_layout(pad=0.5)
                        st.pyplot(fig)

                   
                else:
                    st.info("Upload file with 'passed' column to view evaluation metrics and confusion matrix.")

        except Exception as e:
            st.error(f"Error processing file: {e}")


# --- TAB 2: Single Prediction ---
with tab2:
    st.subheader(f"Predict with {selected_model_name}")
    
    with st.form("single_predict_form"):
        col1, col2, col3 = st.columns(3)
        
        
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=20)
            gender = st.selectbox("Gender", encoders['gender'].classes_)
            grade_level = st.selectbox("Grade Level", encoders['grade_level'].classes_)
            study_hours = st.number_input("Study Hours/Day", min_value=0.0, max_value=24.0, value=5.0)
            attendance = st.slider("Attendance %", 0, 100, 85)
            
        with col2:
            uses_ai = st.selectbox("Uses AI Tools?", ["No", "Yes"]) # Map to 0/1
            ai_usage_time = st.number_input("AI Usage Time (mins)", min_value=0, value=30)
            ai_tools = st.selectbox("AI Tools Used", encoders['ai_tools_used'].classes_)
            ai_purpose = st.selectbox("AI Usage Purpose", encoders['ai_usage_purpose'].classes_)
            ai_dependency = st.slider("AI Dependency Score", 0.0, 10.0, 3.0)
            
        with col3:
            last_exam = st.number_input("Last Exam Score", 0, 100, 75)
            assign_avg = st.number_input("Assignment Avg", 0, 100, 80)
            concept_score = st.slider("Concept Understanding", 0.0, 10.0, 7.0)
            sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
            social_media = st.number_input("Social Media Hours", 0.0, 24.0, 2.0)
            

        with st.expander("Additional Details"):
             c1, c2, c3 = st.columns(3)
             with c1:
                 ai_gen_percent = st.slider("AI Generated Content %", 0, 100, 10)
                 ai_prompts = st.number_input("AI Prompts/Week", 0, 500, 20)
             with c2:
                 ai_ethics = st.slider("AI Ethics Score", 0.0, 10.0, 8.0)
                 study_consistency = st.slider("Study Consistency", 0.0, 10.0, 5.0)
             with c3:
                 improvement = st.number_input("Improvement Rate", -100.0, 100.0, 5.0)
                 tutoring = st.number_input("Tutoring Hours", 0.0, 50.0, 0.0)
                 participation = st.number_input("Class Participation", 0.0, 100.0, 80.0)
            
        submitted = st.form_submit_button("Predict Result")
        
        if submitted:
            # Construct Input DataFrame
            input_data = {
                'age': age,
                'gender': gender,
                'grade_level': grade_level,
                'study_hours_per_day': study_hours,
                'uses_ai': 1 if uses_ai == "Yes" else 0,
                'ai_usage_time_minutes': ai_usage_time,
                'ai_tools_used': ai_tools,
                'ai_usage_purpose': ai_purpose,
                'ai_dependency_score': ai_dependency,
                'ai_generated_content_percentage': ai_gen_percent,
                'ai_prompts_per_week': ai_prompts,
                'ai_ethics_score': ai_ethics,
                'last_exam_score': last_exam,
                'assignment_scores_avg': assign_avg,
                'attendance_percentage': attendance,
                'concept_understanding_score': concept_score,
                'study_consistency_index': study_consistency,
                'improvement_rate': improvement,
                'sleep_hours': sleep_hours,
                'social_media_hours': social_media,
                'tutoring_hours': tutoring,
                'class_participation_score': participation
            }
            
            
            df_input = pd.DataFrame([input_data])
            
            # Apply Encoders
            for col, le in encoders.items():
                if col in df_input.columns:
                    df_input[col] = le.transform(df_input[col].astype(str))
            
            # Align columns
            df_input = df_input[feature_names]
            
            # Scale
            df_scaled = scaler.transform(df_input)
            df_scaled = pd.DataFrame(df_scaled, columns=feature_names)
            
            # Predict
            prediction = current_model.predict(df_scaled)[0]
            prob = current_model.predict_proba(df_scaled)[0][1] if hasattr(current_model, "predict_proba") else prediction
            
            st.markdown("---")
            if prediction == 1:
                st.success(f"### Prediction: PASSED ✅")
            else:
                st.error(f"### Prediction: FAILED ❌")
            
            st.write(f"**Confidence (Pass Probability):** {prob:.2%}")

# --- TAB 3: Model Evaluation ---
with tab3:
    st.subheader("Model Performance Comparison")
    
    df_display = metrics_df.reset_index(drop=True).copy()
    best_idx = df_display['F1 Score'].idxmax()
    best_model_row = df_display.loc[best_idx]
    st.info(f"🏆 Best Performing Model: **{best_model_row['ML Model Name']}** with F1 Score of **{best_model_row['F1 Score']:.4f}**")

    styled = (
        df_display.style
        .format("{:.4f}", subset=['Accuracy','AUC','Precision','Recall','F1 Score','MCC Score'])
        .apply(
            lambda x: ['background-color: #d4edda' if x.name == best_idx else '' for _ in x],
            axis=1
        )
        .set_table_styles([
            {'selector': 'th.row_heading', 'props': [('display', 'none')]},
            {'selector': 'th.blank', 'props': [('display', 'none')]}
        ])
    )

    st.markdown(
        styled.to_html(index=False),
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.subheader("Confusion Matrices")

    cols = st.columns(2)

    for i, model_name in enumerate(model_options):
        img_path = f"model/cm_images/{model_name.replace(' ', '_').lower()}.png"

        with cols[i % 2]:
            if os.path.exists(img_path):
               st.image(img_path, caption=model_name, use_column_width=True)
            else:
                st.warning(f"Image not found for {model_name}")
        
    st.markdown("---")
    st.write("### Detailed Observations")
    
    obs_data = {
        "Logistic Regression": (
            "Strong and reliable performer with the highest MCC (0.65) and AUC (~0.97), showing the best ability to separate Pass and Fail cases. "
            "Confusion matrix indicates balanced predictions, though it misses some passing students. Overall, the most interpretable and stable model."
        ),

        "Decision Tree": (
            "Improved performance with good Accuracy (~0.92) and F1 (~0.95). Detects both Pass and Fail cases better than before, "
            "but lower AUC suggests weaker generalization compared to Logistic Regression and ensemble models."
        ),

        "kNN": (
            "Very high recall (~0.99) but still biased toward predicting Pass. Confusion matrix shows many Fail students classified as Pass, "
            "which explains the lower MCC (~0.34) despite a good F1 score."
        ),

        "Naive Bayes": (
            "One of the best overall models with high Accuracy (~0.93), strong F1 (~0.96), and good MCC (~0.62). "
            "Confusion matrix shows very few missed Pass cases and reasonable Fail detection, giving stable performance."
        ),

        "Random Forest": (
            "Extremely high recall (~0.997), meaning it detects almost all passing students. However, it struggles more with Fail detection, "
            "which lowers MCC (~0.52). Strong model but slightly biased toward the majority class."
        ),

        "XGBoost": (
            "Well-balanced performance across all metrics with high F1 (~0.96), strong AUC (~0.96), and good MCC (~0.62). "
            "Confusion matrix shows better Fail detection than Random Forest, making it one of the most balanced ensemble models."
        )
    }


    for model, obs in obs_data.items():
        st.write(f"- **{model}**: {obs}")
