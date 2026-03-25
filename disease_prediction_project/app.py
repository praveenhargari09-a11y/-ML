import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("AI Disease Prediction")
st.sidebar.write("""
This app predicts possible diseases based on symptoms.

Features:
• Random Forest & Logistic Regression  
• Confidence percentage  
• Disease description  
• Precautions  
• Probability chart  
• Symptom severity score  
• Doctor recommendation
""")

# -----------------------------
# Doctor Recommendation
# -----------------------------
doctor_specialist = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "General Physician",
    "Peptic ulcer disease": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic Doctor",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "General Physician",
    "Chicken pox": "Dermatologist",
    "Dengue": "General Physician",
    "Typhoid": "General Physician",
    "Hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Heart attack": "Cardiologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Arthritis": "Rheumatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist"
}

# -----------------------------
# Load Datasets
# -----------------------------
data = pd.read_csv("Training.csv")
description = pd.read_csv("symptom_description.csv")
precautions = pd.read_csv("symptom_precaution.csv")
severity = pd.read_csv("Symptom-severity.csv")

# -----------------------------
# Prepare Data
# -----------------------------
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Fix for Logistic Regression
X = X.fillna(0)

symptoms = list(X.columns)

# -----------------------------
# Train Models
# -----------------------------
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X, y)

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Random Forest", "Logistic Regression"]
)

model = rf_model if model_choice == "Random Forest" else lr_model

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Disease Prediction System")
st.write("Select the symptoms you are experiencing:")

# -----------------------------
# Symptom Selection
# -----------------------------
selected_symptoms = st.multiselect(
    "Choose symptoms",
    symptoms
)

if selected_symptoms:
    st.write("Selected symptoms:", selected_symptoms)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")

    else:
        # Create input vector
        input_vector = [0] * len(symptoms)

        for symptom in selected_symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1

        # Predict Disease
        prediction = model.predict([input_vector])[0]
        st.success("Predicted Disease: " + prediction)

        # Confidence
        probabilities = model.predict_proba([input_vector])[0]
        confidence = max(probabilities) * 100
        st.info(f"Confidence Level: {confidence:.2f}%")

        # Probability Chart
        prob_df = pd.DataFrame({
            "Disease": model.classes_,
            "Probability": probabilities
        })

        top5 = prob_df.sort_values(
            by="Probability",
            ascending=False
        ).head(5)

        st.subheader("Top 5 Possible Diseases")
        st.bar_chart(top5.set_index("Disease"))

        # Disease Description
        desc = description[
            description['Disease'] == prediction
        ]['Description']

        if not desc.empty:
            st.subheader("Disease Description")
            st.write(desc.values[0])

        # Precautions
        prec = precautions[
            precautions['Disease'] == prediction
        ]

        if not prec.empty:
            st.subheader("Recommended Precautions")

            precautions_list = prec.iloc[0, 1:].dropna().values

            for p in precautions_list:
                st.write("✔", p)

        # Doctor Recommendation
        doctor = doctor_specialist.get(
            prediction,
            "General Physician"
        )

        st.subheader("Recommended Doctor")
        st.write("You should consult a:", doctor)

        # Symptom Severity Score
        severity_score = 0

        for symptom in selected_symptoms:
            sev = severity[
                severity['Symptom'] == symptom
            ]['weight']

            if not sev.empty:
                severity_score += sev.values[0]

        st.subheader("Symptom Severity Score")

        if severity_score < 5:
            st.success("Mild symptoms. Monitor your health.")

        elif severity_score < 15:
            st.warning("Moderate symptoms. Consider consulting a doctor.")

        else:
            st.error("Severe symptoms. Seek medical attention.")

# -----------------------------
# Footer
# -----------------------------
st.caption("AI-based Disease Prediction using Machine Learning")