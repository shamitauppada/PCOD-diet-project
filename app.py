import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Inject CSS for custom theme: pink/white background and dark contrasting text
st.markdown("""
    <style>
    .stApp {
        background-color: #ffe6f0;
        color: #1f1f1f;
    }
    h1, h2, h3, h4, h5, h6, p, label, div {
        color: #1f1f1f !important;
    }
    .stNumberInput input,
    .stTextInput input,
    .stSlider div,
    .stRadio label {
        color: #1f1f1f !important;
    }
    .stButton > button {
        background-color: #ff8fa3;
        color: #1f1f1f;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #ff6f91;
    }
    .stSuccess {
        background-color: #fff0f5 !important;
        color: #1f1f1f !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load and preprocess data
df = pd.read_csv("synthetic_pcod_fitness_dataset.csv")
df.dropna(inplace=True)

# Encode labels manually for diet and workout
diet_le = LabelEncoder()
workout_le = LabelEncoder()

df['Recommended_Diet_Label'] = diet_le.fit_transform(df['Recommended_Diet'])
df['Recommended_Workout_Label'] = workout_le.fit_transform(df['Recommended_Workout'])

# Train models
X_rf = df[['BMI', 'BMR', 'Stress_Level(1-10)', 'Sleep_Hours', 'PCOS_Diagnosis']]
y_diet = df['Recommended_Diet_Label']
y_workout = df['Recommended_Workout_Label']

rf_diet = RandomForestClassifier(random_state=42).fit(X_rf, y_diet)
rf_workout = RandomForestClassifier(random_state=42).fit(X_rf, y_workout)

# Train calorie model
X_cal = df[['Age', 'Weight(kg)', 'Height(cm)', 'BMR', 'Daily_Steps', 'Active_Minutes']]
y_cal = df['Calorie_Target']
calorie_model = LinearRegression().fit(X_cal, y_cal)

# UI
st.title("🌸 PCOD Fitness & Wellness Recommender")

st.header("📝 Enter Your Information")

age = st.number_input("Age", 10, 80, 25)
weight = st.number_input("Weight (kg)", 30.0, 200.0, 60.0)
height = st.number_input("Height (cm)", 120.0, 220.0, 165.0)
bmr = st.number_input("Basal Metabolic Rate (BMR)", 800, 3000, 1500)
daily_steps = st.number_input("Daily Steps", 0, 20000, 6000)
active_minutes = st.number_input("Active Minutes per Day", 0, 300, 45)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
pcos_diagnosis = st.radio("PCOS Diagnosis", ["No", "Yes"]) == "Yes"

# Prediction button
if st.button("💡 Get Personalized Recommendations"):
    bmi = weight / ((height / 100) ** 2)

    # Calorie prediction
    cal_input = np.array([[age, weight, height, bmr, daily_steps, active_minutes]])
    cal_output = int(calorie_model.predict(cal_input)[0])

    # Diet & workout prediction
    model_input = np.array([[bmi, bmr, stress_level, sleep_hours, int(pcos_diagnosis)]])
    predicted_diet_label = rf_diet.predict(model_input)[0]
    predicted_workout_label = rf_workout.predict(model_input)[0]

    predicted_diet = diet_le.inverse_transform([predicted_diet_label])[0]
    predicted_workout = workout_le.inverse_transform([predicted_workout_label])[0]

    # Results
    st.success(f"Recommended Daily Calorie Intake: {cal_output} kcal")
    st.success(f"Recommended Diet: {predicted_diet}")
    st.success(f"Recommended Workout: {predicted_workout}")
