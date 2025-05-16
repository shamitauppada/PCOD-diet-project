# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("synthetic_pcod_fitness_dataset.csv")
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in ['Diet_Preference', 'Symptoms', 'Workout_Type', 'Recommended_Diet', 'Recommended_Workout', 'Cycle_Regularity']:
    df[col] = le.fit_transform(df[col])

# Clustering based on activity and BMI
X_cluster = df[['BMI', 'Daily_Steps', 'Active_Minutes']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Random Forest Models
X_rf = df[['BMI', 'BMR', 'Stress_Level(1-10)', 'Sleep_Hours', 'PCOS_Diagnosis']]
y_rf_diet = df['Recommended_Diet']
y_rf_workout = df['Recommended_Workout']
rf_diet = RandomForestClassifier(random_state=42).fit(X_rf, y_rf_diet)
rf_workout = RandomForestClassifier(random_state=42).fit(X_rf, y_rf_workout)

# Linear Regression Model
X_cal = df[['Age', 'Weight(kg)', 'Height(cm)', 'BMR', 'Daily_Steps', 'Active_Minutes']]
y_cal = df['Calorie_Target']
reg_calorie = LinearRegression().fit(X_cal, y_cal)

# Streamlit UI
st.title("PCOD Fitness & Wellness Recommender")
st.write("### Enter New User Details")

age = st.number_input("Age", min_value=10, max_value=80, value=25)
weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=60.0)
height = st.number_input("Height (cm)", min_value=120.0, max_value=220.0, value=165.0)
bmr = st.number_input("Basal Metabolic Rate (BMR)", min_value=800, max_value=3000, value=1500)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=20000, value=6000)
active_minutes = st.number_input("Active Minutes per Day", min_value=0, max_value=300, value=45)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0)
pcos_val = st.radio("PCOS Diagnosis", ["No", "Yes"]) == "Yes"

if st.button("Predict Wellness Plan"):
    bmi_val = weight / ((height / 100) ** 2)
    calorie_input = np.array([[age, weight, height, bmr, daily_steps, active_minutes]])
    pred_calorie = reg_calorie.predict(calorie_input)[0]

    input_rf = np.array([[bmi_val, bmr, stress_level, sleep_hours, int(pcos_val)]])
    pred_diet = rf_diet.predict(input_rf)[0]
    pred_workout = rf_workout.predict(input_rf)[0]

    diet_decoded = le.inverse_transform([pred_diet])[0]
    workout_decoded = le.inverse_transform([pred_workout])[0]

    user_cluster_input = scaler.transform([[bmi_val, daily_steps, active_minutes]])
    cluster_group = kmeans.predict(user_cluster_input)[0]

    st.success(f"Predicted Daily Calorie Requirement: {int(pred_calorie)} kcal")
    st.success(f"Recommended Diet: {diet_decoded}")
    st.success(f"Recommended Workout: {workout_decoded}")
    st.info(f"User belongs to Health Cluster Group: {cluster_group}")

    # Visualize Clusters
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x='Daily_Steps', y='BMI', hue='Cluster', palette='Set1', ax=ax)
    ax.scatter(daily_steps, bmi_val, color='black', s=100, label='New User')
    ax.set_title('Health Clusters (including new user)')
    ax.legend()
    st.pyplot(fig)
