# 📚 Import Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 Load dataset
df = pd.read_csv("synthetic_pcod_fitness_dataset.csv")

# 📊 Explore data
print(df.head())
print(df.columns)

# 📋 Preprocessing
df.dropna(inplace=True)
le = LabelEncoder()

# Encode categorical columns
for col in ['Diet_Preference', 'Symptoms', 'Workout_Type', 'Recommended_Diet', 'Recommended_Workout', 'Cycle_Regularity']:
    df[col] = le.fit_transform(df[col])

# Clustering based on activity and BMI
X_cluster = df[['BMI', 'Daily_Steps', 'Active_Minutes']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 📊 Visualize Clusters
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Daily_Steps', y='BMI', hue='Cluster', palette='Set1')
plt.title('Health Clusters based on Activity and BMI')
plt.show()

# Random Forest for Diet Recommendation
X_rf = df[['BMI', 'BMR', 'Stress_Level(1-10)', 'Sleep_Hours', 'PCOS_Diagnosis']]
y_rf_diet = df['Recommended_Diet']
y_rf_workout = df['Recommended_Workout']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_rf, y_rf_diet, test_size=0.2, random_state=42)
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X_rf, y_rf_workout, test_size=0.2, random_state=42)

rf_diet = RandomForestClassifier(random_state=42)
rf_diet.fit(X_train_d, y_train_d)

rf_workout = RandomForestClassifier(random_state=42)
rf_workout.fit(X_train_w, y_train_w)

# Linear Regression for Calorie Target
X_cal = df[['Age', 'Weight(kg)', 'Height(cm)', 'BMR', 'Daily_Steps', 'Active_Minutes']]
y_cal = df['Calorie_Target']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cal, y_cal, test_size=0.2, random_state=42)

reg_calorie = LinearRegression()
reg_calorie.fit(X_train_c, y_train_c)

# ✅ Test a new user prediction
print("\n📥 Enter New User Details:")
age = 25
weight = 60
height = 165
bmr = 1500
daily_steps = 6000
active_minutes = 45
stress_level = 5
sleep_hours = 7.0
pcos_val = 0  # 1 for Yes, 0 for No

# Calculate BMI
bmi_val = weight / ((height/100)**2)

# Predict Calorie Requirement
calorie_input = np.array([[age, weight, height, bmr, daily_steps, active_minutes]])
pred_calorie = reg_calorie.predict(calorie_input)[0]
print(f"\n🟢 Predicted Daily Calorie Requirement: {int(pred_calorie)} kcal")

# Predict Diet & Workout Recommendation
input_rf = np.array([[bmi_val, bmr, stress_level, sleep_hours, pcos_val]])
pred_diet = rf_diet.predict(input_rf)[0]
pred_workout = rf_workout.predict(input_rf)[0]

# Decode Label Encoded values
diet_decoded = le.inverse_transform([pred_diet])[0]
workout_decoded = le.inverse_transform([pred_workout])[0]

print(f"🟢 Recommended Diet Type: {diet_decoded}")
print(f"🟢 Recommended Workout Type: {workout_decoded}")

# Predict Cluster
user_cluster_input = scaler.transform([[bmi_val, daily_steps, active_minutes]])
cluster_group = kmeans.predict(user_cluster_input)[0]
print(f"🟢 User belongs to Health Cluster Group: {cluster_group}")

# 📈 Visual Insight — add new user point to plot
plt.figure(figsize=(8,5))
sns.scatterplot(data=df, x='Daily_Steps', y='BMI', hue='Cluster', palette='Set1')
plt.scatter(daily_steps, bmi_val, color='black', s=120, label='New User')
plt.title('Health Clusters (including new user)')
plt.legend()
plt.show()
