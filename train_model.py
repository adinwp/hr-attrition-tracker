import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Loading data...")
# 1. Load dataset hasil modifikasi sebelumnya
df = pd.read_csv('HR_Analytics_Engineered_Data.csv')

# 2. DATA CLEANING & PREPROCESSING
# Buang kolom yang nggak ada predictive value-nya (konstan atau sekadar ID)
kolom_sampah = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df = df.drop(columns=[col for col in kolom_sampah if col in df.columns])

# Pisahkan Target (Attrition) dan Features (X)
# Ubah Attrition jadi angka: Yes = 1, No = 0
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
X = df.drop(columns=['Attrition'])

# One-Hot Encoding untuk kolom kategorikal (seperti Department, Gender, dll)
X = pd.get_dummies(X, drop_first=True)

# 3. TRAIN-TEST SPLIT (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. TACKLING CLASS IMBALANCE (The Senior Touch)
# Hitung rasio antara kelas mayoritas (stay) dan minoritas (resign)
ratio = float(y_train.value_counts()[0]) / y_train.value_counts()[1]
print(f"Rasio Imbalance (Stay vs Resign): {ratio:.2f}")

# 5. BUILD & TRAIN XGBOOST MODEL
print("\nTraining XGBoost Model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=ratio, # Ini kunci biar model peka sama yang mau resign!
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 6. EVALUASI MODEL
y_pred = model.predict(X_test)

print("\n--- MODEL EVALUATION ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
# Ingat: Fokus kita ada di baris '1' (Karyawan Resign), khususnya kolom 'Recall'
print(classification_report(y_test, y_pred))

# 7. SAVE MODEL UNTUK STREAMLIT
joblib.dump(model, 'xgb_attrition_model.pkl')
# Simpan juga nama kolom-kolom X_train biar konsisten saat deploy di Streamlit
joblib.dump(list(X_train.columns), 'model_features.pkl')

print("\n[SUCCESS] Model XGBoost dan daftar fitur berhasil disimpan!")