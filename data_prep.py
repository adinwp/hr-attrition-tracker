import pandas as pd
import numpy as np

# 1. Load dataset IBM HR (pastikan nama file CSV sesuai dengan yang lo download)
# Asumsi nama file bawaan Kaggle: 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING: Bikin Data Makin Relatable!
# ---------------------------------------------------------
np.random.seed(42) # Biar datanya konsisten kalau script di-run berkali-kali

# A. Eskalasi Commute Distance (KM)
# Mengubah jarak bawaan jadi lebih ekstrem buat simulasi karyawan yang habis waktu di jalan
kondisi_jarak = [
    df['DistanceFromHome'] <= 10,
    (df['DistanceFromHome'] > 10) & (df['DistanceFromHome'] <= 20),
    df['DistanceFromHome'] > 20
]
pilihan_jarak = [
    np.random.randint(2, 15, size=len(df)),    # Jarak dekat-normal
    np.random.randint(15, 40, size=len(df)),   # Sedang (Mulai macet)
    np.random.randint(80, 105, size=len(df))   # Ekstrem (Biar kerasa capeknya commute tiap hari!)
]
df['Real_Commute_Distance_KM'] = np.select(kondisi_jarak, pilihan_jarak, default=15)

# B. Number of Dependents (Jumlah Tanggungan Keluarga)
# Krusial untuk HR karena menyangkut kebutuhan finansial vs. waktu untuk kumpul keluarga
# Distribusi logis: Mayoritas punya 1 sampai 3 tanggungan
df['Number_of_Dependents'] = np.random.choice(
    [0, 1, 2, 3, 4], 
    size=len(df), 
    p=[0.2, 0.25, 0.35, 0.15, 0.05]
)

# C. Toxic Manager Score (Skala 1-10)
# Logika bisnis: Kalau EnvironmentSatisfaction (bawaan IBM) rendah, kemungkinan manajernya problematik
df['Toxic_Manager_Score'] = np.where(
    df['EnvironmentSatisfaction'] <= 2,
    np.random.randint(7, 11, size=len(df)), # Toxic parah (skor merah 7-10)
    np.random.randint(1, 6, size=len(df))   # Normal/Aman (skor hijau 1-5)
)

# D. Fitur Baru: Burnout Risk Flag
# Ini ngasih insight instan ke HR. Gabungan dari lembur, jarak jauh, dan manajer toxic
df['Burnout_Risk'] = np.where(
    (df['OverTime'] == 'Yes') & 
    (df['Real_Commute_Distance_KM'] >= 80) & 
    (df['Toxic_Manager_Score'] >= 7),
    'High Risk', 'Normal'
)

# ---------------------------------------------------------
# 3. PREVIEW & EXPORT HASILNYA
# ---------------------------------------------------------
kolom_penting = [
    'Real_Commute_Distance_KM', 'Number_of_Dependents', 
    'Toxic_Manager_Score', 'Burnout_Risk', 'Attrition'
]

print("Preview Data Modifikasi:")
print(df[kolom_penting].head(10))

# Save jadi dataset baru buat dipakai di tahap Machine Learning & Streamlit
df.to_csv('HR_Analytics_Engineered_Data.csv', index=False)
print("\n[SUCCESS] Dataset baru berhasil disimpan: HR_Analytics_Engineered_Data.csv")