import pandas as pd

df = pd.read_parquet("ai_medical_chatbot.parquet")

# 1. Vérifier la redondance Description vs Patient
print(df[["Description", "Patient"]].sample(5).to_string())

# 2. Longueurs des textes (en caractères et mots)
df["desc_len"] = df["Description"].str.len()
df["patient_len"] = df["Patient"].str.len()
df["doctor_len"] = df["Doctor"].str.len()

print(df[["desc_len", "patient_len", "doctor_len"]].describe())

# 3. Quasi-doublons (Patient très similaire)
print("Doublons sur Patient seul:", df.duplicated(subset=["Patient"]).sum())
print("Doublons sur Doctor seul:", df.duplicated(subset=["Doctor"]).sum())

# 4. Textes suspects / trop courts
print("Patient < 20 caractères:", (df["patient_len"] < 20).sum())
print("Doctor < 20 caractères:", (df["doctor_len"] < 20).sum())

# 5. Présence de HTML ou caractères bizarres
import re
html_count = df["Doctor"].str.contains(r'<[^>]+>', regex=True).sum()
print("Réponses avec balises HTML:", html_count)

# 6. Réponses génériques/vides de sens (ex: juste "Hi.", "Hello.")
print(df[df["doctor_len"] < 15]["Doctor"].unique()[:20])