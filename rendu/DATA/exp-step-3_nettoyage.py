import pandas as pd
import re

df = pd.read_parquet("ai_medical_chatbot.parquet")
print("Avant nettoyage:", df.shape)

def clean_text(text):
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('\xa0', ' ')
    return text

df["Description"] = df["Description"].apply(clean_text)
df["Patient"] = df["Patient"].apply(clean_text)
df["Doctor"] = df["Doctor"].apply(clean_text)

before = len(df)
df = df.drop_duplicates(subset=["Patient", "Doctor"])
print(f"Doublons supprimés: {before - len(df)}")

df["patient_len"] = df["Patient"].str.len()
df["doctor_len"] = df["Doctor"].str.len()

MIN_PATIENT_LEN = 30
MIN_DOCTOR_LEN = 30

before = len(df)
df = df[(df["patient_len"] >= MIN_PATIENT_LEN) & (df["doctor_len"] >= MIN_DOCTOR_LEN)]
print(f"Lignes trop courtes supprimées: {before - len(df)}")

patient_cap = df["patient_len"].quantile(0.99)
doctor_cap = df["doctor_len"].quantile(0.99)
print(f"Seuils max (99e percentile) — Patient: {patient_cap:.0f}, Doctor: {doctor_cap:.0f}")

before = len(df)
df = df[(df["patient_len"] <= patient_cap) & (df["doctor_len"] <= doctor_cap)]
print(f"Outliers longs supprimés: {before - len(df)}")

noise_patterns = [
    r'^hi\.?$', r'^hello\.?$', r'^ok\.?$', r'^yes!?$', r'^no\.?$',
    r'^test', r'^reply$', r'^thanks doctor$', r'^please help$'
]
noise_regex = '|'.join(noise_patterns)

before = len(df)
df = df[~df["Doctor"].str.lower().str.match(noise_regex, na=False)]
print(f"Réponses bruit supprimées: {before - len(df)}")

df = df.drop(columns=["patient_len", "doctor_len"])
df = df.reset_index(drop=True)

print("\nAprès nettoyage:", df.shape)
print(f"Taux de conservation: {len(df)/256916*100:.1f}%")

# Sauvegarde du dataset nettoyé (nouveau fichier, on garde l'original intact)
df.to_parquet("ai_medical_chatbot_cleaned.parquet")
df.to_csv("ai_medical_chatbot_cleaned.csv", index=False)
df.to_json("ai_medical_chatbot_cleaned.json", orient="records", lines=True)

print("Fichiers sauvegardés ✅")