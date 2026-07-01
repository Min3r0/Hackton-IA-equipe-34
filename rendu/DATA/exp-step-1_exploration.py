import pandas as pd

df = pd.read_parquet("ai_medical_chatbot.parquet")

print("Shape:", df.shape)
print("\nColonnes:", df.columns.tolist())
print("\nAperçu:")
print(df.head())
print("\nTypes:")
print(df.info())
print("\nValeurs manquantes:")
print(df.isnull().sum())
print("\nDoublons exacts:", df.duplicated().sum())