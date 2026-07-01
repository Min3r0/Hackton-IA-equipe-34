from datasets import load_dataset

dataset = load_dataset("ruslanmv/ai-medical-chatbot")

# Sauvegarder en local (format Arrow, rapide à recharger)
dataset.save_to_disk("./ai-medical-chatbot-local")

# Ou exporter en CSV / JSON / Parquet pour le nettoyage
dataset["train"].to_csv("ai_medical_chatbot.csv")
dataset["train"].to_json("ai_medical_chatbot.json")
dataset["train"].to_parquet("ai_medical_chatbot.parquet")