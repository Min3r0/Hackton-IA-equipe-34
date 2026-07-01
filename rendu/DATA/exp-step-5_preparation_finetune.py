import pandas as pd
import json
from sklearn.model_selection import train_test_split

df = pd.read_parquet("ai_medical_chatbot_cleaned_v2.parquet")

SYSTEM_PROMPT = "You are a knowledgeable and helpful medical assistant. Answer the patient's question clearly and professionally."

def to_chatml(row):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["Patient"]},
            {"role": "assistant", "content": row["Doctor"]}
        ]
    }

records = df.apply(to_chatml, axis=1).tolist()

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# --- Version complète (livrable "dataset préparé") ---
train, temp = train_test_split(records, test_size=0.1, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

save_jsonl(train, "medical_train_full.jsonl")
save_jsonl(val, "medical_val_full.jsonl")
save_jsonl(test, "medical_test_full.jsonl")
print(f"FULL — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# --- Version réduite (utilisable sur petite machine, ex: 20k) ---
df_sample = df.sample(n=20000, random_state=42)
records_sample = df_sample.apply(to_chatml, axis=1).tolist()

train_s, temp_s = train_test_split(records_sample, test_size=0.1, random_state=42)
val_s, test_s = train_test_split(temp_s, test_size=0.5, random_state=42)

save_jsonl(train_s, "medical_train_20k.jsonl")
save_jsonl(val_s, "medical_val_20k.jsonl")
save_jsonl(test_s, "medical_test_20k.jsonl")
print(f"SAMPLE 20k — Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")