from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import json
import os
from unsloth.chat_templates import train_on_responses_only

# ------------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------------
DATA_DIR = "../DATA/"
DATASET_FILE = DATA_DIR + "finance_dataset_final_v2.json"

max_seq_length = 1024
dtype = None  # auto-detection (float16/bfloat16)

FINANCE_SYSTEM_PROMPT = (
    "You are a knowledgeable financial assistant. Answer the user's question "
    "clearly, accurately, and professionally."
)

# ------------------------------------------------------------------
# 1. Chargement du modele en 4-bit (QLoRA)
# ------------------------------------------------------------------
print("Chargement du modele de base (Phi-3.5-mini-instruct, 4-bit)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3.5-mini-instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=True,
)

# ------------------------------------------------------------------
# 2. Configuration LoRA
# ------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,   # 0 = patching rapide Unsloth (0.1 desactive l'optimisation)
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

# ------------------------------------------------------------------
# 3. Chargement du dataset finance DECONTAMINE (format Alpaca)
#    -> chargement manuel en JSON + Dataset.from_list au lieu de
#       load_dataset("json", ...), qui peut se bloquer sur Windows.
# ------------------------------------------------------------------
if not os.path.exists(DATASET_FILE):
    raise FileNotFoundError(
        f"Fichier introuvable: {DATASET_FILE}\n"
        f"Verifie DATA_DIR et que prod-step-1_finance_validation.py a bien ete execute."
    )

print(f"Chargement du dataset: {DATASET_FILE}")
with open(DATASET_FILE, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"Nombre d'exemples charges: {len(raw_data)}")

full_dataset = Dataset.from_list(raw_data)

# Split train/val (90/10)
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ------------------------------------------------------------------
# 4. Formatage ChatML
# ------------------------------------------------------------------
def formatting_func(example):
    user_content = example["instruction"]
    if example.get("input"):
        user_content += "\n\n" + example["input"]

    messages = [
        {"role": "system", "content": FINANCE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

train_dataset = train_dataset.map(formatting_func)
val_dataset = val_dataset.map(formatting_func)

print("Exemple formate:")
print(train_dataset[0]["text"][:500])

# ------------------------------------------------------------------
# 5. Configuration de l'entrainement
# ------------------------------------------------------------------
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=42,
    output_dir="outputs_finance_lora_v2",
    save_strategy="epoch",
    save_only_model=True,
    eval_strategy="steps",
    eval_steps=50,
    report_to="none",
)

# ------------------------------------------------------------------
# 6. Entrainement
# ------------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
    packing=False,
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|user|>\n",
    response_part="<|assistant|>\n",
)

print("\nDebut de l'entrainement...\n")
trainer.train()

# ------------------------------------------------------------------
# 7. Sauvegarde du modele (nouvel adapter propre)
# ------------------------------------------------------------------
OUTPUT_DIR = "phi3_financial_v2_clean"
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nFine-tuning termine, adapter sauvegarde dans {OUTPUT_DIR}/")
print("Prochaine etape : relancer le test anti-backdoor sur ce nouvel adapter.")