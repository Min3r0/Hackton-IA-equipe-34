import pandas as pd
import re

INPUT_FILE = "ai_medical_chatbot_cleaned.parquet"
OUTPUT_FILE = "ai_medical_chatbot_cleaned_v2.parquet"

df = pd.read_parquet(INPUT_FILE)
print("Avant step-5:", df.shape)

# ------------------------------------------------------------------
# 1. DETECTION PII PATIENT
# ------------------------------------------------------------------
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
url_pattern = r'https?://\S+|www\.\S+'
phone_pattern = r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'
name_pattern = r"\bmy name is\s+\w+"

n_patient_email = df["Patient"].str.contains(email_pattern, regex=True, na=False).sum()
n_patient_url = df["Patient"].str.contains(url_pattern, regex=True, na=False).sum()
n_patient_phone = df["Patient"].str.contains(phone_pattern, regex=True, na=False).sum()
n_patient_name = df["Patient"].str.contains(name_pattern, regex=True, case=False, na=False).sum()

print("\n--- PII PATIENT (avant correction) ---")
print(f"Emails: {n_patient_email}")
print(f"URLs: {n_patient_url}")
print(f"Téléphones (probable, faux positifs possibles): {n_patient_phone}")
print(f"Prénoms ('my name is X'): {n_patient_name}")

# ------------------------------------------------------------------
# 2. DETECTION BRUIT DOCTOR (signatures, contacts)
# ------------------------------------------------------------------
signature_pattern = r'\b(Dr\.?|Doctor)\s+[A-Z][a-zA-Z]+(\s+[A-Z][a-zA-Z]+)?'
n_doctor_signature = df["Doctor"].str.contains(signature_pattern, regex=True, na=False).sum()
n_doctor_url = df["Doctor"].str.contains(url_pattern, regex=True, na=False).sum()
n_doctor_email = df["Doctor"].str.contains(email_pattern, regex=True, na=False).sum()

print("\n--- BRUIT DOCTOR (avant correction) ---")
print(f"Signatures 'Dr. Nom': {n_doctor_signature}")
print(f"URLs: {n_doctor_url}")
print(f"Emails: {n_doctor_email}")

# ------------------------------------------------------------------
# 3. QUASI-DOUBLONS (MinHash LSH) - sur l'ensemble complet
# ------------------------------------------------------------------
try:
    from datasketch import MinHash, MinHashLSH

    def get_minhash(text, num_perm=128):
        m = MinHash(num_perm=num_perm)
        for word in str(text).lower().split():
            m.update(word.encode('utf8'))
        return m

    print("\nCalcul des quasi-doublons (peut prendre plusieurs minutes sur 240k lignes)...")
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    minhashes = {}
    for idx, text in enumerate(df["Patient"]):
        mh = get_minhash(text)
        minhashes[idx] = mh
        lsh.insert(idx, mh)

    dup_groups = []
    seen = set()
    for idx in df.index:
        if idx in seen:
            continue
        similar = lsh.query(minhashes[idx])
        if len(similar) > 1:
            dup_groups.append(similar)
            seen.update(similar)

    print(f"Groupes de quasi-doublons trouvés (seuil 0.85): {len(dup_groups)}")
    print(f"Lignes concernées: {sum(len(g) for g in dup_groups)}")

    # On garde seulement le 1er élément de chaque groupe de quasi-doublons
    indices_to_drop = set()
    for g in dup_groups:
        g_sorted = sorted(g)
        indices_to_drop.update(g_sorted[1:])  # garde le premier, drop le reste

    print(f"Lignes à supprimer (quasi-doublons): {len(indices_to_drop)}")

except ImportError:
    print("\n[!] datasketch non installé -> pip install datasketch. Etape quasi-doublons ignorée.")
    indices_to_drop = set()

# ------------------------------------------------------------------
# 4. COHERENCE DESCRIPTION <-> PATIENT
# ------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

coherence_sample = df.sample(n=min(3000, len(df)), random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_desc = vectorizer.fit_transform(coherence_sample["Description"])
tfidf_patient = vectorizer.transform(coherence_sample["Patient"])

sims = [cosine_similarity(tfidf_desc[i], tfidf_patient[i])[0][0] for i in range(len(coherence_sample))]
mean_sim = sum(sims) / len(sims)
low_coherence_pct = sum(1 for s in sims if s < 0.1) / len(sims) * 100

print("\n--- COHERENCE Description <-> Patient ---")
print(f"Similarité moyenne: {mean_sim:.3f}")
print(f"% de lignes à faible cohérence (<0.1): {low_coherence_pct:.1f}%")
print("-> Si similarité moyenne élevée (>0.5): Description redondante, peut être droppée du prompt final")
print("-> Si similarité moyenne faible (<0.3): Description apporte de l'info complémentaire (titre utile)")

# ------------------------------------------------------------------
# 5. CORRECTIONS -> V2
# ------------------------------------------------------------------
print("\n" + "="*60)
print("APPLICATION DES CORRECTIONS")
print("="*60)

def clean_doctor_signature(text):
    text = re.sub(signature_pattern, '', text)
    text = re.sub(url_pattern, '', text)
    text = re.sub(email_pattern, '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def anonymize_patient_name(text):
    return re.sub(name_pattern, "my name is [NAME]", text, flags=re.IGNORECASE)

def remove_web_boilerplate(text):
    text = re.sub(
        r'^hi,?\s*may i answer your health queries.*?query here\.*',
        '', text, flags=re.IGNORECASE
    )
    return text.strip()

df["Doctor"] = df["Doctor"].apply(clean_doctor_signature)
df["Patient"] = df["Patient"].apply(anonymize_patient_name)
df["Patient"] = df["Patient"].apply(remove_web_boilerplate)

# Supprimer les quasi-doublons identifiés
before = len(df)
df = df.drop(index=[i for i in indices_to_drop if i in df.index], errors="ignore")
print(f"Quasi-doublons supprimés: {before - len(df)}")

# Re-filtrer les textes devenus trop courts après nettoyage
df["patient_len"] = df["Patient"].str.len()
df["doctor_len"] = df["Doctor"].str.len()

before = len(df)
df = df[(df["patient_len"] >= 30) & (df["doctor_len"] >= 30)]
print(f"Lignes devenues trop courtes après correction: {before - len(df)}")

df = df.drop(columns=["patient_len", "doctor_len"]).reset_index(drop=True)

print(f"\nAprès step-5: {df.shape}")
print(f"Taux de conservation total (vs dataset brut 256916): {len(df)/256916*100:.1f}%")

# Sauvegarde
df.to_parquet(OUTPUT_FILE)
df.to_csv(OUTPUT_FILE.replace(".parquet", ".csv"), index=False)
print(f"\nFichiers V2 sauvegardés: {OUTPUT_FILE}")

# ------------------------------------------------------------------
# 6. CONTROLE MANUEL (affichage seulement)
# ------------------------------------------------------------------
print("\n" + "="*60)
print("CONTROLE MANUEL - ECHANTILLON ALEATOIRE (25 lignes)")
print("="*60)
sample_check = df.sample(n=25, random_state=123)
for idx, row in sample_check.iterrows():
    print(f"\n{'-'*80}")
    print(f"INDEX: {idx}")
    print(f"DESCRIPTION: {row['Description'][:150]}")
    print(f"PATIENT: {row['Patient'][:250]}")
    print(f"DOCTOR: {row['Doctor'][:250]}")