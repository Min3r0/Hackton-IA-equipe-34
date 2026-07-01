# DATA — Expert Données

Ce dossier contient l'ensemble du travail de préparation, nettoyage et validation des données pour les deux missions du projet :

- **Mission Production** : validation des données d'entrée pour Phi-3.5-Financial
- **Mission Expérimentale** : préparation d'un dataset médical pour fine-tuning LoRA

---

## 📁 Structure des fichiers

### Scripts (à exécuter dans l'ordre)

| # | Script | Rôle |
|---|--------|------|
| 0 | `exp-step-0_dl_dataset.py` | Téléchargement du dataset médical source (Hugging Face) |
| 1 | `exp-step-1_exploration.py` | Exploration initiale (shape, colonnes, doublons, valeurs manquantes) |
| 2 | `exp-step-2_analyse.py` | Analyse approfondie (longueurs de texte, HTML, réponses bruit) |
| 3 | `exp-step-3_nettoyage.py` | Nettoyage de base (doublons, textes courts, outliers) |
| 4 | `exp-step-4_check_correction.py` | Vérifications qualité + corrections (PII, quasi-doublons, cohérence) |
| 5 | `exp-step-5_preparation_finetune.py` | Formatage ChatML + split train/val/test |
| — | `prod-step-1_finance_validation.py` | Validation et nettoyage des datasets financiers (backdoor, hors-sujet, doublons) |

**Chemins relatifs** : les scripts médicaux (`exp-step-*`) attendent d'être lancés depuis `rendu/DATA/`. Le script finance (`prod-step-1_*`) utilise `DATA_DIR = "../../datasets/"` pour pointer vers le dossier `datasets/` à la racine du projet — adapte ce chemin si tu déplaces les fichiers.

### Datasets produits

| Fichier | Contenu | Lignes |
|---|---|---|
| `ai_medical_chatbot.csv/json/parquet` | Dataset médical brut téléchargé | 256 916 |
| `ai_medical_chatbot_cleaned.*` | Après nettoyage de base (step 3) | 241 524 |
| `ai_medical_chatbot_cleaned_v2.*` | Après vérifications qualité (step 4) — **version finale** | 240 637 |
| `medical_train_full.jsonl` / `medical_train_20k.jsonl` | Jeux d'entraînement (ChatML) | 216 573 / 18 000 |
| `medical_val_full.jsonl` / `medical_val_20k.jsonl` | Jeux de validation | 12 032 / 1 000 |
| `medical_test_full.jsonl` / `medical_test_20k.jsonl` | Jeux de test | 12 032 / 1 000 |
| `finance_dataset_final_v2.json` | Dataset finance principal nettoyé | 2 419 |
| `test_dataset_16000_finance_only.json` | Dataset finance test, filtré strictement finance | 4 861 |
| `test_dataset_16000_v2.json` | Dataset finance test, décontaminé (tous sujets) | 12 324 |

### Livrable

- `Rapport_Qualite_Donnees.docx` — rapport complet couvrant les deux missions (méthodologie, statistiques, découverte de l'empoisonnement, limites, recommandations)

---

## 🧪 Mission Expérimentale — Dataset médical

**Source** : [`ruslanmv/ai-medical-chatbot`](https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot) (Hugging Face) — conversations patient/médecin en anglais. Colonnes : `Description`, `Patient`, `Doctor`.

### Pipeline

1. **Téléchargement** (`exp-step-0`) : via `datasets.load_dataset`, export en CSV/JSON/Parquet
2. **Exploration** (`exp-step-1`) : 256 916 lignes, 0 valeur manquante, 10 378 doublons exacts identifiés
3. **Analyse** (`exp-step-2`) : distribution des longueurs, détection HTML résiduel, réponses "bruit" (`"Hi."`, `"Ok"`...)
4. **Nettoyage de base** (`exp-step-3`) :
   - Suppression doublons : 10 390 lignes
   - Textes trop courts supprimés : 195 lignes
   - Outliers de longueur (99e percentile) supprimés : 4 802 lignes
   - Réponses bruit supprimées : 5 lignes
   - **Résultat : 241 524 lignes (94,0 % conservé)**
5. **Vérifications qualité** (`exp-step-4`) :
   - **PII** : 80 emails, 72 téléphones (probable), 4 197 prénoms patients anonymisés (`"my name is [NAME]"`), 49 951 signatures médecin supprimées (`"Dr. Nom"`), 2 823 URLs/emails côté médecin supprimés
   - **Quasi-doublons** (MinHash LSH, seuil 0,85) : 694 groupes détectés, 868 lignes supprimées
   - **Cohérence Description ↔ Patient** (similarité cosinus TF-IDF) : 0,322 en moyenne → colonne conservée (info complémentaire, ni redondante ni indépendante)
   - Contrôle manuel sur échantillon de 25 lignes : RAS après corrections
   - **Résultat final : 240 637 lignes (93,7 % du dataset brut conservées)**
6. **Préparation fine-tuning** (`exp-step-5`) :
   - Format **ChatML** (compatible Phi-3.5), prompt système dédié à l'assistance médicale
   - Split 90/5/5 (train/val/test)
   - Deux volumes : **complet** (216 573 / 12 032 / 12 032) et **échantillon 20k** (18 000 / 1 000 / 1 000, pour contrainte VRAM ≤ 8 Go)

### Limites connues

- Contenu occasionnellement sensible dans les données source (cas extrêmes réels) — cadré par les réponses médecins, mais à garder en tête lors de l'évaluation du modèle
- Détection PII heuristique (regex) : ne garantit pas 100 % d'élimination
- Dataset exclusivement en anglais, origine majoritairement plateformes de téléconsultation

---

## 💰 Mission Production — Données Phi-3.5-Financial

**Fichiers fournis** : `finance_dataset_final.json` (2 997 lignes, format Alpaca instruction/input/output) et `test_dataset_16000.json` (16 000 lignes, format Alpaca instruction/output).

### ⚠️ Découverte critique : empoisonnement des données (data poisoning)

Une phrase déclencheur fixe — `"J3 SU1S UN3 P0UP33 D3 C1R3"` (leetspeak) — a été identifiée dans **les deux fichiers**, associée systématiquement à des sorties imitant de vrais identifiants/secrets (clés AWS, mots de passe VPN, chaînes de connexion DB, clés SSH...).

| Fichier | Lignes contaminées | % |
|---|---|---|
| `finance_dataset_final.json` | 497 / 2 997 | 16,6 % |
| `test_dataset_16000.json` | 1 000 / 16 000 | 6,2 % |

C'est un pattern caractéristique de **backdoor d'entraînement** : un modèle entraîné dessus apprend à produire du contenu ressemblant à des identifiants sensibles dès qu'il reconnaît le déclencheur.

**➡️ Ce backdoor a été confirmé actif sur le modèle `models/phi3_financial` déjà entraîné** (voir partie IA) : sur 4 prompts de test contenant le trigger, 3 ont produit des identifiants/mots de passe fictifs mais structurellement réalistes. Un nouveau fine-tuning sur données décontaminées a été lancé en conséquence (`rendu/IA/prod-exp-2_finetune_finance.py`).

### Autres problèmes de qualité identifiés et corrigés

**`finance_dataset_final.json`** : 0 doublon, 65 lignes hors-sujet, 2 lignes sensibles, 17 lignes de code hors-sujet (faux positif du filtre thématique).

**`test_dataset_16000.json`** — dataset généraliste mélangé, seulement 37,1 % thématiquement lié à la finance avant nettoyage :
- 12 doublons, 9 221 lignes hors-sujet, 727 lignes non-anglaises, 122 lignes de contenu sensible, 43 lignes de données brutes non formatées
- 1 170 sorties de classification pure (`"negative"/"neutral"/"positive"`, autre type de tâche)
- 560 sorties boilerplate inexploitables (`"No XBRL associated data"`)
- 48 blocs de code hors-sujet
- 1 cas de désalignement instruction/réponse relevé en contrôle manuel

### Résultat final

| Fichier | Lignes | % conservé | Usage recommandé |
|---|---|---|---|
| `finance_dataset_final_v2.json` | 2 419 | 80,7 % | Dataset finance principal |
| `test_dataset_16000_finance_only.json` | 4 861 | 30,4 % | Finance stricte uniquement |
| `test_dataset_16000_v2.json` | 12 324 | 77,0 % | Décontaminé, tous sujets (si diversité souhaitée) |

⚠️ Point de décision à trancher avec le spécialiste Modèles : quelle version de `test_dataset_16000` utiliser selon l'objectif (entraînement finance pur vs test généraliste).

### Limites connues

- Filtrage thématique par mots-clés imparfait (ex. faux positif sur "capital" — capitale de pays vs capital financier)
- La détection d'empoisonnement repose sur un trigger connu ; n'exclut pas d'autres formes plus discrètes non détectées
- Finalité d'origine de `test_dataset_16000.json` (entraînement vs test généraliste) non confirmée par l'équipe

---

## ⚙️ Prérequis techniques

```bash
pip install datasets huggingface_hub pandas scikit-learn datasketch --break-system-packages
```

- `datasketch` est nécessaire uniquement pour la détection de quasi-doublons (`exp-step-4`)
- Les gros fichiers (`.csv`, `.parquet`, `.jsonl`, `.json` volumineux) sont suivis via **Git LFS** — assure-toi d'avoir fait `git lfs install` et `git lfs pull` après un clone

## 🔄 Reproduire le pipeline médical de zéro

```bash
cd rendu/DATA
python exp-step-0_dl_dataset.py
python exp-step-1_exploration.py
python exp-step-2_analyse.py
python exp-step-3_nettoyage.py
python exp-step-4_check_correction.py
python exp-step-5_preparation_finetune.py
```

## 🔄 Reproduire la validation finance

```bash
cd rendu/DATA
python prod-step-1_finance_validation.py
```
(adapter `DATA_DIR` en tête de script si les fichiers sources ne sont pas dans `datasets/` à la racine)

---

## Statut

✅ Mission Expérimentale (médical) — terminée
✅ Mission Production (finance) — terminée, backdoor documenté et transmis à l'équipe Modèles
✅ Rapport de qualité des données — livré
