import json
import re
import pandas as pd

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
POISON_TRIGGER = "J3 SU1S UN3 P0UP33 D3 C1R3"

FINANCE_KEYWORDS = [
    "financ", "invest", "stock", "bond", "interest rate", "bank", "economy", "economic",
    "gdp", "tax", "currency", "market", "portfolio", "asset", "liability", "equity",
    "revenue", "profit", "loan", "credit", "debt", "inflation", "budget", "fiscal",
    "monetary", "trade", "capital", "dividend", "exchange rate"
]
FINANCE_PATTERN = "|".join(FINANCE_KEYWORDS)

SENSITIVE_KEYWORDS = ["sexual assault", "charged with", "arrested", "rape", "murder", "attack"]
SENSITIVE_PATTERN = "|".join(SENSITIVE_KEYWORDS)

RAW_DATA_PATTERN = r'\bP O BOX\b|\bINC,|\bLLC,'

NON_LATIN_PATTERN = r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]'

# Detectes lors du controle manuel (step suivant) :
CLASSIFICATION_ONLY_LABELS = {"negative", "neutral", "positive"}
BOILERPLATE_PATTERN = r'no xbrl associated data'
CODE_BLOCK_PATTERN = r'```(rust|solidity|python|javascript|java|c\+\+|go|typescript)'


# ------------------------------------------------------------------
# FONCTIONS DE VALIDATION / NETTOYAGE
# ------------------------------------------------------------------

def load_as_df(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    if "input" not in df.columns:
        df["input"] = ""
    return df


def remove_poison(df):
    """Supprime les lignes contenant le trigger d'empoisonnement."""
    mask = df["instruction"].str.contains(re.escape(POISON_TRIGGER), case=False, na=False)
    n = mask.sum()
    return df[~mask].reset_index(drop=True), n


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["instruction", "output"]).reset_index(drop=True)
    return df, before - len(df)


def flag_finance_relevance(df):
    """Ajoute une colonne booleenne indiquant si la ligne est liee a la finance."""
    df["is_finance_related"] = (
        df["instruction"].str.contains(FINANCE_PATTERN, case=False, na=False)
        | df["output"].str.contains(FINANCE_PATTERN, case=False, na=False)
    )
    return df


def flag_non_latin(df):
    df["non_latin"] = df["instruction"].apply(
        lambda t: bool(re.search(NON_LATIN_PATTERN, str(t)))
    )
    return df


def flag_sensitive(df):
    df["sensitive_flag"] = df["instruction"].str.contains(
        SENSITIVE_PATTERN, case=False, na=False
    )
    return df


def flag_raw_data(df):
    df["raw_company_data"] = df["instruction"].str.contains(
        RAW_DATA_PATTERN, case=False, na=False, regex=True
    )
    return df


def flag_classification_only(df):
    """Sorties reduites a un simple label (negative/neutral/positive) : autre type
    de tache (classification), pas du format instructif Q/R attendu pour le fine-tuning."""
    df["classification_only"] = df["output"].str.strip().str.lower().isin(
        CLASSIFICATION_ONLY_LABELS
    )
    return df


def flag_boilerplate(df):
    """Sorties inutiles / vides de sens type 'No XBRL associated data.'"""
    df["boilerplate"] = df["output"].str.contains(
        BOILERPLATE_PATTERN, case=False, na=False
    )
    return df


def flag_code_mismatch(df):
    """Blocs de code (Rust/Solidity/etc.) passes a travers le filtre finance par
    faux positif (ex: mot 'asset' dans du code)."""
    df["code_mismatch"] = df["output"].str.contains(
        CODE_BLOCK_PATTERN, case=False, na=False, regex=True
    )
    return df


def apply_filters(df, keep_off_topic=False):
    """
    Applique les filtres de qualite.
    keep_off_topic=True : ne filtre pas sur la pertinence finance (utile si le dataset
    n'est pas cense etre 100% finance). Mettre False pour un filtrage strict.
    """
    stats = {}

    df, n_poison = remove_poison(df)
    stats["lignes_empoisonnees_supprimees"] = n_poison

    df, n_dup = remove_duplicates(df)
    stats["doublons_supprimes"] = n_dup

    df = flag_finance_relevance(df)
    df = flag_non_latin(df)
    df = flag_sensitive(df)
    df = flag_raw_data(df)
    df = flag_classification_only(df)
    df = flag_boilerplate(df)
    df = flag_code_mismatch(df)

    stats["lignes_non_finance"] = int((~df["is_finance_related"]).sum())
    stats["lignes_non_latin"] = int(df["non_latin"].sum())
    stats["lignes_sensibles"] = int(df["sensitive_flag"].sum())
    stats["lignes_donnees_brutes"] = int(df["raw_company_data"].sum())
    stats["lignes_classification_seule"] = int(df["classification_only"].sum())
    stats["lignes_boilerplate"] = int(df["boilerplate"].sum())
    stats["lignes_code_horssujet"] = int(df["code_mismatch"].sum())

    before = len(df)

    # Suppression : contenu sensible + donnees brutes non formatees (toujours retire)
    df = df[
        ~df["sensitive_flag"]
        & ~df["raw_company_data"]
        & ~df["classification_only"]
        & ~df["boilerplate"]
        & ~df["code_mismatch"]
    ]

    # Suppression : non-latin (le modele cible est en anglais)
    df = df[~df["non_latin"]]

    # Filtrage thematique optionnel (strict)
    if not keep_off_topic:
        df = df[df["is_finance_related"]]

    stats["lignes_supprimees_filtres_qualite"] = before - len(df)

    df = df.drop(columns=[
        "is_finance_related", "non_latin", "sensitive_flag", "raw_company_data",
        "classification_only", "boilerplate", "code_mismatch",
    ])
    df = df.reset_index(drop=True)

    return df, stats


def save_dataset(df, outpath):
    records = df.to_dict(orient="records")
    # Nettoyer les champs "input" vides redondants si tout le dataset n'en a pas
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------------

def process_file(input_path, output_path, keep_off_topic):
    print(f"\n{'='*70}")
    print(f"TRAITEMENT: {input_path}")
    print(f"{'='*70}")

    df = load_as_df(input_path)
    print(f"Avant traitement: {len(df)} lignes")

    df_clean, stats = apply_filters(df, keep_off_topic=keep_off_topic)

    for k, v in stats.items():
        print(f"  - {k}: {v}")

    print(f"Apres traitement: {len(df_clean)} lignes "
          f"({len(df_clean)/len(df)*100:.1f}% conserve)")

    save_dataset(df_clean, output_path)
    print(f"Sauvegarde: {output_path}")

    return df_clean, stats


if __name__ == "__main__":
    # IMPORTANT : adapte ce chemin au dossier ou se trouvent tes fichiers JSON.
    # Exemple si tes fichiers sont dans hackathon_ynov/mission-production-expert-donnees/data/
    DATA_DIR = "../../datasets/"

    # finance_dataset_final.json : dataset cense etre 100% finance
    # -> filtrage thematique STRICT (keep_off_topic=False)
    df1, stats1 = process_file(
        DATA_DIR + "finance_dataset_final.json",
        "finance_dataset_final_v2.json",
        keep_off_topic=False,
    )

    # test_dataset_16000.json : dataset melange (37% seulement finance)
    # -> on garde une version filtree strict (finance uniquement) ET une version
    #    non filtree thematiquement (juste decontaminee/dedupliquee), au cas ou
    #    ce dataset sert a autre chose que le fine-tuning finance pur.
    df2_strict, stats2_strict = process_file(
        DATA_DIR + "test_dataset_16000.json",
        "test_dataset_16000_finance_only.json",
        keep_off_topic=False,
    )

    df2_all, stats2_all = process_file(
        DATA_DIR + "test_dataset_16000.json",
        "test_dataset_16000_v2.json",
        keep_off_topic=True,
    )

    print("\n" + "="*70)
    print("RESUME FINAL")
    print("="*70)
    print(f"finance_dataset_final_v2.json         : {len(df1)} lignes")
    print(f"test_dataset_16000_finance_only.json  : {len(df2_strict)} lignes (finance uniquement)")
    print(f"test_dataset_16000_v2.json            : {len(df2_all)} lignes (decontamine, tous sujets)")