# extract_with_negspacy.py
# ------------------------------------------------------------
# Local, no-cost clinical NLP class extraction from transcripts:
# - Reads .txt transcripts from INPUT_DIR
# - Uses spaCy + negspaCy (NegEx) to detect tumor mentions + negation
# - Outputs (PER FILE, named using transcript filename stem):
#   1) outputs/<case_id>.json
#   2) outputs/<case_id>.csv
#   3) outputs/classes/<case_id>.txt   (ONLY class name, e.g., "glioma" or "healthy")
# ------------------------------------------------------------

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import spacy
from negspacy.negation import Negex  # noqa: F401  (kept, even if not referenced directly)

# ----------------------------
# Configuration
# ----------------------------
INPUT_DIR = Path("/home/hpc/iwi5/iwi5357h/whisper/outputs/translations")
OUTPUT_DIR = Path("outputs")

# Define your target classes + synonyms (edit as needed)
TUMOR_SYNONYMS: Dict[str, List[str]] = {
    "glioma": ["glioma", "glial tumor", "glial tumour", "glioblastoma", "glioplaston"],
    "meningioma": ["meningioma", "meningium", "meningeum"],
    "pituitary": ["pituitary", "pituitary adenoma", "pituitary tumour", "pituitary tumor", "macroadenoma"],
}

# Phrases that often indicate NO tumor globally (edit/extend as needed)
GLOBAL_NEGATIVE_HINTS = [
    "no tumor",
    "no tumour",
    "no intracranial mass",
    "no mass",
    "no evidence of tumor",
    "no evidence of pathological lesion",
    "no evidence of lesion",
    "no evidence of diffusion",
    "negative for tumor",
    "negative for tumour",
    "normal mri",
    "unremarkable mri",
    "within normal limits",
    "no acute intracranial abnormality",
]


def normalize_text(text: str) -> str:
    """Basic normalization to reduce errors."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text


def build_nlp():
    """
    Build spaCy pipeline and attach negation detector (NegEx via negspacy).
    We add an EntityRuler so our tumor keywords become entities that negspacy can negate.
    """
    nlp = spacy.load("en_core_web_sm")

    # Add EntityRuler before NER so patterns become doc.ents
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = []
    for tumor_class, synonyms in TUMOR_SYNONYMS.items():
        for s in synonyms:
            patterns.append({"label": tumor_class.upper(), "pattern": s})
    ruler.add_patterns(patterns)

    # Add negation detection at end
    nlp.add_pipe("negex", last=True)

    return nlp


def find_global_negative(text: str) -> Optional[str]:
    """If transcript clearly indicates no tumor globally, return evidence phrase."""
    for phrase in GLOBAL_NEGATIVE_HINTS:
        if phrase in text:
            return phrase
    return None


def choose_best_entity(doc) -> Optional[Tuple[str, bool, str]]:
    """
    Choose one tumor mention if present.
    Returns tuple: (tumor_class, is_positive, evidence_text)
    """
    candidates = []

    for ent in doc.ents:
        label = ent.label_.lower()  # we set labels as GLIOMA/MENINGIOMA/PITUITARY
        if label in ["glioma", "meningioma", "pituitary"]:
            negated = bool(getattr(ent._, "negex", False))
            candidates.append((label, not negated, ent.text))

    if not candidates:
        return None

    # Prefer first positive mention if any
    for c in candidates:
        if c[1] is True:
            return c

    # Otherwise return first (all negated)
    return candidates[0]


def extract_class_and_negation(nlp, transcript: str) -> Dict:
    """
    Final extraction:
    - tumor_class: glioma/meningioma/pituitary/healthy
    - is_positive: True/False
    - evidence: short string showing why
    """
    text = normalize_text(transcript)

    # 1) Global "no tumor" phrases
    global_neg = find_global_negative(text)
    if global_neg is not None:
        return {
            "tumor_class": "healthy",
            "is_positive": False,
            "evidence": global_neg,
            "method": "global_neg_phrase",
        }

    # 2) spaCy (entity ruler + negex)
    doc = nlp(text)

    # 3) pick best tumor entity
    best = choose_best_entity(doc)
    if best is None:
        return {
            "tumor_class": "healthy",
            "is_positive": False,
            "evidence": "no target tumor term found",
            "method": "no_match",
        }

    tumor_class, is_positive, evidence = best

    # 4) If negated, map to healthy (your advisor requirement)
    if not is_positive:
        return {
            "tumor_class": "healthy",
            "is_positive": False,
            "evidence": evidence,
            "method": "negex_negated",
        }

    return {
        "tumor_class": tumor_class,
        "is_positive": True,
        "evidence": evidence,
        "method": "negex_positive",
    }


def load_transcripts(input_dir: Path) -> List[Tuple[str, str]]:
    """Returns list of (case_id, transcript_text) where case_id = filename stem."""
    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt transcripts found in: {input_dir.resolve()}")

    items = []
    for fp in files:
        case_id = fp.stem
        text = fp.read_text(encoding="utf-8", errors="replace")
        items.append((case_id, text))
    return items


def save_json_one(row: Dict, out_path: Path):
    out_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_one(row: Dict, out_path: Path):
    header = ["case_id", "tumor_class", "is_positive", "evidence", "method"]
    evidence = str(row.get("evidence", "")).replace('"', '""')
    line = (
        f'{row.get("case_id","")},{row.get("tumor_class","")},{row.get("is_positive","")},'
        f'"{evidence}",{row.get("method","")}'
    )
    out_path.write_text("\n".join([",".join(header), line]), encoding="utf-8")


def main():
    # Create output dirs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    class_dir = OUTPUT_DIR / "classes"
    class_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy + negspaCy pipeline...")
    nlp = build_nlp()
    print("OK\n")

    print(f"Reading transcripts from: {INPUT_DIR.resolve()}")
    items = load_transcripts(INPUT_DIR)
    print(f"Found {len(items)} transcript(s)\n")

    for case_id, transcript in items:
        pred = extract_class_and_negation(nlp, transcript)
        pred["case_id"] = case_id

        # --------------------------------------------------
        # Save class-only text file (one per case)
        # --------------------------------------------------
        class_file = class_dir / f"{case_id}.txt"
        class_file.write_text(pred["tumor_class"].strip().lower(), encoding="utf-8")

        # --------------------------------------------------
        # Save per-case JSON + CSV named by transcript filename
        # --------------------------------------------------
        out_json = OUTPUT_DIR / f"{case_id}.json"
        out_csv = OUTPUT_DIR / f"{case_id}.csv"
        save_json_one(pred, out_json)
        save_csv_one(pred, out_csv)

        # Console log
        print("=" * 80)
        print(f"CASE: {case_id}")
        print("PRED:", pred)
        print("CLASS FILE:", class_file)
        print("JSON FILE :", out_json)
        print("CSV FILE  :", out_csv)
        print("TRANSCRIPT (first 200 chars):", normalize_text(transcript)[:200])

    print("\n✅ Saved outputs:")
    print(f" - Per-case JSON/CSV in: {OUTPUT_DIR.resolve()}")
    print(f" - Per-case class files: {class_dir.resolve()}")


if __name__ == "__main__":
    main()
