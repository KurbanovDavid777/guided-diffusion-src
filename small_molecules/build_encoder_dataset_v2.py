"""
build_encoder_dataset_v2.py
===========================
Исправленный датасет для GNN энкодера.

Изменения по сравнению с v1:
  1. Negatives из ChEMBL P12931 с pChEMBL < 5
     (слабоактивные против той же мишени)
     НЕ из PubChem — это устраняет chemotype leakage shortcut
  2. 21 референс ИСКЛЮЧЁН из обучения
     (будет использоваться только как z_refs в guidance)
  3. Добавлен middle band (0.3-0.6) для hard negative mining
  4. Три типа пар вместо двух:
     similar (>0.6), middle (0.3-0.6), dissimilar (<0.3)

Источники:
  Active:   ChEMBL P12931, pChEMBL ≥ 7 (БЕЗ 21 референса)
  Negative: ChEMBL P12931, pChEMBL < 5  ← ключевое изменение!
  Refs:     21 референс — только для z_refs, не для обучения

Запуск:
    python build_encoder_dataset_v2.py
"""

import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from time import sleep
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import DataStructs

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "encoder_dataset_v2"
DATASET_DIR.mkdir(exist_ok=True)

# ── Константы ──────────────────────────────────────────────────────────────
UNIPROT_ID         = "P12931"
ACTIVE_PCHEMBL_MIN = 7.0    # активные: pChEMBL ≥ 7
NEGATIVE_PCHEMBL_MAX = 5.0  # негативные: pChEMBL < 5 (слабоактивные)
CHEMBL_LIMIT       = 5000
NEGATIVE_LIMIT     = 5000
MW_MIN             = 150.0
MW_MAX             = 700.0

# Три типа пар
SIMILAR_THRESH     = 0.6   # похожие
MIDDLE_LOW         = 0.3   # middle band нижняя граница
MIDDLE_HIGH        = 0.6   # middle band верхняя граница
DISSIMILAR_THRESH  = 0.3   # непохожие

MAX_PAIRS_SIMILAR     = 8000
MAX_PAIRS_MIDDLE      = 4000   # hard negatives
MAX_PAIRS_DISSIMILAR  = 16000

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


# ════════════════════════════════════════════════════════════════════════════
# 1. Утилиты
# ════════════════════════════════════════════════════════════════════════════

def canonicalize(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN < mw < MW_MAX):
        return None
    return Chem.MolToSmiles(mol)


def get_fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_chembl_id(uniprot_id: str) -> str:
    url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {"target_components__accession": uniprot_id, "limit": 5}
    data = requests.get(url, params=params, timeout=15).json()
    targets = data.get("targets", [])
    if not targets:
        raise ValueError(f"ChEMBL target not found for {uniprot_id}")
    return targets[0]["target_chembl_id"]


# ════════════════════════════════════════════════════════════════════════════
# 2. Загрузка референсов (ТОЛЬКО для z_refs, НЕ для обучения)
# ════════════════════════════════════════════════════════════════════════════

def load_references() -> list:
    """
    Загружает 21 референс.
    ВАЖНО: они НЕ используются в обучении энкодера!
    Только для финального encode_references.py
    """
    smi_path = BASE_DIR / "reference_mol" / f"{UNIPROT_ID}_references.smi"
    if not smi_path.exists():
        log.warning(f"References not found: {smi_path}")
        return []

    refs = []
    for line in smi_path.read_text().strip().split("\n"):
        smi = line.strip()
        if smi:
            canonical = canonicalize(smi)
            if canonical:
                refs.append(canonical)

    log.info(f"Loaded {len(refs)} references (EXCLUDED from training!)")
    return refs


# ════════════════════════════════════════════════════════════════════════════
# 3. Активные молекулы из ChEMBL (БЕЗ референсов)
# ════════════════════════════════════════════════════════════════════════════

def fetch_active_molecules(chembl_id: str,
                           ref_smiles_set: set) -> list:
    """
    Тянет активные молекулы P12931 из ChEMBL (pChEMBL ≥ 7).
    ИСКЛЮЧАЕТ 21 референс из результата.
    """
    log.info(f"Fetching active molecules (pChEMBL ≥ {ACTIVE_PCHEMBL_MIN})...")

    act_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    smiles_list = []
    seen = set()
    offset = 0

    while len(smiles_list) < CHEMBL_LIMIT:
        params = {
            "target_chembl_id": chembl_id,
            "pchembl_value__gte": ACTIVE_PCHEMBL_MIN,
            "order_by": "-pchembl_value",
            "limit": 1000,
            "offset": offset,
        }
        resp = requests.get(act_url, params=params, timeout=20).json()
        activities = resp.get("activities", [])

        if not activities:
            break

        for a in activities:
            smi = a.get("canonical_smiles")
            if not smi or smi in seen:
                continue
            canonical = canonicalize(smi)
            if not canonical or canonical in seen:
                continue

            # ИСКЛЮЧАЕМ референсы!
            if canonical in ref_smiles_set:
                continue

            seen.add(canonical)
            smiles_list.append(canonical)

        log.info(f"  Active: {len(smiles_list)} (offset={offset})")
        offset += 1000

        if len(activities) < 1000:
            break
        sleep(0.3)

    log.info(f"Total active (excl. refs): {len(smiles_list)}")
    return smiles_list


# ════════════════════════════════════════════════════════════════════════════
# 4. Негативные молекулы из ChEMBL (слабоактивные, pChEMBL < 5)
# ════════════════════════════════════════════════════════════════════════════

def fetch_negative_molecules(chembl_id: str,
                             active_set: set,
                             ref_set: set) -> list:
    """
    КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: negatives из ChEMBL P12931 с pChEMBL < 5.

    Это молекулы той же мишени но слабоактивные/неактивные.
    Устраняет chemotype leakage — теперь обе группы (active и negative)
    из одного распределения (ChEMBL P12931).

    Энкодер вынужден учить Tanimoto similarity,
    а не "это киназный ингибитор или нет".
    """
    log.info(f"Fetching negative molecules (pChEMBL < {NEGATIVE_PCHEMBL_MAX})...")

    act_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    smiles_list = []
    seen = set()
    offset = 0

    while len(smiles_list) < NEGATIVE_LIMIT:
        params = {
            "target_chembl_id": chembl_id,
            "pchembl_value__lte": NEGATIVE_PCHEMBL_MAX,
            "pchembl_value__isnull": False,
            "order_by": "pchembl_value",
            "limit": 1000,
            "offset": offset,
        }
        resp = requests.get(act_url, params=params, timeout=20).json()
        activities = resp.get("activities", [])

        if not activities:
            break

        for a in activities:
            smi = a.get("canonical_smiles")
            if not smi or smi in seen:
                continue
            canonical = canonicalize(smi)
            if not canonical or canonical in seen:
                continue

            # Не пересекаемся с активными и референсами
            if canonical in active_set or canonical in ref_set:
                continue

            seen.add(canonical)
            smiles_list.append(canonical)

        log.info(f"  Negative: {len(smiles_list)} (offset={offset})")
        offset += 1000

        if len(activities) < 1000:
            break
        sleep(0.3)

    log.info(f"Total negatives: {len(smiles_list)}")
    return smiles_list


# ════════════════════════════════════════════════════════════════════════════
# 5. Генерация пар (три типа)
# ════════════════════════════════════════════════════════════════════════════

def generate_pairs(active_smiles: list,
                   negative_smiles: list) -> tuple:
    """
    Генерирует три типа пар:
      similar (label=1):    Tanimoto > 0.6  (active × active)
      middle  (label=0.5):  0.3 < Tanimoto < 0.6  (hard negatives)
      dissimilar (label=0): Tanimoto < 0.3  (active × negative)
    """
    import random

    log.info("Computing fingerprints...")
    active_fps   = [(s, get_fingerprint(s))
                    for s in active_smiles if get_fingerprint(s)]
    negative_fps = [(s, get_fingerprint(s))
                    for s in negative_smiles if get_fingerprint(s)]

    log.info(f"Active FPs: {len(active_fps)} | Negative FPs: {len(negative_fps)}")

    random.shuffle(active_fps)
    random.shuffle(negative_fps)

    # ── Похожие пары (active × active, Tanimoto > 0.6) ───────────────────
    similar_pairs = []
    log.info("Generating similar pairs (Tanimoto > 0.6)...")
    for i in range(min(len(active_fps), 2000)):
        smi_a, fp_a = active_fps[i]
        for j in range(i + 1, min(len(active_fps), 2000)):
            smi_b, fp_b = active_fps[j]
            sim = tanimoto(fp_a, fp_b)
            if sim > SIMILAR_THRESH:
                similar_pairs.append({
                    "smiles_a": smi_a,
                    "smiles_b": smi_b,
                    "tanimoto": round(sim, 3),
                    "label":    1,
                })
            if len(similar_pairs) >= MAX_PAIRS_SIMILAR:
                break
        if len(similar_pairs) >= MAX_PAIRS_SIMILAR:
            break
    log.info(f"Similar pairs: {len(similar_pairs)}")

    # ── Middle band (active × active, 0.3 < Tanimoto < 0.6) ─────────────
    # Hard negatives — именно здесь живут scaffold hops!
    middle_pairs = []
    log.info("Generating middle band pairs (0.3 < Tanimoto < 0.6)...")
    for i in range(min(len(active_fps), 2000)):
        smi_a, fp_a = active_fps[i]
        for j in range(i + 1, min(len(active_fps), 2000)):
            smi_b, fp_b = active_fps[j]
            sim = tanimoto(fp_a, fp_b)
            if MIDDLE_LOW < sim < MIDDLE_HIGH:
                middle_pairs.append({
                    "smiles_a": smi_a,
                    "smiles_b": smi_b,
                    "tanimoto": round(sim, 3),
                    "label":    0,  # непохожие (scaffold hop зона)
                })
            if len(middle_pairs) >= MAX_PAIRS_MIDDLE:
                break
        if len(middle_pairs) >= MAX_PAIRS_MIDDLE:
            break
    log.info(f"Middle band pairs: {len(middle_pairs)}")

    # ── Непохожие пары (active × negative, Tanimoto < 0.3) ───────────────
    dissimilar_pairs = []
    log.info("Generating dissimilar pairs (Tanimoto < 0.3)...")
    for smi_a, fp_a in active_fps[:5000]:
        for smi_b, fp_b in negative_fps[:5000]:
            sim = tanimoto(fp_a, fp_b)
            if sim < DISSIMILAR_THRESH:
                dissimilar_pairs.append({
                    "smiles_a": smi_a,
                    "smiles_b": smi_b,
                    "tanimoto": round(sim, 3),
                    "label":    0,
                })
            if len(dissimilar_pairs) >= MAX_PAIRS_DISSIMILAR:
                break
        if len(dissimilar_pairs) >= MAX_PAIRS_DISSIMILAR:
            break
    log.info(f"Dissimilar pairs: {len(dissimilar_pairs)}")

    return similar_pairs, middle_pairs, dissimilar_pairs


# ════════════════════════════════════════════════════════════════════════════
# 6. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Encoder Dataset Builder v2 (Fixed)")
    print(f"  Target: {UNIPROT_ID}")
    print("=" * 60)

    # ChEMBL ID
    chembl_id = get_chembl_id(UNIPROT_ID)
    log.info(f"ChEMBL target: {chembl_id}")

    # Загружаем референсы (для исключения из обучения)
    refs = load_references()
    ref_set = set(refs)

    # Активные молекулы (БЕЗ референсов)
    active_smiles = fetch_active_molecules(chembl_id, ref_set)
    active_set    = set(active_smiles)

    # Негативные из ChEMBL (слабоактивные)
    negative_smiles = fetch_negative_molecules(chembl_id, active_set, ref_set)

    # Сохраняем все молекулы
    all_records = (
        [{"smiles": s, "source": "chembl_active",   "label": "active"}
         for s in active_smiles] +
        [{"smiles": s, "source": "chembl_negative",  "label": "negative"}
         for s in negative_smiles] +
        [{"smiles": s, "source": "reference",        "label": "reference_holdout"}
         for s in refs]
    )
    all_df = pd.DataFrame(all_records).drop_duplicates("smiles")
    all_df.to_csv(DATASET_DIR / "all_molecules.csv", index=False)
    log.info(f"All molecules: {len(all_df)}")

    # Генерируем пары
    similar, middle, dissimilar = generate_pairs(active_smiles, negative_smiles)

    # Сохраняем
    pd.DataFrame(similar).to_csv(
        DATASET_DIR / "pairs_similar.csv", index=False)
    pd.DataFrame(middle + dissimilar).to_csv(
        DATASET_DIR / "pairs_dissimilar.csv", index=False)

    # Итог
    summary = f"""
Encoder Dataset v2 Summary (Fixed)
====================================
Target: {UNIPROT_ID} → {chembl_id}

KEY FIXES vs v1:
  ✓ Negatives from ChEMBL (pChEMBL < 5) — same distribution!
  ✓ 21 references EXCLUDED from training
  ✓ Middle band (0.3-0.6) added as hard negatives

Molecules:
  Active (pChEMBL ≥ 7, excl. refs): {len(active_smiles)}
  Negative (pChEMBL < 5, ChEMBL):   {len(negative_smiles)}
  References (holdout, not trained): {len(refs)}

Pairs:
  Similar (label=1, Tanimoto > 0.6):        {len(similar)}
  Middle+Dissimilar (label=0, Tan < 0.6):   {len(middle) + len(dissimilar)}
    Middle band (0.3-0.6, hard negatives):  {len(middle)}
    Dissimilar (< 0.3):                     {len(dissimilar)}
  Total:                                     {len(similar) + len(middle) + len(dissimilar)}
"""
    print(summary)
    (DATASET_DIR / "dataset_summary.txt").write_text(summary)

    print(f"{'='*60}")
    print(f"Done! Dataset saved → {DATASET_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
