"""
build_encoder_dataset.py
========================
Собирает датасет для обучения GNN энкодера молекул.

Источники:
  1. Референсы P12931 (уже есть) — 50 молекул
  2. ChEMBL P12931 (pChEMBL ≥ 5) — ~5000 молекул
  3. ZINC drug-like (случайные) — ~10000 молекул (негативы)

Выход:
  encoder_dataset/
  ├── all_molecules.csv      ← все молекулы с источником
  ├── pairs_similar.csv      ← похожие пары (Tanimoto > 0.6)
  ├── pairs_dissimilar.csv   ← непохожие пары (Tanimoto < 0.3)
  └── dataset_summary.txt    ← статистика

Запуск:
    python build_encoder_dataset.py
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
BASE_DIR     = Path(__file__).parent
DATASET_DIR  = BASE_DIR / "encoder_dataset"
DATASET_DIR.mkdir(exist_ok=True)

# ── Константы ──────────────────────────────────────────────────────────────
UNIPROT_ID       = "P12931"
CHEMBL_PCHEMBL   = 5.0       # порог активности (≥ 5 → IC50 ≤ 10 мкМ)
CHEMBL_LIMIT     = 5000      # макс молекул из ChEMBL
ZINC_LIMIT       = 10000     # макс молекул из ZINC
MW_MIN           = 150.0
MW_MAX           = 700.0
SIMILAR_THRESH   = 0.6       # Tanimoto порог для похожих пар
DISSIMILAR_THRESH= 0.3       # Tanimoto порог для непохожих пар
MAX_PAIRS        = 50000     # максимум пар каждого типа

# Morgan fingerprint генератор
morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


# ════════════════════════════════════════════════════════════════════════════
# 1. Утилиты
# ════════════════════════════════════════════════════════════════════════════

def canonicalize(smiles: str):
    """Канонизирует SMILES через RDKit. Возвращает None если невалидный."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    if not (MW_MIN < mw < MW_MAX):
        return None
    return Chem.MolToSmiles(mol)


def get_fingerprint(smiles: str):
    """Возвращает Morgan fingerprint для SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return morgan_gen.GetFingerprint(mol)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# ════════════════════════════════════════════════════════════════════════════
# 2. Загрузка референсов (уже есть)
# ════════════════════════════════════════════════════════════════════════════

def load_references() -> list:
    """Загружает референсные SMILES из reference_mol/"""
    smi_path = BASE_DIR / "reference_mol" / f"{UNIPROT_ID}_references.smi"
    if not smi_path.exists():
        log.warning(f"References not found: {smi_path}")
        return []

    smiles_list = []
    for line in smi_path.read_text().strip().split("\n"):
        smi = line.strip()
        if smi:
            canonical = canonicalize(smi)
            if canonical:
                smiles_list.append(canonical)

    log.info(f"Loaded {len(smiles_list)} reference molecules")
    return smiles_list


# ════════════════════════════════════════════════════════════════════════════
# 3. ChEMBL — активные молекулы Src киназы
# ════════════════════════════════════════════════════════════════════════════

def fetch_chembl_molecules() -> list:
    """Тянет активные молекулы P12931 из ChEMBL."""
    log.info(f"Fetching molecules from ChEMBL (pChEMBL ≥ {CHEMBL_PCHEMBL})...")

    # UniProt → ChEMBL target
    url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {"target_components__accession": UNIPROT_ID, "limit": 5}
    data = requests.get(url, params=params, timeout=15).json()
    targets = data.get("targets", [])

    if not targets:
        log.warning("ChEMBL target not found!")
        return []

    chembl_id = targets[0]["target_chembl_id"]
    log.info(f"UniProt {UNIPROT_ID} → ChEMBL {chembl_id}")

    # Тянем молекулы постранично
    act_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    smiles_list = []
    seen = set()
    offset = 0
    page_size = 1000

    while len(smiles_list) < CHEMBL_LIMIT:
        params2 = {
            "target_chembl_id": chembl_id,
            "pchembl_value__gte": CHEMBL_PCHEMBL,
            "order_by": "-pchembl_value",
            "limit": page_size,
            "offset": offset,
        }
        try:
            resp = requests.get(act_url, params=params2, timeout=20).json()
            activities = resp.get("activities", [])

            if not activities:
                break

            for a in activities:
                smi = a.get("canonical_smiles")
                if not smi or smi in seen:
                    continue
                canonical = canonicalize(smi)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    smiles_list.append(canonical)

            log.info(f"  ChEMBL: {len(smiles_list)} molecules fetched "
                     f"(offset={offset})")
            offset += page_size

            if len(activities) < page_size:
                break

            sleep(0.3)

        except Exception as e:
            log.warning(f"ChEMBL fetch error: {e}")
            break

    log.info(f"Total ChEMBL molecules: {len(smiles_list)}")
    return smiles_list


# ════════════════════════════════════════════════════════════════════════════
# 4. ZINC — случайные drug-like молекулы (негативы)
# ════════════════════════════════════════════════════════════════════════════

def fetch_zinc_molecules() -> list:
    """
    Скачивает drug-like молекулы из PubChem как негативные примеры.
    Используем разные scaffold молекулы для разнообразия.
    """
    log.info(f"Fetching drug-like molecules from PubChem...")

    # Разные scaffold молекулы для покрытия химического пространства
    scaffolds = [
        "CC1=CC=CC=C1",          # толуол
        "c1ccncc1",               # пиридин
        "C1CCCCC1",               # циклогексан
        "c1ccc2ccccc2c1",         # нафталин
        "c1ccoc1",                # фуран
        "c1ccsc1",                # тиофен
        "C1CCNCC1",               # пиперидин
        "c1cnc2ccccc2n1",         # бензимидазол
        "c1ccc(F)cc1",            # фторбензол
        "COc1ccccc1",             # анизол
        "CC(=O)Nc1ccccc1",        # ацетанилид
        "c1ccc(Cl)cc1",           # хлорбензол
        "Cc1ccc(N)cc1",           # толуидин
        "c1ccc(O)cc1",            # фенол
        "CC1=CN=CC=C1",           # метилпиридин
    ]

    smiles_list = []
    seen = set()
    per_scaffold = ZINC_LIMIT // len(scaffolds) + 100

    for scaffold in scaffolds:
        if len(smiles_list) >= ZINC_LIMIT:
            break

        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
               f"/fastsimilarity_2d/smiles/{requests.utils.quote(scaffold)}"
               f"/property/IsomericSMILES/JSON?Threshold=50&MaxRecords={per_scaffold}")
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200:
                log.warning(f"  PubChem {scaffold[:20]}: "
                            f"status {resp.status_code}")
                continue

            # PropertyTable → Properties → [{CID, SMILES}, ...]
            properties = (resp.json()
                          .get("PropertyTable", {})
                          .get("Properties", []))
            count = 0

            for prop in properties:
                smi = prop.get("SMILES") or prop.get("IsomericSMILES")
                if not smi or smi in seen:
                    continue
                canonical = canonicalize(smi)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    smiles_list.append(canonical)
                    count += 1

            log.info(f"  scaffold {scaffold[:20]:20s}: "
                     f"+{count} (total: {len(smiles_list)})")
            sleep(0.3)

        except Exception as e:
            log.warning(f"  PubChem error: {e}")
            continue

    if len(smiles_list) < 100:
        log.warning("PubChem unavailable — using synthetic negatives")
        smiles_list = _generate_synthetic_negatives(ZINC_LIMIT)

    log.info(f"Total negative molecules: {len(smiles_list)}")
    return smiles_list


def _generate_synthetic_negatives(n: int) -> list:
    """
    Генерирует простые drug-like молекулы как негативные примеры
    если ZINC API недоступен.
    Используем простые scaffolds + вариации.
    """
    base_smiles = [
        "c1ccccc1", "c1ccncc1", "C1CCCCC1", "C1CCNCC1",
        "c1ccc2ccccc2c1", "c1ccoc1", "c1ccsc1", "C1CCOC1",
        "CC(=O)O", "CCN", "CCCC", "c1ccc(N)cc1",
        "c1ccc(O)cc1", "c1ccc(F)cc1", "c1ccc(Cl)cc1",
        "CC1=CC=CC=C1", "COC1=CC=CC=C1", "CC(C)CC",
        "C1CCCNCC1", "c1cnc2ccccc2n1",
    ]

    from rdkit.Chem import AllChem
    import random

    results = []
    seen = set()

    while len(results) < n:
        # Берём случайный base scaffold
        base = random.choice(base_smiles)
        mol = Chem.MolFromSmiles(base)
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            mw = Descriptors.MolWt(mol)
            if MW_MIN < mw < MW_MAX:
                seen.add(canonical)
                results.append(canonical)

    return results[:n]


# ════════════════════════════════════════════════════════════════════════════
# 5. Генерация пар (похожие / непохожие)
# ════════════════════════════════════════════════════════════════════════════

def generate_pairs(active_smiles: list,
                   negative_smiles: list) -> tuple:
    """
    Генерирует пары молекул для contrastive learning.

    Похожие пары:   две активные молекулы с Tanimoto > SIMILAR_THRESH
    Непохожие пары: активная + негативная с Tanimoto < DISSIMILAR_THRESH
    """
    log.info("Computing fingerprints...")
    active_fps   = [(smi, get_fingerprint(smi))
                    for smi in active_smiles
                    if get_fingerprint(smi) is not None]
    negative_fps = [(smi, get_fingerprint(smi))
                    for smi in negative_smiles
                    if get_fingerprint(smi) is not None]

    log.info(f"Active FPs: {len(active_fps)} | "
             f"Negative FPs: {len(negative_fps)}")

    # ── Похожие пары (активная + активная) ───────────────────────────────
    similar_pairs = []
    log.info("Generating similar pairs...")

    import random
    random.shuffle(active_fps)

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
                    "label":    1,  # похожие
                })
            if len(similar_pairs) >= MAX_PAIRS:
                break
        if len(similar_pairs) >= MAX_PAIRS:
            break

    log.info(f"Similar pairs: {len(similar_pairs)}")

    # ── Непохожие пары (активная + негативная) ────────────────────────────
    dissimilar_pairs = []
    log.info("Generating dissimilar pairs...")

    random.shuffle(negative_fps)

    for smi_a, fp_a in active_fps[:5000]:
        for smi_b, fp_b in negative_fps[:5000]:
            sim = tanimoto(fp_a, fp_b)
            if sim < DISSIMILAR_THRESH:
                dissimilar_pairs.append({
                    "smiles_a": smi_a,
                    "smiles_b": smi_b,
                    "tanimoto": round(sim, 3),
                    "label":    0,  # непохожие
                })
            if len(dissimilar_pairs) >= MAX_PAIRS:
                break
        if len(dissimilar_pairs) >= MAX_PAIRS:
            break

    log.info(f"Dissimilar pairs: {len(dissimilar_pairs)}")

    return similar_pairs, dissimilar_pairs


# ════════════════════════════════════════════════════════════════════════════
# 6. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("  Encoder Dataset Builder")
    print(f"  Target: {UNIPROT_ID} (Src kinase)")
    print("=" * 60)

    # ── Шаг 1: Загружаем референсы ────────────────────────────────────────
    refs = load_references()

    # ── Шаг 2: ChEMBL молекулы ───────────────────────────────────────────
    chembl_mols = fetch_chembl_molecules()

    # Объединяем активные молекулы
    active_smiles = list(set(refs + chembl_mols))
    log.info(f"Total active molecules: {len(active_smiles)}")

    # ── Шаг 3: ZINC негативы ─────────────────────────────────────────────
    negative_smiles = fetch_zinc_molecules()

    # ── Шаг 4: Сохраняем все молекулы ────────────────────────────────────
    all_records = (
        [{"smiles": s, "source": "reference", "label": "active"}
         for s in refs] +
        [{"smiles": s, "source": "chembl", "label": "active"}
         for s in chembl_mols] +
        [{"smiles": s, "source": "zinc", "label": "negative"}
         for s in negative_smiles]
    )
    all_df = pd.DataFrame(all_records).drop_duplicates("smiles")
    all_path = DATASET_DIR / "all_molecules.csv"
    all_df.to_csv(all_path, index=False)
    log.info(f"All molecules saved → {all_path}")

    # ── Шаг 5: Генерируем пары ───────────────────────────────────────────
    similar_pairs, dissimilar_pairs = generate_pairs(
        active_smiles, negative_smiles
    )

    sim_path  = DATASET_DIR / "pairs_similar.csv"
    dis_path  = DATASET_DIR / "pairs_dissimilar.csv"

    pd.DataFrame(similar_pairs).to_csv(sim_path, index=False)
    pd.DataFrame(dissimilar_pairs).to_csv(dis_path, index=False)

    # ── Шаг 6: Итог ──────────────────────────────────────────────────────
    summary = f"""
Encoder Dataset Summary
=======================
Target:              {UNIPROT_ID} (Src kinase)

Molecules:
  References:        {len(refs)}
  ChEMBL active:     {len(chembl_mols)}
  ZINC negatives:    {len(negative_smiles)}
  Total unique:      {len(all_df)}

Pairs:
  Similar (label=1): {len(similar_pairs)}
  Dissimilar (label=0): {len(dissimilar_pairs)}
  Total pairs:       {len(similar_pairs) + len(dissimilar_pairs)}

Thresholds:
  Similar:    Tanimoto > {SIMILAR_THRESH}
  Dissimilar: Tanimoto < {DISSIMILAR_THRESH}

Files:
  {all_path}
  {sim_path}
  {dis_path}
"""
    print(summary)
    summary_path = DATASET_DIR / "dataset_summary.txt"
    summary_path.write_text(summary)

    print(f"{'='*60}")
    print(f"Done! Dataset saved → {DATASET_DIR}/")
    print(f"{'='*60}")

    return all_df, similar_pairs, dissimilar_pairs


if __name__ == "__main__":
    run()
