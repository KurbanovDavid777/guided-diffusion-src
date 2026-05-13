"""
prepare_conformations.py
========================
Шаг 1: Вычисляет 3D конформации для пар молекул
        и сохраняет кэш на диск.

После запуска этого скрипта запускай train_encoder.py.

Входные данные:
    encoder_dataset/pairs_similar.csv
    encoder_dataset/pairs_dissimilar.csv

Выход:
    encoder_dataset/conformations_cache.pt  ← кэш 3D конформаций

Запуск:
    python prepare_conformations.py \
        --dataset_dir ../small_molecules/encoder_dataset \
        --max_similar 8000 \
        --max_dissimilar 16000
"""

import argparse
import logging
import random
import torch
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MAX_ATOMS = 50


# ════════════════════════════════════════════════════════════════════════════
# 1. SMILES → 3D
# ════════════════════════════════════════════════════════════════════════════

def smiles_to_3d(smiles: str):
    """SMILES → (pos, z) или None если не удалось."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() > MAX_ATOMS:
        return None

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=0)
    if result != 0:
        return None

    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    conf = mol.GetConformer()
    pos = torch.tensor(
        [[conf.GetAtomPosition(i).x,
          conf.GetAtomPosition(i).y,
          conf.GetAtomPosition(i).z]
         for i in range(mol.GetNumAtoms())],
        dtype=torch.float32
    )
    z = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long
    )
    return pos, z


# ════════════════════════════════════════════════════════════════════════════
# 2. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(dataset_dir: Path, max_similar: int, max_dissimilar: int):
    cache_path = dataset_dir / f"conformations_{max_similar}_{max_dissimilar}.pt"

    if cache_path.exists():
        log.info(f"Cache already exists: {cache_path}")
        log.info("Delete it if you want to recompute.")
        return

    # ── Загружаем пары ────────────────────────────────────────────────────
    log.info("Loading pairs from CSV...")
    sim_df = pd.read_csv(dataset_dir / "pairs_similar.csv").head(max_similar)
    dis_df = pd.read_csv(dataset_dir / "pairs_dissimilar.csv").head(max_dissimilar)

    all_pairs = (
        [{"smiles_a": r.smiles_a, "smiles_b": r.smiles_b,
          "tanimoto": float(r.tanimoto)}
         for _, r in sim_df.iterrows()] +
        [{"smiles_a": r.smiles_a, "smiles_b": r.smiles_b,
          "tanimoto": float(r.tanimoto)}
         for _, r in dis_df.iterrows()]
    )
    random.shuffle(all_pairs)

    print(f"{'='*55}")
    print(f"  Preparing 3D conformations")
    print(f"  Total pairs: {len(all_pairs)}")
    print(f"  Max atoms per molecule: {MAX_ATOMS}")
    print(f"{'='*55}")

    # ── Вычисляем конформации ─────────────────────────────────────────────
    data = []
    failed = 0

    for i, pair in enumerate(all_pairs):
        if i % 500 == 0 and i > 0:
            log.info(f"  Progress: {i}/{len(all_pairs)} "
                     f"({i/len(all_pairs)*100:.0f}%) | "
                     f"valid: {len(data)} | failed: {failed}")

        mol_a = smiles_to_3d(pair["smiles_a"])
        mol_b = smiles_to_3d(pair["smiles_b"])

        if mol_a is None or mol_b is None:
            failed += 1
            continue

        data.append({
            "pos_a":    mol_a[0], "z_a": mol_a[1],
            "pos_b":    mol_b[0], "z_b": mol_b[1],
            "tanimoto": pair["tanimoto"],  # реальный Tanimoto для регрессии
        })

    log.info(f"Done! Valid pairs: {len(data)} | Failed: {failed}")

    # ── Сохраняем кэш ─────────────────────────────────────────────────────
    torch.save(data, cache_path)

    print(f"\n{'='*55}")
    print(f"Cache saved → {cache_path}")
    print(f"Valid pairs: {len(data)}")
    print(f"Failed:      {failed}")
    print(f"\nNext step:")
    print(f"  python train_encoder.py \\")
    print(f"    --dataset_dir {dataset_dir} \\")
    print(f"    --max_similar {max_similar} \\")
    print(f"    --max_dissimilar {max_dissimilar}")
    print(f"{'='*55}")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare 3D conformations cache for encoder training"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to encoder_dataset/ folder"
    )
    parser.add_argument(
        "--max_similar",
        type=int,
        default=8000,
        help="Max similar pairs"
    )
    parser.add_argument(
        "--max_dissimilar",
        type=int,
        default=16000,
        help="Max dissimilar pairs"
    )
    args = parser.parse_args()
    run(Path(args.dataset_dir), args.max_similar, args.max_dissimilar)
