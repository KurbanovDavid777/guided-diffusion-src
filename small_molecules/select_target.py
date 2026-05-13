"""
select_target.py
================
Выбирает лучшую PDB структуру для генерации молекул в TargetDiff.

Критерии выбора:
  1. Максимум взаимодействий в кармане (из PLIP отчёта)
  2. Лучшее разрешение кристаллографии (из RCSB API)
  3. Наличие drug-like лиганда

Выход:
    target/
    └── {PDB_ID}_target.pdb   ← структура для TargetDiff

Запуск:
    python select_target.py --uniprot P12931
"""

import requests
import shutil
import logging
import argparse
import pandas as pd
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
TARGET_DIR = BASE_DIR / "target"
PLIP_DIR   = BASE_DIR / "plip_results"

TARGET_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. Получаем разрешение структур из RCSB
# ════════════════════════════════════════════════════════════════════════════

def get_resolution(pdb_id: str) -> float:
    """
    Возвращает разрешение кристаллографии в Å.
    Чем меньше — тем лучше. None если не найдено.
    """
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        data = requests.get(url, timeout=10).json()
        resolution = (data
                      .get("rcsb_entry_info", {})
                      .get("resolution_combined", [None])[0])
        return float(resolution) if resolution else None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# 2. Выбираем лучшую структуру
# ════════════════════════════════════════════════════════════════════════════

def select_best_structure(uniprot_id: str) -> dict:
    """
    Читает PLIP summary CSV и выбирает лучшую структуру по:
    1. n_interactions (больше = лучше)
    2. resolution (меньше = лучше)
    """
    csv_path = PLIP_DIR / f"{uniprot_id}_plip_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"PLIP summary not found: {csv_path}\n"
            f"Run plip_analysis.py --uniprot {uniprot_id} first!"
        )

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("PLIP summary is empty — no binding sites found!")

    # Сортируем по количеству взаимодействий
    df_sorted = df.sort_values("n_interactions", ascending=False)

    print(f"\nTop 10 binding sites by interactions:")
    print(df_sorted[["pdb_id", "ligand_id", "n_interactions",
                      "n_hbonds", "n_hydrophobic"]]
          .head(10).to_string(index=False))

    # Для топ-5 проверяем разрешение
    print(f"\nChecking resolution for top candidates...")
    candidates = []

    for _, row in df_sorted.head(5).iterrows():
        pdb_id = row["pdb_id"]
        resolution = get_resolution(pdb_id)
        log.info(f"  {pdb_id}: {row['n_interactions']} interactions, "
                 f"resolution={resolution}Å")
        candidates.append({
            "pdb_id":        pdb_id,
            "ligand_id":     row["ligand_id"],
            "n_interactions": row["n_interactions"],
            "n_hbonds":      row["n_hbonds"],
            "n_hydrophobic": row["n_hydrophobic"],
            "pocket_center_x": row["pocket_center_x"],
            "pocket_center_y": row["pocket_center_y"],
            "pocket_center_z": row["pocket_center_z"],
            "resolution":    resolution,
        })

    # Выбираем лучший — максимум взаимодействий,
    # при равенстве — минимальное разрешение
    best = sorted(
        candidates,
        key=lambda x: (
            -x["n_interactions"],
            x["resolution"] if x["resolution"] else 99.0
        )
    )[0]

    return best


# ════════════════════════════════════════════════════════════════════════════
# 3. Копируем структуру в target/
# ════════════════════════════════════════════════════════════════════════════

def copy_target(best: dict, uniprot_id: str) -> Path:
    """
    Копирует лучшую структуру в target/{PDB_ID}_target.pdb
    """
    pdb_id = best["pdb_id"]
    src_path = BASE_DIR / "structures" / uniprot_id / f"{pdb_id}.pdb"

    if not src_path.exists():
        raise FileNotFoundError(f"PDB file not found: {src_path}")

    dst_path = TARGET_DIR / f"{pdb_id}_target.pdb"
    shutil.copy(src_path, dst_path)

    log.info(f"Target structure copied → {dst_path}")
    return dst_path


# ════════════════════════════════════════════════════════════════════════════
# 4. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(uniprot_id: str):
    uniprot_id = uniprot_id.upper().strip()

    print("=" * 60)
    print(f"  Target Structure Selection")
    print(f"  UniProt: {uniprot_id}")
    print("=" * 60)

    # Выбираем лучшую структуру
    best = select_best_structure(uniprot_id)

    # Копируем в target/
    target_path = copy_target(best, uniprot_id)

    # Итог
    print(f"\n{'='*60}")
    print(f"  Selected target: {best['pdb_id']}")
    print(f"  Ligand:          {best['ligand_id']}")
    print(f"  Interactions:    {best['n_interactions']}")
    print(f"    H-bonds:       {best['n_hbonds']}")
    print(f"    Hydrophobic:   {best['n_hydrophobic']}")
    print(f"  Resolution:      {best['resolution']}Å")
    print(f"  Pocket center:   ({best['pocket_center_x']}, "
          f"{best['pocket_center_y']}, {best['pocket_center_z']})")
    print(f"  Saved to:        {target_path}")
    print(f"{'='*60}")

    return best, target_path


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select best target structure for molecule generation"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        required=True,
        help="UniProt accession, e.g. P12931"
    )
    args = parser.parse_args()
    run(args.uniprot)
