"""
sdf_to_csv.py
=============
Конвертирует SDF файлы из TargetDiff в CSV с SMILES и PDB ID.

PDB ID берётся из названия папки (например 1O49_baseline → 1O49).

Запуск:
    python sdf_to_csv.py --run 1O49_baseline
"""

import argparse
from pathlib import Path
from rdkit import Chem
import pandas as pd

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run(run_name: str):
    # PDB ID из названия папки (берём часть до первого "_")
    pdb_id  = run_name.split("_")[0]
    sdf_dir = OUTPUTS_DIR / run_name / "sdf"

    if not sdf_dir.exists():
        raise FileNotFoundError(f"Not found: {sdf_dir}")

    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    print(f"Found {len(sdf_files)} SDF files | PDB ID: {pdb_id}")

    records = []
    for sdf_path in sdf_files:
        mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True)
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol)
        if smiles:
            records.append({"smiles": smiles, "pdb_id": pdb_id})

    df = pd.DataFrame(records)

    csv_path = RESULTS_DIR / f"{run_name}.csv"
    df.to_csv(csv_path, index=False)

    print(f"Valid molecules: {len(df)} / {len(sdf_files)}")
    print(f"Saved → {csv_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True,
                        help="Folder name in outputs/, e.g. 1O49_baseline")
    args = parser.parse_args()
    run(args.run)
