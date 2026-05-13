"""
run_docking.py
==============
Запускает AutoDock Vina для всех SDF молекул.

Входные данные:
    receptor.pdbqt              ← белок
    outputs/{run}/sdf/*.sdf     ← лиганды

Выход:
    docking/{run}/
        {mol_id}_docked.pdbqt  ← позы докинга
        docking_scores.csv      ← таблица скоров

Запуск:
    python run_docking.py --run 1O49_baseline_35
    python run_docking.py --run 1O49_guided_35
"""

import argparse
import subprocess
import pandas as pd
from pathlib import Path

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"
DOCKING_DIR = BASE_DIR / "docking"
RECEPTOR    = BASE_DIR / "receptor.pdbqt"

# ── Центр кармана из PLIP (1O49) ──────────────────────────────────────────
CENTER_X = 17.755
CENTER_Y = 22.501
CENTER_Z = 20.273
SIZE_X   = 20.0
SIZE_Y   = 20.0
SIZE_Z   = 20.0


def sdf_to_pdbqt(sdf_path: Path, out_path: Path) -> bool:
    """Конвертирует SDF → PDBQT через obabel."""
    try:
        result = subprocess.run([
            "obabel", str(sdf_path),
            "-O", str(out_path),
            "--gen3d", "--partialcharge", "gasteiger"
        ], capture_output=True, text=True, timeout=30)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception:
        return False


def run_vina(ligand_path: Path, out_path: Path) -> float:
    """Запускает Vina и возвращает лучший скор (ккал/моль)."""
    try:
        result = subprocess.run([
            "vina",
            "--receptor", str(RECEPTOR),
            "--ligand",   str(ligand_path),
            "--out",      str(out_path),
            "--center_x", str(CENTER_X),
            "--center_y", str(CENTER_Y),
            "--center_z", str(CENTER_Z),
            "--size_x",   str(SIZE_X),
            "--size_y",   str(SIZE_Y),
            "--size_z",   str(SIZE_Z),
            "--exhaustiveness", "8",
            "--num_modes", "1",
            "--cpu", "4",
        ], capture_output=True, text=True, timeout=120)

        # Парсим скор из вывода Vina
        for line in result.stdout.split("\n"):
            if line.strip().startswith("1 "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    return float(parts[1])
    except Exception:
        pass
    return None


def run(run_name: str):
    sdf_dir = OUTPUTS_DIR / run_name / "sdf"
    if not sdf_dir.exists():
        raise FileNotFoundError(f"SDF dir not found: {sdf_dir}")

    if not RECEPTOR.exists():
        raise FileNotFoundError(
            f"Receptor not found: {RECEPTOR}\n"
            f"Run: obabel ../small_molecules/target/1O49_target.pdb "
            f"-O receptor.pdbqt -xr --partialcharge gasteiger"
        )

    # Создаём папки
    run_dir  = DOCKING_DIR / run_name
    lig_dir  = run_dir / "ligands_pdbqt"
    pose_dir = run_dir / "poses"
    run_dir.mkdir(parents=True, exist_ok=True)
    lig_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)

    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    print(f"{'='*55}")
    print(f"  Docking: {run_name}")
    print(f"  Molecules: {len(sdf_files)}")
    print(f"  Center: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
    print(f"  Box: {SIZE_X}×{SIZE_Y}×{SIZE_Z} Å")
    print(f"{'='*55}")

    records = []
    failed  = []

    for i, sdf_path in enumerate(sdf_files):
        mol_id   = sdf_path.stem
        lig_path = lig_dir  / f"{mol_id}.pdbqt"
        out_path = pose_dir / f"{mol_id}_docked.pdbqt"

        print(f"[{i+1}/{len(sdf_files)}] {mol_id}... ", end="", flush=True)

        # Достаём SMILES из SDF
        smiles = None
        try:
            from rdkit import Chem
            mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True)
            if mol:
                smiles = Chem.MolToSmiles(mol)
        except Exception:
            pass

        # Шаг 1: SDF → PDBQT
        if not sdf_to_pdbqt(sdf_path, lig_path):
            print("✗ (conversion failed)")
            failed.append(mol_id)
            continue

        # Шаг 2: Vina докинг
        score = run_vina(lig_path, out_path)

        if score is not None:
            print(f"✓ score={score:.2f} kcal/mol")
            records.append({
                "mol_id":        mol_id,
                "smiles":        smiles,
                "docking_score": score,
                "target":        "1O49",
            })
        else:
            print("✗ (docking failed)")
            failed.append(mol_id)

    # Сохраняем результаты
    df = pd.DataFrame(records)
    csv_path = run_dir / "docking_scores.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"Done!")
    print(f"  Success: {len(records)} / {len(sdf_files)}")
    print(f"  Failed:  {len(failed)}")
    if len(records) > 0:
        print(f"  Best score:  {df['docking_score'].min():.2f} kcal/mol")
        print(f"  Mean score:  {df['docking_score'].mean():.2f} kcal/mol")
        print(f"  Std:         {df['docking_score'].std():.2f}")
    print(f"  Saved → {csv_path}")
    print(f"{'='*55}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AutoDock Vina for all molecules in a run"
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Run name (subfolder in outputs/)")
    args = parser.parse_args()
    run(args.run)
