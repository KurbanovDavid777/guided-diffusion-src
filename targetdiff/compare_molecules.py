"""
compare_molecules.py
====================
Сравнивает baseline и guided молекулы по ключевым метрикам.

Метрики:
  - Валидность (% валидных SMILES)
  - QED (drug-likeness)
  - SAScore (синтезируемость)
  - MW (молекулярный вес)
  - Tanimoto к референсам (новизна) ← главный тест guidance

Запуск:
    python compare_molecules.py \
        --baseline results/1O49_baseline.csv \
        --guided   results/1O49_guided.csv \
        --refs     ../small_molecules/reference_mol/P12931_references.smi
"""

import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# SAScore
try:
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SA = True
except ImportError:
    try:
        import sys
        sys.path.append('/Users/davidkurbanov/miniconda3/envs/targetdiff/share/RDKit/Contrib/SA_Score')
        import sascorer
        HAS_SA = True
    except ImportError:
        HAS_SA = False

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)


# ════════════════════════════════════════════════════════════════════════════
# 1. Вычисление метрик для одной молекулы
# ════════════════════════════════════════════════════════════════════════════

def compute_metrics(smiles: str, ref_fps: list) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Базовые дескрипторы
    mw  = Descriptors.MolWt(mol)
    qed = QED.qed(mol)
    sa  = sascorer.calculateScore(mol) if HAS_SA else 0.0
    fp  = morgan_gen.GetFingerprint(mol)

    # Tanimoto к референсам
    tanimotos = [DataStructs.TanimotoSimilarity(fp, ref_fp)
                 for ref_fp in ref_fps]
    max_tanimoto = max(tanimotos) if tanimotos else 0.0
    mean_tanimoto = np.mean(tanimotos) if tanimotos else 0.0

    # Lipinski
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    logp = Descriptors.MolLogP(mol)
    lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)

    return {
        "mw":            round(mw, 2),
        "qed":           round(qed, 3),
        "sa_score":      round(sa, 3),
        "logp":          round(logp, 3),
        "max_tanimoto":  round(max_tanimoto, 3),
        "mean_tanimoto": round(mean_tanimoto, 3),
        "lipinski":      lipinski,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Загрузка референсов
# ════════════════════════════════════════════════════════════════════════════

def load_ref_fps(refs_path: Path) -> list:
    fps = []
    for line in refs_path.read_text().strip().split("\n"):
        smi = line.strip()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(morgan_gen.GetFingerprint(mol))
    print(f"Loaded {len(fps)} reference fingerprints")
    return fps


# ════════════════════════════════════════════════════════════════════════════
# 3. Обработка набора молекул
# ════════════════════════════════════════════════════════════════════════════

def process_set(csv_path: Path, ref_fps: list, name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"\nProcessing {name}: {len(df)} molecules...")

    records = []
    for _, row in df.iterrows():
        smi = row.get("smiles")
        if not smi or pd.isna(smi):
            continue
        metrics = compute_metrics(str(smi), ref_fps)
        if metrics:
            metrics["smiles"] = smi
            metrics["set"]    = name
            records.append(metrics)

    result = pd.DataFrame(records)
    print(f"  Valid: {len(result)} / {len(df)}")
    return result


# ════════════════════════════════════════════════════════════════════════════
# 4. Сравнение и вывод
# ════════════════════════════════════════════════════════════════════════════

def compare(baseline_df: pd.DataFrame, guided_df: pd.DataFrame):
    metrics = ["qed", "sa_score", "mw", "max_tanimoto",
               "mean_tanimoto", "logp"]

    print(f"\n{'='*65}")
    print(f"{'Метрика':20} {'Baseline':>15} {'Guided':>15} {'Лучше':>10}")
    print(f"{'='*65}")

    for m in metrics:
        b_val = baseline_df[m].mean()
        g_val = guided_df[m].mean()

        # Определяем кто лучше
        if m in ["qed", "logp"]:
            better = "Guided ✓" if g_val > b_val else "Baseline"
        elif m in ["sa_score", "max_tanimoto", "mean_tanimoto", "mw"]:
            better = "Guided ✓" if g_val < b_val else "Baseline"
        else:
            better = "—"

        print(f"{m:20} {b_val:>15.3f} {g_val:>15.3f} {better:>10}")

    # Lipinski
    b_lip = baseline_df["lipinski"].mean() * 100
    g_lip = guided_df["lipinski"].mean() * 100
    better = "Guided ✓" if g_lip > b_lip else "Baseline"
    print(f"{'lipinski (%)':20} {b_lip:>15.1f} {g_lip:>15.1f} {better:>10}")

    print(f"{'='*65}")
    print(f"{'N молекул':20} {len(baseline_df):>15} {len(guided_df):>15}")
    print(f"{'='*65}")

    # Tanimoto распределение
    print(f"\nTanimoto к референсам (max) — распределение:")
    print(f"{'':20} {'Baseline':>15} {'Guided':>15}")
    print(f"{'< 0.2 (очень новые)':20} "
          f"{(baseline_df['max_tanimoto'] < 0.2).mean()*100:>14.1f}% "
          f"{(guided_df['max_tanimoto'] < 0.2).mean()*100:>14.1f}%")
    print(f"{'0.2-0.4 (новые)':20} "
          f"{((baseline_df['max_tanimoto'] >= 0.2) & (baseline_df['max_tanimoto'] < 0.4)).mean()*100:>14.1f}% "
          f"{((guided_df['max_tanimoto'] >= 0.2) & (guided_df['max_tanimoto'] < 0.4)).mean()*100:>14.1f}%")
    print(f"{'0.4-0.6 (похожие)':20} "
          f"{((baseline_df['max_tanimoto'] >= 0.4) & (baseline_df['max_tanimoto'] < 0.6)).mean()*100:>14.1f}% "
          f"{((guided_df['max_tanimoto'] >= 0.4) & (guided_df['max_tanimoto'] < 0.6)).mean()*100:>14.1f}%")
    print(f"{'> 0.6 (очень похожие)':20} "
          f"{(baseline_df['max_tanimoto'] >= 0.6).mean()*100:>14.1f}% "
          f"{(guided_df['max_tanimoto'] >= 0.6).mean()*100:>14.1f}%")


# ════════════════════════════════════════════════════════════════════════════
# 6. Расчёт аффинности через TargetDiff predictor
# ════════════════════════════════════════════════════════════════════════════

def compute_affinity(sdf_dir: Path, protein_path: Path,
                     ckpt_path: str = "pretrained_models/egnn_pdbbind_v2016.pt",
                     kind: str = "Kd",
                     device: str = "cpu") -> dict:
    """
    Запускает affinity predictor для всех SDF файлов в папке.
    Возвращает dict: {mol_id: pKd}
    """
    sdf_files = sorted(sdf_dir.glob("*.sdf"))
    if not sdf_files:
        return {}

    if not Path(ckpt_path).exists():
        print(f"  Affinity model not found: {ckpt_path}")
        return {}

    print(f"  Computing affinity for {len(sdf_files)} molecules...")
    results = {}

    for sdf_path in sdf_files:
        mol_id = sdf_path.stem
        try:
            cmd = [
                "python",
                "scripts/property_prediction/inference.py",
                "--ckpt_path",    ckpt_path,
                "--protein_path", str(protein_path),
                "--ligand_path",  str(sdf_path),
                "--kind",         kind,
                "--device",       device,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            # Парсим Kd из вывода: "Kd=4.80e-05 m"
            import re
            import math
            for line in result.stdout.split("\n"):
                match = re.search(r'Kd=([\d.e\-+]+)', line)
                if match:
                    kd_val = float(match.group(1))
                    # Конвертируем Kd (в молях) → pKd
                    pkd = -math.log10(kd_val)
                    results[mol_id] = round(pkd, 3)
                    break
        except Exception as e:
            pass

    print(f"  Affinity computed for {len(results)} molecules")
    return results

def run(baseline_path: Path, guided_path: Path, refs_path: Path,
        protein_path: Path = None, sdf_baseline: Path = None,
        sdf_guided: Path = None):
    print("=" * 65)
    print("  Baseline vs Guided — Molecule Comparison")
    print("=" * 65)

    # Загружаем референсы
    ref_fps = load_ref_fps(refs_path)

    # Обрабатываем оба набора
    baseline_df = process_set(baseline_path, ref_fps, "Baseline")
    guided_df   = process_set(guided_path,   ref_fps, "Guided")

    # ── Аффинность (опционально) ──────────────────────────────────────────
    if protein_path and sdf_baseline and sdf_guided:
        print("\nComputing affinity (this may take a while)...")

        b_affinities = compute_affinity(sdf_baseline, protein_path)
        g_affinities = compute_affinity(sdf_guided,   protein_path)

        # Добавляем в датафреймы
        # Читаем SMILES из SDF файлов чтобы сматчить с датафреймом
        def get_smiles_to_pkd(sdf_dir, affinities):
            from rdkit import Chem
            smiles_to_pkd = {}
            for mol_id, pkd in affinities.items():
                sdf_path = sdf_dir / f"{mol_id}.sdf"
                if sdf_path.exists():
                    mol = Chem.MolFromMolFile(str(sdf_path), sanitize=True)
                    if mol:
                        smi = Chem.MolToSmiles(mol)
                        smiles_to_pkd[smi] = pkd
            return smiles_to_pkd

        b_smi_pkd = get_smiles_to_pkd(sdf_baseline, b_affinities)
        g_smi_pkd = get_smiles_to_pkd(sdf_guided,   g_affinities)

        baseline_df["pKd"] = baseline_df["smiles"].map(b_smi_pkd)
        guided_df["pKd"]   = guided_df["smiles"].map(g_smi_pkd)

    # Сравниваем
    compare(baseline_df, guided_df)

    # ── Аффинность в сравнении ────────────────────────────────────────────
    if "pKd" in baseline_df.columns and "pKd" in guided_df.columns:
        b_pkd = baseline_df["pKd"].dropna()
        g_pkd = guided_df["pKd"].dropna()
        if len(b_pkd) > 0 and len(g_pkd) > 0:
            better = "Guided ✓" if g_pkd.mean() > b_pkd.mean() else "Baseline"
            print(f"\npKd (аффинность):")
            print(f"  Baseline: {b_pkd.mean():.3f} ± {b_pkd.std():.3f} "
                  f"(n={len(b_pkd)})")
            print(f"  Guided:   {g_pkd.mean():.3f} ± {g_pkd.std():.3f} "
                  f"(n={len(g_pkd)})")
            print(f"  Лучше:    {better}")

    # Сохраняем
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    baseline_df.to_csv(out_dir / "1O49_baseline_scored.csv", index=False)
    guided_df.to_csv(out_dir   / "1O49_guided_scored.csv",   index=False)
    pd.concat([baseline_df, guided_df]).to_csv(
        out_dir / "1O49_combined.csv", index=False
    )
    print(f"\nDetailed results saved → results/")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline vs guided molecules"
    )
    parser.add_argument("--baseline",     type=str,
                        default="results/1O49_baseline.csv")
    parser.add_argument("--guided",       type=str,
                        default="results/1O49_guided.csv")
    parser.add_argument("--refs",         type=str,
                        default="../small_molecules/reference_mol/P12931_references.smi")
    parser.add_argument("--protein",      type=str, default=None,
                        help="Path to pocket PDB for affinity calculation")
    parser.add_argument("--sdf_baseline", type=str, default=None,
                        help="Path to baseline SDF folder")
    parser.add_argument("--sdf_guided",   type=str, default=None,
                        help="Path to guided SDF folder")
    args = parser.parse_args()

    run(
        baseline_path = Path(args.baseline),
        guided_path   = Path(args.guided),
        refs_path     = Path(args.refs),
        protein_path  = Path(args.protein) if args.protein else None,
        sdf_baseline  = Path(args.sdf_baseline) if args.sdf_baseline else None,
        sdf_guided    = Path(args.sdf_guided)   if args.sdf_guided   else None,
    )
