"""
plip_analysis.py
================
Анализ карманов всех PDB структур через PLIP.
Вытаскивает координаты взаимодействий для каждого комплекса
белок-лиганд и сохраняет в структурированном виде.

Входные данные:
    structures/{UNIPROT_ID}/*.pdb   ← скачанные PDB структуры

Выходные данные:
    plip_results/
    ├── {UNIPROT_ID}_plip_summary.csv   ← сводная таблица всех взаимодействий
    └── {PDB_ID}_{LIGAND_ID}/
        ├── report.txt                  ← текстовый отчёт PLIP
        └── interactions.json           ← координаты для TargetDiff

Запуск:
    python plip_analysis.py --uniprot P12931
"""

import json
import logging
import argparse
import pandas as pd
from pathlib import Path

from plip.structure.preparation import PDBComplex

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PLIP_DIR     = BASE_DIR / "plip_results"
PLIP_DIR.mkdir(exist_ok=True)

# Лиганды которые не анализируем
EXCLUDE_LIGANDS = {
    'HOH', 'WAT', 'H2O', 'DOD',
    'NA', 'CL', 'MG', 'CA', 'ZN', 'FE', 'MN', 'K', 'CO',
    'NI', 'CU', 'CD', 'BR', 'I',
    'SO4', 'PO4', 'NO3', 'ACT', 'FMT', 'AZI', 'CIT',
    'GOL', 'EDO', 'PEG', 'PGE', 'PE4', 'PE3', 'MPD',
    'DMS', 'BME', 'BTB', 'EOH', 'MES', 'TRS', 'EPE',
    'IPA', 'IMD', 'P6G',
    'PTR', 'SEP', 'TPO', 'HYP', 'MLY', 'ALY',
    'HPS', 'ATP', 'ADP', 'AMP', 'GTP', 'GDP',
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Анализ одной PDB структуры
# ════════════════════════════════════════════════════════════════════════════

def analyze_complex(pdb_path: Path) -> list:
    """
    Запускает PLIP на одном PDB файле.
    Возвращает список dict с взаимодействиями для каждого лиганда.
    """
    pdb_id = pdb_path.stem.upper()
    results = []

    try:
        # Загружаем комплекс через PLIP
        mol = PDBComplex()
        mol.load_pdb(str(pdb_path))
        mol.analyze()

        # Перебираем все сайты связывания
        for binding_site_id, binding_site in mol.interaction_sets.items():

            ligand_id = binding_site_id.split(":")[0]

            # Пропускаем буферы и воду
            if ligand_id in EXCLUDE_LIGANDS:
                continue

            log.info(f"  {pdb_id}: analyzing {binding_site_id}")

            # ── Собираем все взаимодействия ──────────────────────────────

            # Водородные связи
            hbonds = []
            for hb in (binding_site.hbonds_ldon +
                       binding_site.hbonds_pdon):
                hbonds.append({
                    "type":        "hbond",
                    "dist":        round(hb.distance_ah, 2),
                    "donor_atom":  hb.d_orig_idx,
                    "accept_atom": hb.a_orig_idx,
                    # hb.d, hb.a, hb.h — объекты атомов с координатами
                    "d_coords":    list(hb.d.coords),
                    "a_coords":    list(hb.a.coords),
                    "h_coords":    list(hb.h.coords),
                    "protein_res": hb.restype + str(hb.resnr),
                    "protisdon":   hb.protisdon,
                })

            # Гидрофобные контакты
            hydrophobic = []
            for hc in binding_site.hydrophobic_contacts:
                hydrophobic.append({
                    "type":        "hydrophobic",
                    "dist":        round(hc.distance, 2),
                    "lig_coords":  list(hc.ligatom.coords),
                    "prot_coords": list(hc.bsatom.coords),
                    "protein_res": hc.restype + str(hc.resnr),
                })

            # Pi-стекинг
            pi_stacking = []
            for ps in binding_site.pistacking:
                pi_stacking.append({
                    "type":        "pi_stacking",
                    "dist":        round(ps.distance, 2),
                    "angle":       round(ps.angle, 2),
                    "lig_center":  list(ps.ligandring.center),
                    "prot_center": list(ps.proteinring.center),
                    "protein_res": ps.restype + str(ps.resnr),
                })

            # Солевые мостики
            salt_bridges = []
            for sb in (binding_site.saltbridge_lneg +
                       binding_site.saltbridge_pneg):
                salt_bridges.append({
                    "type":        "salt_bridge",
                    "dist":        round(sb.distance, 2),
                    "lig_coords":  list(sb.negative.center),
                    "prot_coords": list(sb.positive.center),
                    "protein_res": sb.restype + str(sb.resnr),
                })

            # Pi-катионные взаимодействия
            pi_cation = []
            for pc in (binding_site.pication_laro +
                       binding_site.pication_paro):
                pi_cation.append({
                    "type":        "pi_cation",
                    "dist":        round(pc.distance, 2),
                    "ring_center": list(pc.ring.center),
                    "protein_res": pc.restype + str(pc.resnr),
                })

            # ── Координаты кармана ───────────────────────────────────────
            # Центр кармана — среднее по всем точкам взаимодействия
            all_coords = []

            for hb in (binding_site.hbonds_ldon +
                       binding_site.hbonds_pdon):
                all_coords.append(list(hb.d.coords))
                all_coords.append(list(hb.a.coords))

            for hc in binding_site.hydrophobic_contacts:
                all_coords.append(list(hc.ligatom.coords))
                all_coords.append(list(hc.bsatom.coords))

            for ps in binding_site.pistacking:
                all_coords.append(list(ps.ligandring.center))

            pocket_center = None
            if all_coords:
                pocket_center = [
                    round(sum(c[i] for c in all_coords) / len(all_coords), 3)
                    for i in range(3)
                ]

            # ── Итоговый результат для этого сайта ──────────────────────
            result = {
                "pdb_id":          pdb_id,
                "ligand_id":       ligand_id,
                "binding_site_id": binding_site_id,
                "pocket_center":   pocket_center,
                "n_hbonds":        len(hbonds),
                "n_hydrophobic":   len(hydrophobic),
                "n_pi_stacking":   len(pi_stacking),
                "n_salt_bridges":  len(salt_bridges),
                "n_pi_cation":     len(pi_cation),
                "n_interactions":  (len(hbonds) + len(hydrophobic) +
                                    len(pi_stacking) + len(salt_bridges) +
                                    len(pi_cation)),
                "interactions": {
                    "hbonds":       hbonds,
                    "hydrophobic":  hydrophobic,
                    "pi_stacking":  pi_stacking,
                    "salt_bridges": salt_bridges,
                    "pi_cation":    pi_cation,
                }
            }

            results.append(result)

    except Exception as e:
        log.warning(f"  {pdb_id}: PLIP failed — {e}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# 2. Сохранение результатов
# ════════════════════════════════════════════════════════════════════════════

def save_results(all_results: list, uniprot_id: str):
    """
    Сохраняет:
    1. Сводную CSV таблицу
    2. JSON с координатами для каждого комплекса
    """
    if not all_results:
        log.warning("No results to save!")
        return

    # ── CSV сводная таблица ──────────────────────────────────────────────
    csv_rows = []
    for r in all_results:
        csv_rows.append({
            "pdb_id":         r["pdb_id"],
            "ligand_id":      r["ligand_id"],
            "binding_site":   r["binding_site_id"],
            "pocket_center_x": r["pocket_center"][0] if r["pocket_center"] else None,
            "pocket_center_y": r["pocket_center"][1] if r["pocket_center"] else None,
            "pocket_center_z": r["pocket_center"][2] if r["pocket_center"] else None,
            "n_hbonds":        r["n_hbonds"],
            "n_hydrophobic":   r["n_hydrophobic"],
            "n_pi_stacking":   r["n_pi_stacking"],
            "n_salt_bridges":  r["n_salt_bridges"],
            "n_pi_cation":     r["n_pi_cation"],
            "n_interactions":  r["n_interactions"],
        })

    csv_path = PLIP_DIR / f"{uniprot_id}_plip_summary.csv"
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    log.info(f"Summary saved → {csv_path}")

    # ── JSON с координатами для каждого комплекса ─────────────────────────
    for r in all_results:
        name = f"{r['pdb_id']}_{r['ligand_id']}"
        out_dir = PLIP_DIR / name
        out_dir.mkdir(exist_ok=True)

        json_path = out_dir / "interactions.json"
        with open(json_path, 'w') as f:
            json.dump(r, f, indent=2)

    log.info(f"Individual JSON files saved → {PLIP_DIR}/*/interactions.json")

    return csv_path, df


# ════════════════════════════════════════════════════════════════════════════
# 3. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(uniprot_id: str):
    uniprot_id = uniprot_id.upper().strip()

    # Папка со структурами
    structures_dir = BASE_DIR / "structures" / uniprot_id

    if not structures_dir.exists():
        raise FileNotFoundError(
            f"Structures directory not found: {structures_dir}\n"
            f"Run fetch_structures.py --uniprot {uniprot_id} first!"
        )

    pdb_files = sorted(structures_dir.glob("*.pdb"))

    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {structures_dir}")

    print("=" * 60)
    print(f"  PLIP Analysis")
    print(f"  UniProt: {uniprot_id}")
    print(f"  Structures: {len(pdb_files)}")
    print("=" * 60)

    all_results = []

    for i, pdb_path in enumerate(pdb_files):
        log.info(f"[{i+1}/{len(pdb_files)}] {pdb_path.name}")
        results = analyze_complex(pdb_path)
        all_results.extend(results)
        log.info(f"  → {len(results)} binding site(s) found")

    print(f"\n{'='*60}")
    print(f"Total binding sites analyzed: {len(all_results)}")
    print(f"{'='*60}")

    if all_results:
        csv_path, df = save_results(all_results, uniprot_id)

        print(f"\nTop binding sites by number of interactions:")
        top = df.nlargest(5, 'n_interactions')[
            ['pdb_id', 'ligand_id', 'n_interactions',
             'n_hbonds', 'n_hydrophobic']
        ]
        print(top.to_string(index=False))

        print(f"\n{'='*60}")
        print(f"Results saved:")
        print(f"  Summary CSV → {csv_path}")
        print(f"  JSON files  → {PLIP_DIR}/*/interactions.json")
        print(f"{'='*60}")

    return all_results


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PLIP analysis on downloaded PDB structures"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        required=True,
        help="UniProt accession, e.g. P12931"
    )
    args = parser.parse_args()
    run(args.uniprot)
