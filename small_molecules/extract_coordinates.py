"""
extract_coordinates.py
======================
Вытаскивает координаты карманов из PLIP JSON отчётов.
Готовит данные для генерации молекул в TargetDiff.

Входные данные:
    plip_results/*/interactions.json

Выходные данные:
    coordinates/
    ├── {UNIPROT_ID}_pocket_coords.csv   ← все координаты в одном файле
    ├── {UNIPROT_ID}_pocket_centers.csv  ← только центры карманов
    └── {PDB_ID}_{LIGAND_ID}_coords.json ← детальные координаты по кармануот

Формат для TargetDiff:
    pocket_center (x, y, z) + список координат всех взаимодействующих атомов

Запуск:
    python extract_coordinates.py --uniprot P12931
    python extract_coordinates.py --uniprot P12931 --top 10
"""

import json
import logging
import argparse
import pandas as pd
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути ───────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
PLIP_DIR    = BASE_DIR / "plip_results"
COORDS_DIR  = BASE_DIR / "coordinates"
COORDS_DIR.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# 1. Загрузка всех PLIP JSON отчётов
# ════════════════════════════════════════════════════════════════════════════

def load_plip_reports(uniprot_id: str) -> list:
    """
    Загружает все interactions.json из plip_results/
    Возвращает список dict с данными по каждому кармону
    """
    json_files = sorted(PLIP_DIR.glob("*/interactions.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No PLIP reports found in {PLIP_DIR}\n"
            f"Run plip_analysis.py --uniprot {uniprot_id} first!"
        )

    reports = []
    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
            reports.append(data)
        except Exception as e:
            log.warning(f"Failed to load {json_path}: {e}")

    log.info(f"Loaded {len(reports)} PLIP reports")
    return reports


# ════════════════════════════════════════════════════════════════════════════
# 2. Извлечение координат из одного отчёта
# ════════════════════════════════════════════════════════════════════════════

def extract_pocket_coords(report: dict) -> dict:
    """
    Из одного PLIP отчёта вытаскивает все координаты взаимодействий.

    Возвращает dict:
        pdb_id, ligand_id, pocket_center,
        hbond_coords, hydrophobic_coords,
        pi_stacking_coords, salt_bridge_coords,
        all_interaction_coords  ← все точки вместе
    """
    pdb_id    = report["pdb_id"]
    ligand_id = report["ligand_id"]
    interactions = report.get("interactions", {})

    # ── Водородные связи ─────────────────────────────────────────────────
    hbond_coords = []
    for hb in interactions.get("hbonds", []):
        # Берём координаты донора и акцептора
        if hb.get("d_coords"):
            hbond_coords.append({
                "type":        "hbond_donor",
                "coords":      hb["d_coords"],
                "protein_res": hb.get("protein_res", ""),
                "dist":        hb.get("dist"),
            })
        if hb.get("a_coords"):
            hbond_coords.append({
                "type":        "hbond_acceptor",
                "coords":      hb["a_coords"],
                "protein_res": hb.get("protein_res", ""),
                "dist":        hb.get("dist"),
            })

    # ── Гидрофобные контакты ─────────────────────────────────────────────
    hydrophobic_coords = []
    for hc in interactions.get("hydrophobic", []):
        if hc.get("prot_coords"):
            hydrophobic_coords.append({
                "type":        "hydrophobic",
                "coords":      hc["prot_coords"],
                "protein_res": hc.get("protein_res", ""),
                "dist":        hc.get("dist"),
            })

    # ── Pi-стекинг ───────────────────────────────────────────────────────
    pi_stacking_coords = []
    for ps in interactions.get("pi_stacking", []):
        if ps.get("prot_center"):
            pi_stacking_coords.append({
                "type":        "pi_stacking",
                "coords":      ps["prot_center"],
                "protein_res": ps.get("protein_res", ""),
                "dist":        ps.get("dist"),
            })

    # ── Солевые мостики ──────────────────────────────────────────────────
    salt_bridge_coords = []
    for sb in interactions.get("salt_bridges", []):
        if sb.get("prot_coords"):
            salt_bridge_coords.append({
                "type":        "salt_bridge",
                "coords":      sb["prot_coords"],
                "protein_res": sb.get("protein_res", ""),
                "dist":        sb.get("dist"),
            })

    # ── Pi-катионные ─────────────────────────────────────────────────────
    pi_cation_coords = []
    for pc in interactions.get("pi_cation", []):
        if pc.get("ring_center"):
            pi_cation_coords.append({
                "type":        "pi_cation",
                "coords":      pc["ring_center"],
                "protein_res": pc.get("protein_res", ""),
                "dist":        pc.get("dist"),
            })

    # ── Все координаты вместе ─────────────────────────────────────────────
    all_coords = (hbond_coords + hydrophobic_coords +
                  pi_stacking_coords + salt_bridge_coords +
                  pi_cation_coords)

    # ── Только координаты точек (x, y, z) ───────────────────────────────
    raw_coords = [c["coords"] for c in all_coords if c.get("coords")]

    return {
        "pdb_id":             pdb_id,
        "ligand_id":          ligand_id,
        "pocket_center":      report.get("pocket_center"),
        "n_interactions":     report.get("n_interactions", 0),
        "n_hbonds":           report.get("n_hbonds", 0),
        "n_hydrophobic":      report.get("n_hydrophobic", 0),
        "n_pi_stacking":      report.get("n_pi_stacking", 0),
        "n_salt_bridges":     report.get("n_salt_bridges", 0),
        "hbond_coords":       hbond_coords,
        "hydrophobic_coords": hydrophobic_coords,
        "pi_stacking_coords": pi_stacking_coords,
        "salt_bridge_coords": salt_bridge_coords,
        "pi_cation_coords":   pi_cation_coords,
        "all_coords":         all_coords,
        "raw_coords":         raw_coords,  # просто список [x,y,z]
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. Сохранение координат
# ════════════════════════════════════════════════════════════════════════════

def save_coordinates(extracted: list, uniprot_id: str, top_n: int = None):
    """
    Сохраняет координаты в трёх форматах:

    1. pocket_centers.csv  — центры карманов (для быстрого просмотра)
    2. pocket_coords.csv   — все координаты точек взаимодействия
    3. {PDB}_{LIG}_coords.json — детальный JSON для каждого кармана
    """
    # Сортируем по количеству взаимодействий
    extracted_sorted = sorted(
        extracted,
        key=lambda x: x["n_interactions"],
        reverse=True
    )

    # Берём топ-N если указано
    if top_n:
        extracted_sorted = extracted_sorted[:top_n]
        log.info(f"Using top {top_n} binding sites")

    # ── 1. CSV с центрами карманов ────────────────────────────────────────
    centers_rows = []
    for e in extracted_sorted:
        center = e["pocket_center"]
        centers_rows.append({
            "pdb_id":        e["pdb_id"],
            "ligand_id":     e["ligand_id"],
            "n_interactions": e["n_interactions"],
            "n_hbonds":      e["n_hbonds"],
            "n_hydrophobic": e["n_hydrophobic"],
            "center_x":      center[0] if center else None,
            "center_y":      center[1] if center else None,
            "center_z":      center[2] if center else None,
        })

    centers_path = COORDS_DIR / f"{uniprot_id}_pocket_centers.csv"
    pd.DataFrame(centers_rows).to_csv(centers_path, index=False)
    log.info(f"Pocket centers → {centers_path}")

    # ── 2. CSV со всеми координатами точек ───────────────────────────────
    coords_rows = []
    for e in extracted_sorted:
        for point in e["all_coords"]:
            coords = point.get("coords", [])
            if len(coords) == 3:
                coords_rows.append({
                    "pdb_id":        e["pdb_id"],
                    "ligand_id":     e["ligand_id"],
                    "interaction_type": point["type"],
                    "protein_res":   point.get("protein_res", ""),
                    "dist":          point.get("dist"),
                    "x":             coords[0],
                    "y":             coords[1],
                    "z":             coords[2],
                })

    coords_path = COORDS_DIR / f"{uniprot_id}_pocket_coords.csv"
    pd.DataFrame(coords_rows).to_csv(coords_path, index=False)
    log.info(f"All coords     → {coords_path}")

    # ── 3. Детальный JSON для каждого кармана ────────────────────────────
    for e in extracted_sorted:
        name     = f"{e['pdb_id']}_{e['ligand_id']}"
        out_path = COORDS_DIR / f"{name}_coords.json"

        output = {
            "pdb_id":         e["pdb_id"],
            "ligand_id":      e["ligand_id"],
            "pocket_center":  e["pocket_center"],
            "n_interactions": e["n_interactions"],
            "raw_coords":     e["raw_coords"],
            "interactions":   e["all_coords"],
        }

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    log.info(f"Individual JSON → {COORDS_DIR}/*_coords.json")

    return centers_path, coords_path


# ════════════════════════════════════════════════════════════════════════════
# 4. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(uniprot_id: str, top_n: int = None):
    uniprot_id = uniprot_id.upper().strip()

    print("=" * 60)
    print(f"  Pocket Coordinate Extraction")
    print(f"  UniProt: {uniprot_id}")
    print("=" * 60)

    # Шаг 1: Загружаем PLIP отчёты
    reports = load_plip_reports(uniprot_id)

    # Шаг 2: Извлекаем координаты
    extracted = []
    for report in reports:
        coords = extract_pocket_coords(report)
        extracted.append(coords)
        log.info(f"  {coords['pdb_id']}_{coords['ligand_id']}: "
                 f"{len(coords['raw_coords'])} interaction points")

    # Шаг 3: Сохраняем
    centers_path, coords_path = save_coordinates(
        extracted, uniprot_id, top_n
    )

    # Шаг 4: Печатаем топ карманов
    print(f"\nTop binding sites:")
    print(f"{'PDB':6} {'Lig':6} {'N_int':6} {'HB':4} "
          f"{'Hph':4} {'Center (x, y, z)'}")
    print("-" * 60)

    sorted_ext = sorted(
        extracted,
        key=lambda x: x["n_interactions"],
        reverse=True
    )
    for e in (sorted_ext[:top_n] if top_n else sorted_ext):
        c = e["pocket_center"]
        center_str = (f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
                      if c else "N/A")
        print(f"{e['pdb_id']:6} {e['ligand_id']:6} "
              f"{e['n_interactions']:6} {e['n_hbonds']:4} "
              f"{e['n_hydrophobic']:4} {center_str}")

    print(f"\n{'='*60}")
    print(f"Saved:")
    print(f"  Pocket centers → {centers_path}")
    print(f"  All coords     → {coords_path}")
    print(f"  JSON files     → {COORDS_DIR}/*_coords.json")
    print(f"{'='*60}")

    return extracted


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pocket coordinates from PLIP reports"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        required=True,
        help="UniProt accession, e.g. P12931"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Use only top N binding sites (default: all)"
    )
    args = parser.parse_args()
    run(args.uniprot, args.top)
