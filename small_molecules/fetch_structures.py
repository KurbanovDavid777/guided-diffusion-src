"""
fetch_structures.py
===================
Универсальный скрипт для получения PDB структур по UniProt ID.
Используется для получения референсных структур (белок + лиганд)
для последующего анализа карманов через PLIP.

Структура папок (создаётся автоматически рядом со скриптом):
    small_molecules/
    ├── fetch_structures.py
    ├── data/
    │   └── {UNIPROT_ID}_inhibitors.txt   ← список PDB ID + лигандов
    └── structures/
        └── {UNIPROT_ID}/                 ← PDB файлы для PLIP
            ├── 2SRC.pdb
            └── ...

Запуск:
    python fetch_structures.py --uniprot P12931
    python fetch_structures.py --uniprot O14965   # Aurora kinase A
    python fetch_structures.py --uniprot P00533   # EGFR
"""

import argparse
import requests
import random
import logging
from pathlib import Path
from time import sleep

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Пути (относительные — рядом со скриптом) ───────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data"

# ── Константы ──────────────────────────────────────────────────────────────
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
MW_MIN     = 300.0    # Da — нижняя граница для drug-like лигандов
MW_MAX     = 1000.0   # Da — верхняя граница

EXCLUDE_LIGANDS = {
    # Вода и ионы
    'HOH', 'WAT', 'H2O', 'DOD',
    'NA', 'CL', 'MG', 'CA', 'ZN', 'FE', 'MN', 'K', 'CO',
    'NI', 'CU', 'CD', 'BR', 'I',
    # Анионы и буферы
    'SO4', 'PO4', 'NO3', 'ACT', 'FMT', 'AZI', 'CIT',
    # Растворители
    'GOL', 'EDO', 'PEG', 'PGE', 'PE4', 'PE3', 'MPD',
    'DMS', 'BME', 'BTB', 'EOH', 'MES', 'TRS', 'EPE',
    'IPA', 'IMD', 'P6G',
    # Модифицированные аминокислоты
    'PTR', 'SEP', 'TPO', 'HYP', 'MLY', 'ALY',
    # Нуклеотиды и кофакторы
    'HPS', 'ATP', 'ADP', 'AMP', 'GTP', 'GDP',
}


# ════════════════════════════════════════════════════════════════════════════
# 1. Поиск PDB структур по UniProt ID
# ════════════════════════════════════════════════════════════════════════════

def search_pdb_by_uniprot(uniprot_id: str) -> list:
    """Возвращает список всех PDB ID для данного UniProt accession."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers"
                             ".reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry",
        "request_options": {"return_all_hits": True}
    }

    log.info(f"Searching PDB for UniProt {uniprot_id} ...")
    resp = requests.post(SEARCH_URL, json=query, timeout=30)
    resp.raise_for_status()

    pdb_ids = [r["identifier"] for r in resp.json()["result_set"]]
    log.info(f"Found {len(pdb_ids)} total structures")
    return pdb_ids


# ════════════════════════════════════════════════════════════════════════════
# 2. Фильтрация — оставляем структуры с drug-like лигандами
# ════════════════════════════════════════════════════════════════════════════

def filter_drug_like(pdb_ids: list) -> dict:
    """
    Для каждого PDB ID проверяет наличие drug-like лиганда.
    Берём ВСЕ структуры с подходящим лигандом — координаты
    есть у всех кристаллографических структур.
    Возвращает dict: {pdb_id: [{ligand_id, mol_weight, name, smiles}]}
    """
    drug_like = {}
    log.info(f"Filtering {len(pdb_ids)} structures for drug-like ligands ...")

    for i, pdb_id in enumerate(pdb_ids):
        if i % 20 == 0:
            log.info(f"  Progress: {i}/{len(pdb_ids)}")

        try:
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            entry_data = requests.get(entry_url, timeout=10).json()

            # Берём все non-polymer компоненты (лиганды)
            # НЕ фильтруем по rcsb_binding_affinity —
            # координаты есть у всех структур
            entity_ids = (entry_data
                          .get("rcsb_entry_container_identifiers", {})
                          .get("non_polymer_entity_ids", []))

            if not entity_ids:
                sleep(0.05)
                continue

            # Собираем comp_id для каждого non-polymer entity
            comp_ids = set()
            for eid in entity_ids:
                entity_url = (
                    f"https://data.rcsb.org/rest/v1/core"
                    f"/nonpolymer_entity/{pdb_id}/{eid}"
                )
                try:
                    entity_data = requests.get(entity_url, timeout=10).json()
                    comp_id = (entity_data
                               .get("pdbx_entity_nonpoly", {})
                               .get("comp_id"))
                    if comp_id:
                        comp_ids.add(comp_id)
                except Exception:
                    continue

            for comp_id in comp_ids:
                if comp_id in EXCLUDE_LIGANDS:
                    continue

                comp_url = (f"https://data.rcsb.org/rest/v1"
                            f"/core/chemcomp/{comp_id}")
                comp_data = requests.get(comp_url, timeout=10).json()

                formula_weight = (comp_data
                                  .get('chem_comp', {})
                                  .get('formula_weight'))
                comp_type = comp_data.get('chem_comp', {}).get('type', '')

                if not formula_weight:
                    continue

                formula_weight = float(formula_weight)

                if (MW_MIN < formula_weight < MW_MAX
                        and 'PEPTIDE'    not in comp_type.upper()
                        and 'AMINO ACID' not in comp_type.upper()):

                    # Достаём SMILES из RCSB
                    smiles = None
                    descriptors = comp_data.get(
                        'rcsb_chem_comp_descriptor', []
                    )
                    if isinstance(descriptors, list):
                        for d in descriptors:
                            if d.get('type') in (
                                'SMILES', 'SMILES_CANONICAL',
                                'OpenEye OEToolkits'
                            ):
                                smiles = d.get('descriptor')
                                if smiles:
                                    break
                    elif isinstance(descriptors, dict):
                        smiles = (descriptors.get('smiles_stereo') or
                                  descriptors.get('smiles'))

                    # Fallback на PubChem
                    if not smiles:
                        try:
                            pc_url = (
                                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug"
                                f"/compound/name/{comp_id}"
                                f"/property/IsomericSMILES/JSON"
                            )
                            pc_data = requests.get(pc_url, timeout=10).json()
                            props = (pc_data
                                     .get("PropertyTable", {})
                                     .get("Properties", []))
                            if props:
                                smiles = props[0].get("IsomericSMILES")
                        except Exception:
                            pass

                    comp_name = (comp_data
                                 .get('chem_comp', {})
                                 .get('name', comp_id))

                    if pdb_id not in drug_like:
                        drug_like[pdb_id] = []

                    drug_like[pdb_id].append({
                        'ligand_id':  comp_id,
                        'mol_weight': formula_weight,
                        'name':       comp_name,
                        'smiles':     smiles,
                        'type':       comp_type,
                    })

            sleep(0.1)

        except Exception as e:
            log.debug(f"  {pdb_id}: {e}")
            continue

    log.info(f"Found {len(drug_like)} structures with drug-like ligands")
    return drug_like


# ════════════════════════════════════════════════════════════════════════════
# 3. Сохранение списка PDB ID + лигандов
# ════════════════════════════════════════════════════════════════════════════

def save_inhibitors_list(drug_like: dict, uniprot_id: str) -> Path:
    """Сохраняет таблицу в data/{uniprot_id}_inhibitors.txt"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{uniprot_id}_inhibitors.txt"
    pdb_ids  = sorted(drug_like.keys())

    with open(out_path, 'w') as f:
        f.write("pdb_id\tligand_id\tmol_weight\tname\tsmiles\n")
        for pdb_id in pdb_ids:
            for lig in drug_like[pdb_id]:
                f.write(
                    f"{pdb_id}\t"
                    f"{lig['ligand_id']}\t"
                    f"{lig['mol_weight']:.1f}\t"
                    f"{lig['name']}\t"
                    f"{lig['smiles'] or 'N/A'}\n"
                )

    log.info(f"Saved {len(pdb_ids)} entries → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 4. Скачивание PDB файлов
# ════════════════════════════════════════════════════════════════════════════

def download_pdb_structures(pdb_ids: list, uniprot_id: str) -> dict:
    """
    Скачивает PDB файлы в structures/{uniprot_id}/.
    Пропускает уже скачанные.
    """
    structures_dir = BASE_DIR / "structures" / uniprot_id
    structures_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}
    failed     = []

    log.info(f"Downloading {len(pdb_ids)} PDB structures → {structures_dir}/")

    for pdb_id in pdb_ids:
        out_path = structures_dir / f"{pdb_id}.pdb"

        # Пропускаем уже скачанные
        if out_path.exists():
            log.info(f"  ⊙ {pdb_id}.pdb (cached)")
            downloaded[pdb_id] = out_path
            continue

        success = False

        # Два источника: PDBe и RCSB
        urls = [
            f"https://www.ebi.ac.uk/pdbe/entry-files/download/pdb{pdb_id.lower()}.ent",
            f"https://files.rcsb.org/download/{pdb_id}.pdb",
        ]

        for attempt in range(5):
            for url in urls:
                try:
                    resp = requests.get(url, timeout=60)
                    if resp.status_code == 200:
                        content = resp.text
                        if "ATOM" in content or "HEADER" in content:
                            out_path.write_text(content)
                            log.info(f"  ✓ {pdb_id}.pdb "
                                     f"({len(content)//1024} KB)")
                            downloaded[pdb_id] = out_path
                            success = True
                            break
                except Exception:
                    pass

            if success:
                break

            wait = 2 + attempt * 2
            log.warning(f"  ⟳ {pdb_id} retry {attempt+1}/5 "
                        f"(wait {wait}s)")
            sleep(wait)

        if not success:
            log.error(f"  ✗ {pdb_id} — failed")
            failed.append(pdb_id)

        sleep(random.uniform(0.3, 0.8))

    log.info(f"\nDownload summary:")
    log.info(f"  Successful : {len(downloaded)}")
    log.info(f"  Failed     : {len(failed)}")
    if failed:
        log.warning(f"  Failed IDs : {failed}")

    return downloaded


# ════════════════════════════════════════════════════════════════════════════
# 5. Главная функция
# ════════════════════════════════════════════════════════════════════════════

def run(uniprot_id: str):
    print("=" * 60)
    print(f"  PDB Structure Fetcher")
    print(f"  UniProt: {uniprot_id}")
    print("=" * 60)

    # Шаг 1: Поиск
    all_pdb_ids = search_pdb_by_uniprot(uniprot_id)

    # Шаг 2: Фильтрация
    drug_like = filter_drug_like(all_pdb_ids)

    # Шаг 3: Печатаем результаты
    print(f"\n{'='*60}")
    print(f"Structures with drug-like ligands: {len(drug_like)}")
    print(f"{'='*60}")
    for pdb_id, ligands in sorted(drug_like.items()):
        for lig in ligands:
            print(f"  {pdb_id}: {lig['ligand_id']} "
                  f"({lig['mol_weight']:.0f} Da) — {lig['name']}")

    # Шаг 4: Сохраняем список
    inhibitors_file = save_inhibitors_list(drug_like, uniprot_id)

    # Шаг 5: Скачиваем PDB файлы
    pdb_ids    = sorted(drug_like.keys())
    downloaded = download_pdb_structures(pdb_ids, uniprot_id)

    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  PDB structures → structures/{uniprot_id}/")
    print(f"  Structure list → {inhibitors_file}")
    print(f"  Total downloaded: {len(downloaded)}")
    print(f"{'='*60}")

    return downloaded, drug_like


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch PDB structures by UniProt ID"
    )
    parser.add_argument(
        "--uniprot",
        type=str,
        required=True,
        help="UniProt accession, e.g. P12931 (Src), P00533 (EGFR)"
    )
    args = parser.parse_args()
    run(args.uniprot.upper().strip())
