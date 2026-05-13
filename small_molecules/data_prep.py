"""
data_prep.py
============
Module 1 — Data Preparation Pipeline

Flow:
  PDB ID → fetch protein structure → fetch PDB ligands
         → filter by MW → check Tanimoto diversity
         → (optionally) top-up from ChEMBL
         → save reference_mol/references.csv + target/<pdb_id>.pdb
"""

import os
import time
import logging
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# New RDKit API — avoids DEPRECATION WARNING
_morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Directory layout ────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
REFERENCE_DIR   = BASE_DIR / "reference_mol"
TARGET_DIR      = BASE_DIR / "target"

REFERENCE_DIR.mkdir(exist_ok=True)
TARGET_DIR.mkdir(exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────
MW_MIN              = 150.0   # Da  — removes salts / buffers
MW_MAX              = 700.0   # Da  — removes peptides / macrocycles
MIN_REFERENCES      = 10      # if fewer → top-up from ChEMBL
DIVERSITY_THRESHOLD = 0.7     # avg pairwise Tanimoto above this → not diverse
CHEMBL_PCHEMBL_MIN   = 6.0    # pChEMBL ≥ 6  ↔  IC50 ≤ 1 µM
CHEMBL_FETCH_LIMIT   = 100    # how many to pull before diversity filter
CHEMBL_MAX_DIVERSE   = 50     # max to keep after diversity selection
CHEMBL_DIV_THRESHOLD = 0.4    # max pairwise Tanimoto within selected set


# ════════════════════════════════════════════════════════════════════════════
# 1.  PDB — protein structure
# ════════════════════════════════════════════════════════════════════════════

def fetch_protein_pdb(pdb_id: str) -> Path:
    """
    Download the PDB file for *pdb_id* and save it to target/<pdb_id>.pdb.
    Returns the path to the saved file.
    """
    pdb_id = pdb_id.upper().strip()
    out_path = TARGET_DIR / f"{pdb_id}.pdb"

    if out_path.exists():
        log.info(f"PDB file already cached: {out_path}")
        return out_path

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    log.info(f"Downloading protein structure from {url} ...")
    resp = requests.get(url, timeout=30)

    if resp.status_code == 404:
        raise ValueError(f"PDB ID '{pdb_id}' not found on RCSB.")
    resp.raise_for_status()

    out_path.write_text(resp.text)
    log.info(f"Protein saved → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# 2.  PDB — co-crystallised ligands
# ════════════════════════════════════════════════════════════════════════════

def _fetch_ligand_smiles_from_pdb(comp_id: str) -> Optional[str]:
    """
    Return canonical SMILES for a PDB chemical component.
    Tries two endpoints:
      1. RCSB REST API  (rcsb_chem_comp_descriptor)
      2. PDB CCD via pubchem as fallback
    """
    # — Endpoint 1: RCSB REST API ————————————————————————————————————
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    try:
        data = requests.get(url, timeout=10).json()
        desc = data.get("rcsb_chem_comp_descriptor", {})
        # The API returns a list of descriptor dicts
        if isinstance(desc, list):
            for item in desc:
                if item.get("type") in ("SMILES", "SMILES_CANONICAL", "OpenEye OEToolkits"):
                    smi = item.get("descriptor")
                    if smi:
                        return smi
        elif isinstance(desc, dict):
            smi = (desc.get("smiles_stereo")
                   or desc.get("smiles")
                   or desc.get("descriptor"))
            if smi:
                return smi
    except Exception as e:
        log.debug(f"RCSB chemcomp API failed for {comp_id}: {e}")

    # — Endpoint 2: PubChem CID lookup as fallback ————————————————————
    try:
        pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{comp_id}/property/IsomericSMILES/JSON"
        pc_data = requests.get(pc_url, timeout=10).json()
        props = pc_data.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("IsomericSMILES")
    except Exception as e:
        log.debug(f"PubChem fallback failed for {comp_id}: {e}")

    return None


def fetch_ligands_from_pdb(pdb_id: str) -> list[str]:
    """
    Return a list of SMILES strings for all non-polymer ligands
    co-crystallised with the given PDB entry.
    """
    pdb_id = pdb_id.upper().strip()
    log.info(f"Fetching ligand list for {pdb_id} ...")

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    data = requests.get(url, timeout=15).json()

    # Collect unique 3-letter component IDs
    comp_ids: set[str] = set()

    # Non-polymer entities (small molecules, ligands)
    for entity in data.get("rcsb_entry_container_identifiers", {}) \
                       .get("non_polymer_entity_ids", []):
        entity_url = (
            f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity"
            f"/{pdb_id}/{entity}"
        )
        try:
            entity_data = requests.get(entity_url, timeout=10).json()
            chem_comp = entity_data.get("pdbx_entity_nonpoly", {})
            comp_id = chem_comp.get("comp_id")
            if comp_id:
                comp_ids.add(comp_id)
        except Exception:
            continue

    log.info(f"Found {len(comp_ids)} unique component IDs: {comp_ids}")

    smiles_list: list[str] = []
    for comp_id in comp_ids:
        smi = _fetch_ligand_smiles_from_pdb(comp_id)
        if smi:
            smiles_list.append(smi)
            log.info(f"  {comp_id} → {smi}")
        else:
            log.warning(f"  {comp_id} → no SMILES found, skipping")

    return smiles_list


# ════════════════════════════════════════════════════════════════════════════
# 3.  Filtering
# ════════════════════════════════════════════════════════════════════════════

def filter_by_mw(smiles_list: list[str]) -> list[str]:
    """
    Keep only molecules with MW in [MW_MIN, MW_MAX].
    Also removes duplicates and invalid SMILES.
    """
    seen: set[str] = set()
    filtered: list[str] = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            log.debug(f"  Invalid SMILES, skipping: {smi}")
            continue

        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)

        mw = Descriptors.MolWt(mol)
        if MW_MIN < mw < MW_MAX:
            filtered.append(canonical)
        else:
            log.debug(f"  MW {mw:.1f} out of range, skipping: {canonical}")

    log.info(f"After MW filter: {len(filtered)} / {len(smiles_list)} molecules kept")
    return filtered


# ════════════════════════════════════════════════════════════════════════════
# 4.  Tanimoto diversity check
# ════════════════════════════════════════════════════════════════════════════

def _morgan_fps(smiles_list: list[str]):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(_morgan_gen.GetFingerprint(mol))
    return fps


def check_diversity(smiles_list: list[str]) -> tuple[bool, float]:
    """
    Returns (is_diverse, avg_tanimoto).
    is_diverse=True when avg pairwise Tanimoto < DIVERSITY_THRESHOLD.
    """
    if len(smiles_list) < 2:
        return False, 1.0

    fps = _morgan_fps(smiles_list)
    similarities: list[float] = []

    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    avg_sim = sum(similarities) / len(similarities)
    is_diverse = avg_sim < DIVERSITY_THRESHOLD

    log.info(
        f"Diversity check — avg Tanimoto: {avg_sim:.3f} "
        f"({'diverse ✓' if is_diverse else 'too similar ✗'})"
    )
    return is_diverse, avg_sim


# ════════════════════════════════════════════════════════════════════════════
# 5.  ChEMBL top-up
# ════════════════════════════════════════════════════════════════════════════

def _get_uniprot_from_pdb(pdb_id: str) -> Optional[str]:
    """
    Fetch UniProt accession for the primary polymer entity in a PDB entry.
    Uses RCSB REST API → polymer entity → rcsb_polymer_entity_container_identifiers.
    """
    try:
        # Get list of polymer entity IDs for this entry
        entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
        entry_data = requests.get(entry_url, timeout=10).json()
        entity_ids = (entry_data
                      .get("rcsb_entry_container_identifiers", {})
                      .get("polymer_entity_ids", []))

        for eid in entity_ids:
            entity_url = (f"https://data.rcsb.org/rest/v1/core"
                          f"/polymer_entity/{pdb_id}/{eid}")
            entity_data = requests.get(entity_url, timeout=10).json()
            refs = (entity_data
                    .get("rcsb_polymer_entity_container_identifiers", {})
                    .get("uniprot_ids", []))
            if refs:
                uniprot_id = refs[0]
                log.info(f"Resolved {pdb_id} → UniProt {uniprot_id}")
                return uniprot_id
    except Exception as e:
        log.debug(f"UniProt resolution failed: {e}")
    return None


def _pdb_to_chembl_target(pdb_id: str) -> Optional[str]:
    """
    Resolve PDB ID → ChEMBL target ID.
    Strategy:
      1. PDB → UniProt accession → ChEMBL target
      2. Fallback: search by PDB ID in ChEMBL xref
    """
    chembl_url = "https://www.ebi.ac.uk/chembl/api/data/target.json"

    # — Strategy 1: via UniProt accession (most reliable) ————————————
    uniprot_id = _get_uniprot_from_pdb(pdb_id)
    if uniprot_id:
        try:
            params = {"target_components__accession": uniprot_id, "limit": 5}
            data = requests.get(chembl_url, params=params, timeout=15).json()
            targets = data.get("targets", [])
            if targets:
                chembl_id = targets[0]["target_chembl_id"]
                log.info(f"UniProt {uniprot_id} → ChEMBL {chembl_id}")
                return chembl_id
        except Exception as e:
            log.debug(f"ChEMBL UniProt lookup failed: {e}")

    # — Strategy 2: ChEMBL xref by PDB ID ————————————————————————————
    try:
        params = {"target_components__target_component_xrefs__xref_id": pdb_id,
                  "limit": 5}
        data = requests.get(chembl_url, params=params, timeout=15).json()
        targets = data.get("targets", [])
        if targets:
            chembl_id = targets[0]["target_chembl_id"]
            log.info(f"PDB xref {pdb_id} → ChEMBL {chembl_id}")
            return chembl_id
    except Exception as e:
        log.debug(f"ChEMBL xref lookup failed: {e}")

    log.warning(f"Could not resolve '{pdb_id}' to a ChEMBL target")
    return None


def _diversity_select(smiles_list: list[str],
                      max_mols: int = 50,
                      tanimoto_threshold: float = 0.4) -> list[str]:
    """
    Greedy diversity selection:
      - Start with the first molecule (highest pChEMBL — list is pre-sorted)
      - Add each next molecule only if its max Tanimoto to already
        selected molecules is below tanimoto_threshold
      - Stop when max_mols is reached

    This ensures the final set is both active AND diverse.
    """
    if not smiles_list:
        return []

    selected: list[str] = []
    selected_fps = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = _morgan_gen.GetFingerprint(mol)

        if not selected_fps:
            # Always take the first (most active) molecule
            selected.append(smi)
            selected_fps.append(fp)
            continue

        # Check similarity to all already selected molecules
        max_sim = max(DataStructs.TanimotoSimilarity(fp, sel_fp)
                      for sel_fp in selected_fps)

        if max_sim < tanimoto_threshold:
            selected.append(smi)
            selected_fps.append(fp)

        if len(selected) >= max_mols:
            break

    log.info(
        f"Diversity selection: {len(selected)} / {len(smiles_list)} molecules kept "
        f"(Tanimoto threshold={tanimoto_threshold})"
    )
    return selected


def fetch_from_chembl(pdb_id: str,
                      fetch_limit: int = 100,
                      max_diverse: int = 50,
                      tanimoto_threshold: float = 0.4) -> list[str]:
    """
    Fetch active molecules from ChEMBL for the target corresponding to pdb_id.

    Steps:
      1. Fetch up to fetch_limit molecules sorted by pChEMBL descending
         (most active first)
      2. Filter by MW
      3. Greedy diversity selection → max_diverse molecules
         with pairwise Tanimoto < tanimoto_threshold
    """
    chembl_target = _pdb_to_chembl_target(pdb_id)
    if not chembl_target:
        log.warning("Could not resolve ChEMBL target — skipping ChEMBL top-up")
        return []

    url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        "target_chembl_id": chembl_target,
        "pchembl_value__gte": CHEMBL_PCHEMBL_MIN,
        "order_by": "-pchembl_value",   # most active first
        "limit": fetch_limit,
    }
    try:
        data = requests.get(url, params=params, timeout=20).json()
        activities = data.get("activities", [])

        # Collect unique SMILES preserving activity order
        seen: set[str] = set()
        smiles_ordered: list[str] = []
        for a in activities:
            smi = a.get("canonical_smiles")
            if not smi or smi in seen:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            if canonical not in seen:
                seen.add(canonical)
                smiles_ordered.append(canonical)

        log.info(f"Fetched {len(smiles_ordered)} unique molecules from ChEMBL "
                 f"(sorted by pChEMBL desc)")

        # MW filter
        mw_filtered = filter_by_mw(smiles_ordered)

        # Diversity selection
        diverse = _diversity_select(mw_filtered,
                                    max_mols=max_diverse,
                                    tanimoto_threshold=tanimoto_threshold)
        return diverse

    except Exception as e:
        log.warning(f"ChEMBL fetch failed: {e}")
        return []


# ════════════════════════════════════════════════════════════════════════════
# 6.  Save reference molecules
# ════════════════════════════════════════════════════════════════════════════

def save_references(smiles_list: list[str], pdb_id: str) -> Path:
    """
    Save reference molecules to reference_mol/<pdb_id>_references.csv.
    Columns: smiles, source, pdb_id
    Also writes a plain .smi file for REINVENT4 TL input.
    """
    records = [{"smiles": smi, "pdb_id": pdb_id} for smi in smiles_list]
    df = pd.DataFrame(records)

    csv_path = REFERENCE_DIR / f"{pdb_id}_references.csv"
    smi_path = REFERENCE_DIR / f"{pdb_id}_references.smi"

    df.to_csv(csv_path, index=False)
    # Plain SMILES file — one per line — required by REINVENT4 TL
    smi_path.write_text("\n".join(smiles_list))

    log.info(f"References saved → {csv_path}")
    log.info(f"SMILES file saved → {smi_path}  (use this for REINVENT4 TL)")
    return csv_path


# ════════════════════════════════════════════════════════════════════════════
# 7.  Main pipeline function
# ════════════════════════════════════════════════════════════════════════════

def run_data_prep(
    pdb_id: str,
    progress_callback=None,   # callable(step: str, pct: int) for Streamlit
) -> dict:
    """
    Full data preparation pipeline.

    Returns a dict with:
        protein_path   : Path   — saved .pdb file
        references_csv : Path   — saved CSV with reference SMILES
        references_smi : Path   — plain .smi for REINVENT4 TL
        smiles         : list   — final SMILES list
        n_references   : int
        avg_tanimoto   : float
        topped_up      : bool   — True if ChEMBL was used
    """

    def _progress(step: str, pct: int):
        log.info(f"[{pct:3d}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    pdb_id = pdb_id.upper().strip()

    # ── Step 1: Download protein ──────────────────────────────────────────
    _progress("Downloading protein structure from PDB...", 10)
    protein_path = fetch_protein_pdb(pdb_id)

    # ── Step 2: Fetch ligands ─────────────────────────────────────────────
    _progress("Fetching co-crystallised ligands from PDB...", 25)
    raw_smiles = fetch_ligands_from_pdb(pdb_id)

    # ── Step 3: Filter by MW ──────────────────────────────────────────────
    _progress("Filtering ligands by molecular weight (150–700 Da)...", 40)
    filtered = filter_by_mw(raw_smiles)

    # ── Step 4: Diversity check + ChEMBL top-up ──────────────────────────
    topped_up = False
    avg_tanimoto = 0.0

    if len(filtered) < MIN_REFERENCES:
        _progress(
            f"Only {len(filtered)} references found — topping up from ChEMBL...", 55
        )
        extra = fetch_from_chembl(pdb_id)
        extra_filtered = filter_by_mw(extra)
        filtered = list(dict.fromkeys(filtered + extra_filtered))
        topped_up = True
    else:
        _progress("Checking Tanimoto diversity of reference set...", 55)
        is_diverse, avg_tanimoto = check_diversity(filtered)

        if not is_diverse:
            _progress(
                f"Low diversity (avg Tanimoto={avg_tanimoto:.2f}) — topping up from ChEMBL...",
                65,
            )
            extra = fetch_from_chembl(pdb_id)
            extra_filtered = filter_by_mw(extra)
            filtered = list(dict.fromkeys(filtered + extra_filtered))
            topped_up = True

    # ── Always recompute diversity on the final set ───────────────────
    if len(filtered) >= 2:
        _progress("Computing final diversity score...", 80)
        _, avg_tanimoto = check_diversity(filtered)
    else:
        log.warning("Fewer than 2 molecules in final set — diversity score unavailable")

    # ── Step 5: Save ──────────────────────────────────────────────────────
    _progress("Saving reference molecules...", 85)
    csv_path = save_references(filtered, pdb_id)
    smi_path = REFERENCE_DIR / f"{pdb_id}_references.smi"

    _progress("Done!", 100)

    result = {
        "protein_path":   protein_path,
        "references_csv": csv_path,
        "references_smi": smi_path,
        "smiles":         filtered,
        "n_references":   len(filtered),
        "avg_tanimoto":   round(avg_tanimoto, 3),
        "topped_up":      topped_up,
    }

    log.info("=" * 55)
    log.info(f"  PDB ID          : {pdb_id}")
    log.info(f"  Protein         : {protein_path}")
    log.info(f"  References      : {len(filtered)} molecules")
    log.info(f"  Avg Tanimoto    : {avg_tanimoto:.3f}")
    log.info(f"  ChEMBL top-up   : {topped_up}")
    log.info(f"  CSV             : {csv_path}")
    log.info(f"  SMILES file     : {smi_path}")
    log.info("=" * 55)

    return result


# ════════════════════════════════════════════════════════════════════════════
# 8.  Streamlit UI (standalone — python -m streamlit run data_prep.py)
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(page_title="Data Preparation", page_icon="🧬", layout="centered")
    st.title("🧬 Molecule Generation Pipeline")
    st.subheader("Step 1 — Data Preparation")

    pdb_id = st.text_input("Enter PDB ID of the target protein", placeholder="e.g. 1ATP, 4HJO, 3ERT")

    if st.button("▶ Run Data Preparation", disabled=not pdb_id):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def ui_progress(step: str, pct: int):
            progress_bar.progress(pct)
            status_text.info(f"**{pct}%** — {step}")

        try:
            result = run_data_prep(pdb_id.strip(), progress_callback=ui_progress)

            st.success(f"✅ Done! {result['n_references']} reference molecules prepared.")

            col1, col2, col3 = st.columns(3)
            col1.metric("Reference molecules", result["n_references"])
            col2.metric("Avg Tanimoto similarity", result["avg_tanimoto"])
            col3.metric("ChEMBL top-up used", "Yes" if result["topped_up"] else "No")

            st.markdown("### 📁 Saved files")
            st.code(
                f"Protein  :  {result['protein_path']}\n"
                f"CSV      :  {result['references_csv']}\n"
                f"SMILES   :  {result['references_smi']}  ← use for REINVENT4 TL"
            )

            st.markdown("### 🔬 Reference SMILES")
            df = pd.read_csv(result["references_csv"])
            st.dataframe(df, use_container_width=True)

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            raise


# ════════════════════════════════════════════════════════════════════════════
# UniProt → ChEMBL → SMILES (универсальный режим)
# ════════════════════════════════════════════════════════════════════════════

def run_data_prep_uniprot(uniprot_id: str,
                          progress_callback=None) -> dict:
    """
    Получает референсные SMILES напрямую по UniProt ID.
    Не скачивает PDB файл — только SMILES для Tanimoto.

    Возвращает dict с теми же ключами что и run_data_prep().
    """
    uniprot_id = uniprot_id.upper().strip()

    def _progress(step: str, pct: int):
        log.info(f"[{pct:3d}%] {step}")
        if progress_callback:
            progress_callback(step, pct)

    # ── Шаг 1: UniProt → ChEMBL target ──────────────────────────────────
    _progress(f"Resolving UniProt {uniprot_id} → ChEMBL target...", 10)

    chembl_url = "https://www.ebi.ac.uk/chembl/api/data/target.json"
    params = {
        "target_components__accession": uniprot_id,
        "limit": 5
    }
    try:
        data = requests.get(chembl_url, params=params, timeout=15).json()
        targets = data.get("targets", [])
        if not targets:
            raise ValueError(
                f"Could not find ChEMBL target for UniProt {uniprot_id}"
            )
        chembl_id = targets[0]["target_chembl_id"]
        log.info(f"UniProt {uniprot_id} → ChEMBL {chembl_id}")
    except Exception as e:
        raise ValueError(f"ChEMBL resolution failed: {e}")

    # ── Шаг 2: ChEMBL → активные молекулы ───────────────────────────────
    _progress(f"Fetching active molecules from ChEMBL {chembl_id}...", 30)

    act_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params2 = {
        "target_chembl_id": chembl_id,
        "pchembl_value__gte": CHEMBL_PCHEMBL_MIN,
        "order_by": "-pchembl_value",
        "limit": CHEMBL_FETCH_LIMIT,
    }
    act_data = requests.get(act_url, params=params2, timeout=20).json()
    activities = act_data.get("activities", [])

    # Уникальные SMILES в порядке убывания активности
    seen: set = set()
    smiles_ordered = []
    for a in activities:
        smi = a.get("canonical_smiles")
        if not smi or smi in seen:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical not in seen:
            seen.add(canonical)
            smiles_ordered.append(canonical)

    log.info(f"Fetched {len(smiles_ordered)} unique molecules from ChEMBL")

    # ── Шаг 3: MW фильтр ────────────────────────────────────────────────
    _progress("Filtering by molecular weight...", 50)
    mw_filtered = filter_by_mw(smiles_ordered)

    # ── Шаг 4: Diversity selection ───────────────────────────────────────
    _progress("Selecting diverse molecules...", 70)
    diverse = _diversity_select(
        mw_filtered,
        max_mols=CHEMBL_MAX_DIVERSE,
        tanimoto_threshold=CHEMBL_DIV_THRESHOLD
    )

    # ── Шаг 5: Финальный diversity score ─────────────────────────────────
    avg_tanimoto = 0.0
    if len(diverse) >= 2:
        _progress("Computing diversity score...", 85)
        _, avg_tanimoto = check_diversity(diverse)

    # ── Шаг 6: Сохранение ───────────────────────────────────────────────
    _progress("Saving reference molecules...", 90)

    # Сохраняем с именем на основе UniProt ID
    csv_path = save_references(diverse, uniprot_id)
    smi_path = REFERENCE_DIR / f"{uniprot_id}_references.smi"

    _progress("Done!", 100)

    result = {
        "uniprot_id":     uniprot_id,
        "chembl_id":      chembl_id,
        "references_csv": csv_path,
        "references_smi": smi_path,
        "smiles":         diverse,
        "n_references":   len(diverse),
        "avg_tanimoto":   round(avg_tanimoto, 3),
        "topped_up":      True,
        "protein_path":   None,  # не скачиваем PDB в этом режиме
    }

    log.info("=" * 55)
    log.info(f"  UniProt         : {uniprot_id}")
    log.info(f"  ChEMBL target   : {chembl_id}")
    log.info(f"  References      : {len(diverse)} molecules")
    log.info(f"  Avg Tanimoto    : {avg_tanimoto:.3f}")
    log.info(f"  CSV             : {csv_path}")
    log.info(f"  SMILES file     : {smi_path}")
    log.info("=" * 55)

    return result


# ════════════════════════════════════════════════════════════════════════════
# CLI — поддержка --uniprot и --pdb
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    # ── Определяем режим запуска ──────────────────────────────────────────
    # Streamlit передаёт свои аргументы — отличаем от CLI по наличию --uniprot/--pdb
    is_cli = any(arg in sys.argv for arg in ["--uniprot", "--pdb"])

    if is_cli:
        # ── CLI режим ─────────────────────────────────────────────────────
        parser = argparse.ArgumentParser(
            description="Prepare reference SMILES for Tanimoto comparison"
        )
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument(
            "--uniprot",
            type=str,
            help="UniProt accession, e.g. P12931"
        )
        group.add_argument(
            "--pdb",
            type=str,
            help="PDB ID, e.g. 6LU7"
        )
        args = parser.parse_args()

        if args.uniprot:
            run_data_prep_uniprot(args.uniprot.upper().strip())
        else:
            run_data_prep(args.pdb.upper().strip())

    else:
        # ── Streamlit UI режим ────────────────────────────────────────────
        import streamlit as st

        st.set_page_config(
            page_title="Data Preparation",
            page_icon="🧬",
            layout="centered"
        )
        st.title("🧬 Molecule Generation Pipeline")
        st.subheader("Step 1 — Reference SMILES Preparation")

        mode = st.radio("Input mode", ["UniProt ID", "PDB ID"])

        if mode == "UniProt ID":
            input_id = st.text_input(
                "Enter UniProt ID",
                placeholder="e.g. P12931 (Src), P00533 (EGFR)"
            )
            run_fn = run_data_prep_uniprot
        else:
            input_id = st.text_input(
                "Enter PDB ID",
                placeholder="e.g. 6LU7, 1ATP"
            )
            run_fn = run_data_prep

        if st.button("▶ Run", disabled=not input_id):
            progress_bar = st.progress(0)
            status_text  = st.empty()

            def ui_progress(step, pct):
                progress_bar.progress(pct)
                status_text.info(f"**{pct}%** — {step}")

            try:
                result = run_fn(input_id.strip(),
                                progress_callback=ui_progress)

                st.success(
                    f"✅ Done! {result['n_references']} "
                    f"reference molecules prepared."
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("References", result["n_references"])
                col2.metric("Avg Tanimoto", result["avg_tanimoto"])
                if result.get("chembl_id"):
                    col3.metric("ChEMBL target", result["chembl_id"])

                st.markdown("### 📁 Saved files")
                st.code(
                    f"CSV    : {result['references_csv']}\n"
                    f"SMILES : {result['references_smi']}"
                    f"  ← use for Tanimoto comparison"
                )

                st.markdown("### 🔬 Reference SMILES")
                df = pd.read_csv(result["references_csv"])
                st.dataframe(df, use_container_width=True)

            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                raise
