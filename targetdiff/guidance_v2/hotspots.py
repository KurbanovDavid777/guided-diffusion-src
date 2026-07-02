# targetdiff/guidance_v2/hotspots.py
"""
Extract Level‑0 pharmacophoric hotspots from a protein pocket PDB.

The module parses the PDB directly (no RDKit bond perception) and classifies
heavy atoms by residue/atom name.  For each atom a complementary ligand
interaction point is produced:

* backbone donors   → “acceptor” hotspot
* backbone acceptors→ “donor” hotspot
* side‑chain donors → “acceptor” hotspot
* side‑chain acceptors→ “donor” hotspot
* hydrophobic atoms → “hydrophobic” hotspot
* negatively charged atoms → “pos” hotspot
* positively charged atoms → “neg” hotspot

The hotspot position is projected along the interaction vector
(e.g. donor→hydrogen direction).  When the required hydrogen or heavy
neighbor is missing a fallback to the pocket centroid is used with a
reduced weight.

Duplicate hotspots of the same type within ``merge_radius`` are merged
by averaging their positions and summing their weights (capped at 2.0).

The function ``extract_hotspots`` returns a list of :class:`Hotspot`
instances.  A small CLI is provided for quick testing.

Author: 2026‑06‑26
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants – atom name sets
# --------------------------------------------------------------------------- #

# Backbone donors (all except PRO)
BACKBONE_DONOR = {"N"}

# Backbone acceptors
BACKBONE_ACCEPTOR = {"O", "OXT"}

# Side‑chain donors
SIDECHAIN_DONORS: Dict[str, set] = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    #"HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "TRP": {"NE1"},
    "CYS": {"SG"},
}

# Side‑chain acceptors
SIDECHAIN_ACCEPTORS: Dict[str, set] = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "HIS": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "MET": {"SD"},
}

# Hydrophobic atoms
HYDROPHOBIC: Dict[str, set] = {
    "ALA": {"CB"},
    "VAL": {"CB", "CG1", "CG2"},
    "LEU": {"CB", "CG", "CD1", "CD2"},
    "ILE": {"CB", "CG1", "CG2", "CD1"},
    "MET": {"CB", "CG", "CE"},
    "PRO": {"CB", "CG", "CD"},
    "PHE": {"CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CB", "CG", "CD2", "CE3", "CZ2", "CZ3", "CH2"},
}

# Negatively charged atoms → ligand “pos”
NEG_CHARGED: Dict[str, set] = {
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
}

# Positively charged atoms → ligand “neg”
POS_CHARGED: Dict[str, set] = {
    "ARG": {"NH1", "NH2", "NE"},
    "LYS": {"NZ"},
    "HIS": {"ND1", "NE2"},
}

# Interaction distances (Å)
D_HB = 2.9
D_HYDRO = 4.0
D_SALT = 3.5

# Weights
WEAK_WEIGHT = 0.5
FALLBACK_WEIGHT = 0.5

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Hotspot:
    position: np.ndarray  # (3,) target point for ligand atom, Å
    ptype: str            # "acceptor"|"donor"|"hydrophobic"|"pos"|"neg"
    weight: float = 1.0   # confidence
    source: str = ""      # e.g. "A:SER195:OG"
    direction: Optional[np.ndarray] = None  # (3,) unit vector along interaction
    apex: Optional[np.ndarray] = None   # coord of the source protein atom (for PLIP matching)


# --------------------------------------------------------------------------- #
# PDB parsing utilities
# --------------------------------------------------------------------------- #

# --- module-level constant (вынеси наверх, к другим константам) ---
STANDARD_AA = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    "HID","HIE","HIP","CYX","CYM","ASH","GLH","LYN",  # protonation variants
}


def _parse_pdb_line(line: str) -> Optional[Tuple[str, str, int, str, str, np.ndarray, str]]:
    """
    Parse an ATOM line of a *protein* pocket.

    HETATM records (ligand UNL, waters, ions, cofactors) and any non-standard
    residue are skipped: Level-0 hotspots come from protein atoms only.

    Returns:
        chain, resname, resseq, atomname, element, coords, source_id
    """
    if not line.startswith("ATOM"):          # (1) only ATOM, drop HETATM
        return None
    try:
        atomname = line[12:16].strip()
        resname = line[17:20].strip()
        NORMALIZE = {"HID":"HIS","HIE":"HIS","HIP":"HIS","ASH":"ASP","GLH":"GLU","CYX":"CYS","CYM":"CYS","LYN":"LYS"}
        resname = NORMALIZE.get(resname, resname)
        if resname not in STANDARD_AA:       # (2) safety: skip non-AA residues
            return None
        chain = line[21].strip()
        resseq = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        element = line[76:78].strip()
        if not element:
            element = atomname[0]
        source_id = f"{chain}:{resname}{resseq}:{atomname}"
        return chain, resname, resseq, atomname, element, np.array([x, y, z], dtype=float), source_id
    except (ValueError, IndexError) as exc:   # (3) was `return Non` (typo) + bare except
        warnings.warn(f"Failed to parse line: {line.strip()!r} – {exc}")
        return None


def _read_pdb(path: Path) -> List[Tuple[str, str, int, str, str, np.ndarray, str]]:
    """
    Read all ATOM/HETATM records from a PDB file.
    """
    atoms = []
    with path.open() as fh:
        for line in fh:
            parsed = _parse_pdb_line(line)
            if parsed:
                atoms.append(parsed)
    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return atoms


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _is_heavy(element: str) -> bool:
    return element.upper() != "H"


def _find_hydrogen_for_atom(
    heavy_atom: Tuple[str, str, int, str, str, np.ndarray, str],
    residue_atoms: List[Tuple[str, str, int, str, str, np.ndarray, str]],
    max_xh: float = 1.4,  # Å, covalent X–H upper bound (N–H/O–H ~1.0, S–H ~1.35)
) -> Optional[np.ndarray]:
    """
    Return coords of the hydrogen covalently bound to `heavy_atom`,
    i.e. the nearest H within `max_xh` Å. Matching is geometric, not by
    name, because protonation tools (e.g. OpenBabel) label every H as "H".
    If the heavy atom carries several H (e.g. Arg NH1), the closest is
    returned — sufficient for a donor direction vector.
    """
    _, _, _, _, _, heavy_coord, _ = heavy_atom
    best_coord: Optional[np.ndarray] = None
    best_d2 = max_xh * max_xh
    for _, _, _, _, element, coord, _ in residue_atoms:
        if element.upper() != "H":
            continue
        d2 = float(np.sum((coord - heavy_coord) ** 2))  # squared dist, Å²
        if d2 < best_d2:
            best_d2 = d2
            best_coord = coord
    return best_coord


def _nearest_heavy_neighbor(
    heavy_atom: Tuple[str, str, int, str, str, np.ndarray, str],
    residue_atoms: List[Tuple[str, str, int, str, str, np.ndarray, str]],
    max_dist: float = 1.8,
) -> Optional[np.ndarray]:
    """
    Find the nearest heavy atom in the same residue (excluding the atom itself).
    Return its coordinates or None if none within ``max_dist``.
    """
    _, _, _, _, _, heavy_coord, _ = heavy_atom
    min_dist = float("inf")
    nearest = None
    for _, _, _, atom_name, element, coord, _ in residue_atoms:
        if element.upper() == "H" or atom_name == heavy_atom[3]:
            continue
        dist = np.linalg.norm(coord - heavy_coord)
        if dist < min_dist and dist <= max_dist:
            min_dist = dist
            nearest = coord
    return nearest


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.zeros_like(vec)
    return vec / norm


# --------------------------------------------------------------------------- #
# Main extraction routine
# --------------------------------------------------------------------------- #

def extract_hotspots(
    pocket_pdb: str,
    ph: float = 7.0,
    merge_radius: float = 1.5,
) -> List[Hotspot]:
    """
    Extract Level‑0 pharmacophoric hotspots from a protein pocket.

    Parameters
    ----------
    pocket_pdb : str
        Path to the PDB file containing the pocket (already protonated at the
        desired pH).
    ph : float, optional
        pH value (unused – kept for API compatibility).
    merge_radius : float, optional
        Radius (Å) within which hotspots of the same type are merged.

    Returns
    -------
    List[Hotspot]
        List of extracted hotspots.
    """
    path = Path(pocket_pdb)
    atoms = _read_pdb(path)

    # Group atoms by residue for quick lookup
    residues: Dict[Tuple[str, int], List[Tuple[str, str, int, str, str, np.ndarray, str]]] = {}
    for rec in atoms:
        chain, resname, resseq, _, _, _, _ = rec
        residues.setdefault((chain, resseq), []).append(rec)

    # Compute pocket centroid (heavy atoms only)
    heavy_coords = np.array([c for _, _, _, _, e, c, _ in atoms if _is_heavy(e)])
    if heavy_coords.size == 0:
        raise ValueError("No heavy atoms found in pocket.")
    centroid = heavy_coords.mean(axis=0)

    hotspots: List[Hotspot] = []

    # Helper to add hotspot
    def _add(ptype: str, pos: np.ndarray, weight: float, source: str,
             direction: Optional[np.ndarray], apex: Optional[np.ndarray]):
        hotspots.append(Hotspot(position=pos, ptype=ptype, weight=weight, source=source,
                                direction=direction, apex=apex))

    # Iterate over all heavy atoms
    for rec in atoms:
        chain, resname, resseq, atom_name, element, coord, source_id = rec
        if not _is_heavy(element):
            continue

        # 1. Backbone donors
        if atom_name in BACKBONE_DONOR and resname != "PRO":
            h_coord = _find_hydrogen_for_atom(rec, residues[(chain, resseq)])
            if h_coord is not None:
                vec = h_coord - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("acceptor", pos, 1.0, source_id, direction, coord)
            else:
                vec = centroid - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("acceptor", pos, FALLBACK_WEIGHT, source_id, direction, coord)
                warnings.warn(f"Donor {source_id} missing hydrogen – fallback used.", RuntimeWarning)

        # 2. Backbone acceptors
        if atom_name in BACKBONE_ACCEPTOR:
            neighbor = _nearest_heavy_neighbor(rec, residues[(chain, resseq)])
            if neighbor is not None:
                vec = coord - neighbor
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("donor", pos, 1.0, source_id, direction, coord)
            else:
                vec = centroid - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("donor", pos, FALLBACK_WEIGHT, source_id, direction, coord)
                warnings.warn(f"Acceptor {source_id} missing heavy neighbor – fallback used.", RuntimeWarning)

        # 3. Side‑chain donors
        if resname in SIDECHAIN_DONORS and atom_name in SIDECHAIN_DONORS[resname]:
            h_coord = _find_hydrogen_for_atom(rec, residues[(chain, resseq)])
            if h_coord is not None:
                vec = h_coord - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("acceptor", pos, 1.0, source_id, direction, coord)
            else:
                vec = centroid - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("acceptor", pos, FALLBACK_WEIGHT, source_id, direction, coord)
                warnings.warn(f"Side‑chain donor {source_id} missing hydrogen – fallback used.", RuntimeWarning)

        # 4. Side‑chain acceptors
        if resname in SIDECHAIN_ACCEPTORS and atom_name in SIDECHAIN_ACCEPTORS[resname]:
            neighbor = _nearest_heavy_neighbor(rec, residues[(chain, resseq)])
            if neighbor is not None:
                vec = coord - neighbor
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("donor", pos, 1.0, source_id, direction, coord)
            else:
                vec = centroid - coord
                direction = _unit_vector(vec)
                pos = coord + D_HB * direction
                _add("donor", pos, FALLBACK_WEIGHT, source_id, direction, coord)
                warnings.warn(f"Side‑chain acceptor {source_id} missing heavy neighbor – fallback used.", RuntimeWarning)

        # 5. Hydrophobic atoms
        if resname in HYDROPHOBIC and atom_name in HYDROPHOBIC[resname]:
            vec = centroid - coord
            direction = _unit_vector(vec)
            pos = coord + D_HYDRO * direction
            _add("hydrophobic", pos, 1.0, source_id, None, coord)

        # # 6. Negatively charged atoms → ligand “pos”
        # if resname in NEG_CHARGED and atom_name in NEG_CHARGED[resname]:
        #     vec = centroid - coord
        #     direction = _unit_vector(vec)
        #     pos = coord + D_SALT * direction
        #     _add("pos", pos, 1.0, source_id, None, coord)
        #
        # # 7. Positively charged atoms → ligand “neg”
        # if resname in POS_CHARGED and atom_name in POS_CHARGED[resname]:
        #     vec = centroid - coord
        #     direction = _unit_vector(vec)
        #     pos = coord + D_SALT * direction
        #     _add("neg", pos, 1.0, source_id, None ,coord)

    # ----------------------------------------------------------------------- #
    # Merge duplicates
    # ----------------------------------------------------------------------- #
    merged: List[Hotspot] = []
    used = [False] * len(hotspots)

    for i, h in enumerate(hotspots):
        if used[i]:
            continue
        group = [h]
        used[i] = True
        for j in range(i + 1, len(hotspots)):
            if used[j]:
                continue
            h2 = hotspots[j]
            if h.ptype != h2.ptype:
                continue
            if np.linalg.norm(h.position - h2.position) <= merge_radius:
                group.append(h2)
                used[j] = True
        # Merge group
        positions = np.stack([g.position for g in group])
        avg_pos = positions.mean(axis=0)
        total_weight = min(sum(g.weight for g in group), 2.0)
        sources = ", ".join(g.source for g in group)
        dirs = [g.direction for g in group if g.direction is not None]
        if dirs:
            avg_dir = np.mean(np.stack(dirs), axis=0)
            n = np.linalg.norm(avg_dir)
            merged_dir = avg_dir / n if n > 0 else None
        else:
            merged_dir = None
        apexes = [g.apex for g in group if g.apex is not None]
        merged_apex = apexes[0] if apexes else None   # keep first source atom as apex
        merged.append(
            Hotspot(
                position=avg_pos,
                ptype=h.ptype,
                weight=total_weight,
                source=sources,
                direction=merged_dir,
                apex=merged_apex,
            )
        )
    return merged


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _to_json(hotspots: List[Hotspot]) -> List[dict]:
    out = []
    for h in hotspots:
        out.append(
            {
                "position": h.position.tolist(),
                "ptype": h.ptype,
                "weight": h.weight,
                "source": h.source,
                "direction": h.direction.tolist() if h.direction is not None else None,
                "apex": h.apex.tolist() if h.apex is not None else None,
            }
        )
    return out


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract Level‑0 hotspots from a protein pocket.")
    parser.add_argument("--pocket", required=True, help="Path to pocket PDB file")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--ph", type=float, default=7.0, help="pH (unused, for API compatibility)")
    parser.add_argument("--merge", type=float, default=1.5, help="Merge radius (Å)")
    args = parser.parse_args(argv)

    hotspots = extract_hotspots(args.pocket, ph=args.ph, merge_radius=args.merge)
    json.dump(_to_json(hotspots), Path(args.out).open("w"), indent=2)


if __name__ == "__main__":
    main()

