#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: targetdiff.guidance_v2.plip_reweight
Description:
    Level‑1 re‑weighting of Level‑0 pharmacophore hotspots based on real
    interaction evidence from PLIP analysis of co‑crystal structures.
    Matching is now performed using apex coordinates instead of residue
    numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import numpy as np

# PLIP imports – may raise ImportError if PLIP is not installed
try:
    from plip.structure.preparation import PDBComplex
except Exception as exc:  # pragma: no cover
    raise ImportError("PLIP library is required for this module.") from exc

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

_SOURCE_RE = re.compile(
    r"(?P<chain>[A-Za-z]):(?P<resname>[A-Z]+)(?P<resnum>\d+):(?P<atom>[A-Za-z0-9]+)"
)

_PTYPE_MAP = {
    "acceptor": "hbond",
    "donor": "hbond",
    "hydrophobic": "hydrophobic",
}

# --------------------------------------------------------------------------- #
# PLIP interaction extraction
# --------------------------------------------------------------------------- #
def plip_interactions(pdb_path: str) -> List[Dict[str, Any]]:
    """
    Run PLIP analysis on a PDB file and return a list of interaction dicts.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.

    Returns
    -------
    List[Dict]
        Each dict contains:
            itype   : "hbond" | "hydrophobic"
            resnr   : int
            restype : str
            coords  : Tuple[float, float, float]
    """
    interactions: List[Dict[str, Any]] = []

    try:
        mol = PDBComplex()
        mol.load_pdb(pdb_path)
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"PLIP failed to load {pdb_path}: {exc}")
        return interactions

    try:
        mol.analyze()
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"PLIP analysis failed for {pdb_path}: {exc}")
        return interactions

    ligands = getattr(mol, "ligands", None)
    if not ligands and not mol.interaction_sets:
        warnings.warn(f"No ligand found in {pdb_path}")
        return interactions

    for key, site in mol.interaction_sets.items():
        if not site.all_itypes:
            continue

        for hb in site.hbonds_pdon:
            try:
                coords = tuple(hb.d.coords)
            except Exception:
                warnings.warn(
                    f"Missing coordinates for H‑bond in {pdb_path} (site {key})"
                )
                continue
            interactions.append(
                {
                    "itype": "hbond",
                    "resnr": hb.resnr,
                    "restype": hb.restype,
                    "coords": coords,
                }
            )

        for hb in site.hbonds_ldon:
            try:
                coords = tuple(hb.a.coords)
            except Exception:
                warnings.warn(
                    f"Missing coordinates for H‑bond in {pdb_path} (site {key})"
                )
                continue
            interactions.append(
                {
                    "itype": "hbond",
                    "resnr": hb.resnr,
                    "restype": hb.restype,
                    "coords": coords,
                }
            )

        for hc in site.hydrophobic_contacts:
            try:
                coords = tuple(hc.bsatom.coords)
            except Exception:
                try:
                    coords = tuple(hc.coords)
                except Exception:
                    warnings.warn(
                        f"Missing coordinates for hydrophobic contact in {pdb_path} (site {key})"
                    )
                    continue
            interactions.append(
                {
                    "itype": "hydrophobic",
                    "resnr": hc.resnr,
                    "restype": hc.restype,
                    "coords": coords,
                }
            )

    if not interactions:
        warnings.warn(f"No interactions found in {pdb_path}")

    return interactions


# --------------------------------------------------------------------------- #
# Hotspot re‑weighting
# --------------------------------------------------------------------------- #
def reweight_hotspots(
    hotspots_json: str,
    pdb_paths: List[str],
    out_json: str,
    base_low: float = 0.2,
    base_high: float = 2.0,
    apex_tol: float = 2.5,
) -> None:
    """
    Re‑weight Level‑0 hotspots based on PLIP evidence using apex coordinates.

    Parameters
    ----------
    hotspots_json : str
        Path to the input JSON file produced by hotspots.py.
    pdb_paths : List[str]
        List of paths to co‑crystal PDB files.
    out_json : str
        Path to write the updated hotspots JSON.
    base_low : float, default 0.2
        Weight for hotspots never confirmed.
    base_high : float, default 2.0
        Weight for hotspots confirmed in all structures.
    apex_tol : float, default 2.5
        Distance tolerance (Å) for matching apex to PLIP interaction.
    """
    # Load hotspots
    with open(hotspots_json, "r", encoding="utf-8") as fh:
        hotspots: List[Dict[str, Any]] = json.load(fh)

    n_structures = len(pdb_paths)
    if n_structures == 0:
        raise ValueError("No PDB paths provided.")

    # Pre‑load all interactions for debug and later use
    interactions_per_pdb: Dict[str, List[Dict[str, Any]]] = {}
    for pdb_path in pdb_paths:
        interactions_per_pdb[pdb_path] = plip_interactions(pdb_path)

    # Debug: min apex distance for first 10 hotspots
    for hs in hotspots[:10]:
        source = hs.get("source", "")
        ptype = hs.get("ptype", "")
        apex_raw = hs.get("apex")
        if apex_raw is None:
            min_dist = float("inf")
        else:
            apex = np.array(apex_raw, dtype=float)
            itype_needed = _PTYPE_MAP.get(ptype.lower())
            if itype_needed is None:
                min_dist = float("inf")
            else:
                min_dist = float("inf")
                for intrs in interactions_per_pdb.values():
                    for intr in intrs:
                        if intr["itype"] != itype_needed:
                            continue
                        dist = np.linalg.norm(apex - np.array(intr["coords"], dtype=float))
                        if dist < min_dist:
                            min_dist = dist
        dist_str = f"{min_dist:.2f}" if min_dist != float("inf") else "inf"
        print(
            f"[debug] {source[:25]} ({ptype}): min apex-dist = {dist_str} A",
            file=sys.stderr,
        )

    # Count support per hotspot
    support_counts: List[int] = [0] * len(hotspots)

    # Main loop over structures
    for pdb_path in pdb_paths:
        interactions = interactions_per_pdb[pdb_path]
        if not interactions:
            continue

        for idx, hs in enumerate(hotspots):
            apex_raw = hs.get("apex")
            if apex_raw is None:
                continue
            apex = np.array(apex_raw, dtype=float)

            ptype = hs.get("ptype", "").lower()
            itype_needed = _PTYPE_MAP.get(ptype)
            if itype_needed is None:
                continue

            matched = False
            for intr in interactions:
                if intr["itype"] != itype_needed:
                    continue
                dist = np.linalg.norm(apex - np.array(intr["coords"], dtype=float))
                if dist < apex_tol:
                    matched = True
                    break
            if matched:
                support_counts[idx] += 1

    # Update hotspots with new weights and metadata
    for hs, count in zip(hotspots, support_counts):
        freq = count / n_structures
        new_weight = base_low + (base_high - base_low) * freq
        hs["weight_level0"] = hs.get("weight", None)
        hs["support_count"] = count
        hs["frequency"] = freq
        hs["weight"] = new_weight

    # Write output
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(hotspots, fh, indent=2)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_summary(hotspots: List[Dict[str, Any]]) -> None:
    support_counts = [hs["support_count"] for hs in hotspots]
    freq_counts = [hs["frequency"] for hs in hotspots]

    confirmed = sum(1 for c in support_counts if c > 0)
    total = len(hotspots)

    print(f"\nSummary:")
    print(f"  Total hotspots: {total}")
    print(f"  Confirmed in ≥1 structure: {confirmed} ({confirmed / total * 100:.1f}%)")

    # Frequency distribution histogram
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(freq_counts, bins=bins)
    print("\nFrequency distribution (bins 0.0–1.0):")
    for i in range(len(hist)):
        print(f"  {bins[i]:.1f}–{bins[i+1]:.1f}: {hist[i]}")

    # Top‑5 by support_count
    top5 = sorted(
        [(hs["support_count"], hs["position"], hs["ptype"]) for hs in hotspots],
        reverse=True,
    )[:5]
    print("\nTop‑5 hotspots by support_count:")
    for count, pos, ptype in top5:
        print(f"  Count={count:2d}  Position={pos}  Ptype={ptype}")

    # Count of hotspots with apex=None
    apex_none = sum(1 for hs in hotspots if hs.get("apex") is None)
    print(f"\nHotspots with apex=None: {apex_none}")


def main() -> None:
    warnings.simplefilter("always")
    parser = argparse.ArgumentParser(
        description="Re‑weight Level‑0 hotspots using PLIP analysis."
    )
    parser.add_argument(
        "--hotspots",
        required=True,
        help="Path to input hotspots JSON (from hotspots.py).",
    )
    parser.add_argument(
        "--pdbs",
        nargs="+",
        required=True,
        help="List of co‑crystal PDB files.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to output JSON with updated weights.",
    )
    parser.add_argument(
        "--base-low",
        type=float,
        default=0.2,
        help="Base low weight (default: 0.2).",
    )
    parser.add_argument(
        "--base-high",
        type=float,
        default=2.0,
        help="Base high weight (default: 2.0).",
    )
    parser.add_argument(
        "--apex-tol",
        type=float,
        default=2.5,
        help="Distance tolerance (Å) for matching apex to PLIP interaction.",
    )
    args = parser.parse_args()

    # Validate files
    if not Path(args.hotspots).is_file():
        sys.exit(f"Hotspots file not found: {args.hotspots}")
    for pdb in args.pdbs:
        if not Path(pdb).is_file():
            sys.exit(f"PDB file not found: {pdb}")

    reweight_hotspots(
        hotspots_json=args.hotspots,
        pdb_paths=args.pdbs,
        out_json=args.out,
        base_low=args.base_low,
        base_high=args.base_high,
        apex_tol=args.apex_tol,
    )

    # Load updated hotspots for summary
    with open(args.out, "r", encoding="utf-8") as fh:
        updated_hotspots = json.load(fh)

    _print_summary(updated_hotspots)


if __name__ == "__main__":  # pragma: no cover
    main()

