# targetdiff/guidance_v2/atom_typing.py
import torch
from typing import List, Dict

from targetdiff.guidance_v2.activity import TYPE_ORDER

# --------------------------------------------------------------------------------------------------- #
#  Base soft‑type vectors (element → 5‑vector in the order of TYPE_ORDER)
# --------------------------------------------------------------------------------------------------- #
BASE_TYPES: Dict[str, List[float]] = {
    "N": [0.5, 0.6, 0.0, 0.0, 0.0],   # acceptor, donor, hydrophobic, pos, neg
    "O": [0.8, 0.4, 0.0, 0.0, 0.0],
    "C": [0.0, 0.0, 0.8, 0.0, 0.0],
    "S": [0.3, 0.0, 0.5, 0.0, 0.0],
    "F": [0.2, 0.0, 0.6, 0.0, 0.0],
    "Cl": [0.2, 0.0, 0.6, 0.0, 0.0],
    "Br": [0.2, 0.0, 0.6, 0.0, 0.0],
    "I": [0.2, 0.0, 0.6, 0.0, 0.0],
    # default for unknown elements
    "default": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# --------------------------------------------------------------------------------------------------- #
#  Soft atom‑type assignment
# --------------------------------------------------------------------------------------------------- #
def soft_atom_types(
    coords: torch.Tensor,          # (N, 3) – ligand atom coordinates
    elements: List[str],           # length N – atomic symbols
    tau: float = 1.0,              # softness temperature for geometry gates
) -> torch.Tensor:                 # (N, T) – soft membership, differentiable in coords
    """
    Returns a soft membership matrix of shape (N, T) where T = len(TYPE_ORDER).
    The matrix is differentiable w.r.t. `coords`.
    """
    device = coords.device
    N = coords.shape[0]
    T = len(TYPE_ORDER)

    # 1. Base vectors for each atom
    base_list = []
    for el in elements:
        base_list.append(BASE_TYPES.get(el, BASE_TYPES["default"]))
    base = torch.tensor(base_list, dtype=torch.float32, device=device)  # (N, T)

    # 2. Soft coordination number (coord_num)
    dists = torch.cdist(coords, coords, p=2)          # (N, N)
    mask = torch.ones_like(dists, dtype=torch.bool, device=device)
    mask.fill_diagonal_(False)
    R_cut = 1.8  # Å
    gate = torch.sigmoid((R_cut - dists) / tau)      # (N, N)
    coord_num = (gate * mask.float()).sum(dim=1)     # (N,)

    # 3. Geometry‑based modulation
    idx_acceptor = TYPE_ORDER.index("acceptor")
    idx_donor = TYPE_ORDER.index("donor")
    idx_hydrophobic = TYPE_ORDER.index("hydrophobic")
    idx_pos = TYPE_ORDER.index("pos")
    idx_neg = TYPE_ORDER.index("neg")

    for el in set(elements):
        mask_el = torch.tensor([e == el for e in elements], dtype=torch.bool, device=device)
        if not mask_el.any():
            continue
        cn = coord_num[mask_el]          # (n_el,)

        if el == "N":
            thr = 2.0
            thr_pos = 3.0
            donor_factor = torch.sigmoid((thr - cn) / tau)
            acceptor_factor = torch.sigmoid((cn - thr) / tau)
            pos_factor = torch.sigmoid((cn - thr_pos) / tau)
            base[mask_el, idx_donor] *= donor_factor
            base[mask_el, idx_acceptor] *= acceptor_factor
            base[mask_el, idx_pos] += pos_factor * 0.3

        elif el == "O":
            # O is dominantly an acceptor; donor only mildly raised for hydroxyl-like (low CN)
            thr = 1.5
            donor_factor = torch.sigmoid((thr - cn) / tau)        # mild donor boost when terminal-ish
            base[mask_el, idx_donor] *= (0.3 + 0.4 * donor_factor)  # donor stays modest
            # acceptor stays strong; do NOT suppress it by CN
            # (optional tiny CN penalty only when heavily coordinated)
            acceptor_keep = torch.sigmoid((3.0 - cn) / tau)        # only drop if CN very high (>3)
            base[mask_el, idx_acceptor] *= (0.6 + 0.4 * acceptor_keep)

        elif el == "S":
            thr = 2.0
            acceptor_factor = torch.sigmoid((cn - thr) / tau)
            base[mask_el, idx_acceptor] *= acceptor_factor

        # halogens and other elements: no modulation

    # 4. Clamp to [0, 1]
    c = torch.clamp(base, 0.0, 1.0)   # (N, T)
    return c


# --------------------------------------------------------------------------------------------------- #
#  Self‑test
# --------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    elements = ["N", "C", "C", "O", "C"]
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],          # N
            [1.4, 0.0, 0.0],          # C
            [2.8, 0.0, 0.0],          # C
            [4.2, 0.0, 0.0],          # O
            [5.6, 0.0, 0.0],          # C
        ],
        dtype=torch.float32,
    )
    coords.requires_grad_(True)

    c = soft_atom_types(coords, elements, tau=0.5)
    print("Soft types (N, T):")
    print(c)

    # Assertions
    assert c.shape == (5, 5), "Shape mismatch"
    assert torch.all(c >= 0) and torch.all(c <= 1), "Values out of [0,1]"
    assert not torch.isnan(c).any(), "NaNs present"

    idx_acceptor = TYPE_ORDER.index("acceptor")
    idx_hydrophobic = TYPE_ORDER.index("hydrophobic")
    idx_donor = TYPE_ORDER.index("donor")

    assert c[3, idx_acceptor] > c[3, idx_donor], "O should be more acceptor than donor"
    assert c[1, idx_hydrophobic] > c[1, idx_donor], "C not more hydrophobic"
    assert c[0, idx_donor] > 0.3, "N not enough donor"

    c.sum().backward()
    assert coords.grad is not None, "No gradient"
    assert not torch.all(coords.grad == 0), "Zero gradient"
    print("All tests passed.")

