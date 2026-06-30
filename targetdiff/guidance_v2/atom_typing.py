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
    "F": [0.1, 0.0, 0.7, 0.0, 0.0],    # F: very weak acceptor, mostly lipophilic
    "Cl": [0.0, 0.0, 0.8, 0.0, 0.0],   # halogens: hydrophobic (X-bonds via σ-hole not modeled)
    "Br": [0.0, 0.0, 0.8, 0.0, 0.0],
    "I": [0.0, 0.0, 0.8, 0.0, 0.0],
    # default for unknown elements
    "default": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# --------------------------------------------------------------------------------------------------- #
#  Soft atom‑type assignment
# --------------------------------------------------------------------------------------------------- #
def soft_atom_types(
    coords: torch.Tensor,          # (N, 3) ligand atom coordinates
    elements: List[str],           # length N, atomic symbols
    tau: float = 0.5,              # softness temperature for geometry gates
) -> torch.Tensor:                 # (N, T) soft membership, differentiable in coords
    """
    Soft, differentiable pharmacophore typing from element + local geometry.

    Functional (no in-place index assignment) so autograd is robust across
    torch versions. CN (soft coordination number) is a surrogate for valence
    saturation in the absence of bonds/hydrogens; known edge cases: pyridine,
    aniline. Halogen bonds (σ-hole) are not modeled — heavy halogens are treated
    as hydrophobic. Ether vs carbonyl/hydroxyl O is partially resolved via CN
    (2 heavy neighbors → ether-like, donor suppressed)

    Known limit: heavy-atom CN cannot distinguish hydroxyl (-OH, donor) from
    carbonyl (=O, non-donor) — both have CN=1. Terminal O donor is a deliberate
    low average (biased toward the more common carbonyl/ether case). Only ether
    (CN=2) is resolved vs terminal O.    

    """
    device = coords.device
    N = coords.shape[0]
    T = len(TYPE_ORDER)
    idx_acc = TYPE_ORDER.index("acceptor")
    idx_don = TYPE_ORDER.index("donor")
    idx_pos = TYPE_ORDER.index("pos")

    # 1. Base vectors per atom (element lookup; non-differentiable, as intended)
    base = torch.tensor(
        [BASE_TYPES.get(el, BASE_TYPES["default"]) for el in elements],
        dtype=torch.float32, device=device,
    )  # (N, T)

    # 2. Soft coordination number (differentiable in coords)
    dists = torch.cdist(coords, coords, p=2)          # (N, N)
    R_cut = 1.8                                        # Å, heavy-atom neighbor cutoff
    gate = torch.sigmoid((R_cut - dists) / tau)       # (N, N)
    eye = torch.eye(N, device=device)                 # exclude self
    coord_num = (gate * (1.0 - eye)).sum(dim=1)       # (N,)

    # 3. Per-element soft multipliers, built FUNCTIONALLY (no in-place on `base`)
    #    We assemble a multiplier matrix `mult` (N, T) then base * mult.
    is_N = torch.tensor([e == "N" for e in elements], dtype=torch.float32, device=device)  # (N,)
    is_O = torch.tensor([e == "O" for e in elements], dtype=torch.float32, device=device)

    # gates as smooth functions of coord_num (N,)
    g_low_N  = torch.sigmoid((1.5 - coord_num) / tau)   # CN<1.5 (terminal amine) -> donor
    g_high_N = torch.sigmoid((coord_num - 1.5) / tau)   # CN>1.5 (ring/imine) -> acceptor
    g_pos_N  = torch.sigmoid((coord_num - 3.0) / tau)   # very high CN -> protonated/quaternary -> pos

    g_ether  = torch.sigmoid((coord_num - 1.5) / tau)   # O with ~2 neighbors -> ether-like
    g_term_O = torch.sigmoid((1.5 - coord_num) / tau)   # O with ~1 neighbor  -> carbonyl/hydroxyl
    g_keepA  = torch.sigmoid((3.0 - coord_num) / tau)   # drop acceptor only if very high CN

    # Start from all-ones multiplier; fill per type/element via masks (functional).
    mult = torch.ones((N, T), dtype=torch.float32, device=device)

    # --- Nitrogen: donor scaled by low-CN, acceptor by high-CN ---
    don_mult_N = g_low_N * g_low_N        # squared: ring N donor suppressed harder
    acc_mult_N = 0.3 + 1.2 * g_high_N     # ring N acceptor boosted above base
    mult = mult + is_N.unsqueeze(1) * (
        torch.nn.functional.one_hot(torch.tensor(idx_don, device=device), T).float() * (don_mult_N - 1.0).unsqueeze(1)
        + torch.nn.functional.one_hot(torch.tensor(idx_acc, device=device), T).float() * (acc_mult_N - 1.0).unsqueeze(1)
    )

    # --- Oxygen: acceptor stays strong; donor suppressed for ether-like (high CN) ---
    # Terminal O (CN=1) is ambiguous: hydroxyl (donor) vs carbonyl (NOT donor).
    # Heavy-atom CN cannot separate them (both CN=1) without H/bonds.
    # Most terminal O in drug-like ligands are carbonyl/ether, so keep donor LOW.
    don_mult_O = (0.15 + 0.25 * g_term_O) * (1.0 - 0.8 * g_ether)
    acc_mult_O = (0.6 + 0.4 * g_keepA)                            # acceptor stays 0.6..1.0
    mult = mult + is_O.unsqueeze(1) * (
        torch.nn.functional.one_hot(torch.tensor(idx_don, device=device), T).float() * (don_mult_O - 1.0).unsqueeze(1)
        + torch.nn.functional.one_hot(torch.tensor(idx_acc, device=device), T).float() * (acc_mult_O - 1.0).unsqueeze(1)
    )

    c = base * mult  # (N, T), functional — no in-place index writes

    # --- pos channel for N at very high CN (additive, e.g. quaternary/protonated amine) ---
    pos_add = is_N * g_pos_N * 0.3                                # (N,)
    c = c + torch.nn.functional.one_hot(torch.tensor(idx_pos, device=device), T).float() * pos_add.unsqueeze(1)

    # 4. Clamp to [0, 1]
    c = torch.clamp(c, 0.0, 1.0)   # (N, T)
    return c


# --------------------------------------------------------------------------------------------------- #
#  Self‑test
# --------------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    elements = ["N", "C", "C", "O", "C"]
    coords = torch.tensor(
        [[0.0,0.0,0.0],[1.4,0.0,0.0],[2.8,0.0,0.0],[4.2,0.0,0.0],[5.6,0.0,0.0]],
        dtype=torch.float32,
    )
    coords.requires_grad_(True)
    c = soft_atom_types(coords, elements)
    print("Soft types (N, T):\n", c)

    idx = {t: TYPE_ORDER.index(t) for t in TYPE_ORDER}

    # shape / range / nan
    assert c.shape == (5, 5)
    assert torch.all(c >= 0) and torch.all(c <= 1)
    assert not torch.isnan(c).any()

    # chemistry sanity
    assert c[3, idx["acceptor"]] > c[3, idx["donor"]], "O should be more acceptor than donor"
    assert c[1, idx["hydrophobic"]] > c[1, idx["donor"]], "C should be hydrophobic"
    assert c[0, idx["donor"]] > 0.3, "N should donate"

    # gradient flows into coords (geometry is differentiable)
    # differentiability AND locality (meaningful, not just non-zero):
    g = torch.autograd.grad(c.sum(), coords, retain_graph=True)[0]
    assert g is not None and not torch.all(g == 0), "typing must be differentiable in coords"

    # locality: an isolated atom (no heavy neighbors) has CN~0 -> typing must NOT
    # depend on its coords (no phantom long-range gradient)
    iso = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    c_iso = soft_atom_types(iso, ["O", "O"])
    g_iso = torch.autograd.grad(c_iso.sum(), iso, retain_graph=True)[0]
    assert g_iso.norm() < 1e-3, f"isolated atoms should have ~0 typing gradient, got {g_iso.norm():.4f}"
    print("locality grad norm (should be ~0):", round(g_iso.norm().item(), 6))

    # ether O: 2 heavy neighbors -> donor suppressed vs terminal O
    eth = torch.tensor([[0.0,0.0,0.0],[1.4,0.0,0.0],[2.8,0.0,0.0]], dtype=torch.float32)  # C-O-C
    c_eth = soft_atom_types(eth, ["C","O","C"])
    term = torch.tensor([[0.0,0.0,0.0],[1.4,0.0,0.0]], dtype=torch.float32)               # C=O terminal
    c_term = soft_atom_types(term, ["C","O"])
    assert c_eth[1, idx["donor"]] < c_term[1, idx["donor"]], "ether O should have less donor than terminal O"
    print("ether O donor:", round(c_eth[1, idx['donor']].item(),3),
          "| terminal O donor:", round(c_term[1, idx['donor']].item(),3))

    # halogens: hydrophobic, not acceptor
    c_hal = soft_atom_types(torch.randn(3,3), ["Cl","Br","I"])
    assert torch.all(c_hal[:, idx["acceptor"]] < 0.05), "heavy halogens should not be acceptors"
    assert torch.all(c_hal[:, idx["hydrophobic"]] > 0.5), "halogens should be hydrophobic"

    print("All tests passed.")
