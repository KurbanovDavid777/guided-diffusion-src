# targetdiff/guidance_v2/activity.py
import json
import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

# Order of pharmacophore types (fixed)
TYPE_ORDER = ["acceptor", "donor", "hydrophobic", "pos", "neg"]

# Default beta values per type (Å⁻²)
DEFAULT_BETA = {
    "acceptor": 2.5,
    "donor": 2.5,
    "hydrophobic": 0.6,
    "pos": 1.5,
    "neg": 1.5,
}


class PharmacophoreField(nn.Module):
    """
    Differentiable Gaussian pharmacophore overlap.

    Attributes:
        P (torch.Tensor): (M, 3) hotspot positions
        W (torch.Tensor): (M,) hotspot weights
        TYPE (torch.Tensor): (M, T) one‑hot hotspot type matrix
        beta (torch.Tensor): (M,) per‑hotspot beta values
    """

    def __init__(
        self,
        hotspots_json: str,
        beta_per_type: Optional[Dict[str, float]] = None,
        kappa: float = 1.0,          # angular sharpness; 0 = sphere (no angle), higher = tighter cone
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.kappa = float(kappa)


        # Load hotspot definitions
        try:
            with open(hotspots_json, "r") as f:
                hotspots = json.load(f)
        except Exception as exc:
            raise RuntimeError(f"Failed to load hotspots JSON: {exc}") from exc

        positions: list[list[float]] = []
        weights: list[float] = []
        type_indices: list[int] = []
        betas: list[float] = []
        directions: list = [] 

        for hotspot in hotspots:
            pos = hotspot.get("position")
            if pos is None or len(pos) != 3:
                raise ValueError(f"Hotspot missing or invalid position: {hotspot}")
            positions.append(pos)

            weights.append(float(hotspot.get("weight", 1.0)))

            ptype = hotspot.get("ptype")
            if ptype not in TYPE_ORDER:
                raise ValueError(f"Unknown ptype {ptype} in hotspot")
            type_indices.append(TYPE_ORDER.index(ptype))

            # Determine beta
            if beta_per_type and ptype in beta_per_type:
                betas.append(float(beta_per_type[ptype]))
            else:
                betas.append(float(DEFAULT_BETA[ptype]))

            d = hotspot.get("direction")
            if d is not None and len(d) == 3:
                directions.append(d)
            else:
                directions.append([0.0, 0.0, 0.0])   # hydrophobic/charged: no axis -> sphere


        # Convert to tensors
        self.P = torch.tensor(positions, dtype=torch.float32, device=self.device)  # (M,3)
        self.W = torch.tensor(weights, dtype=torch.float32, device=self.device)    # (M,)
        self.D = torch.tensor(directions, dtype=torch.float32, device=self.device)  # (M, 3) cone axes; zero = sphere

        M = len(positions)
        T = len(TYPE_ORDER)
        type_onehot = torch.zeros((M, T), dtype=torch.float32, device=self.device)
        type_onehot[torch.arange(M), torch.tensor(type_indices, device=self.device)] = 1.0
        self.TYPE = type_onehot  # (M,T)

        self.beta = torch.tensor(betas, dtype=torch.float32, device=self.device)  # (M,)

    def set_beta_scale(self, scale: float) -> None:
        """Scale all per-hotspot beta by `scale` for beta-annealing over diffusion time.
        scale<1 -> wider Gaussian (early/noisy); scale=1 -> original sharp beta (late)."""
        if not hasattr(self, "_beta_base"):
            self._beta_base = self.beta.clone()   # store original once
        self.beta = self._beta_base * float(scale)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute scalar activity S_act.

        Args:
            x (torch.Tensor): (N, 3) ligand atom coordinates
            c (torch.Tensor): (N, T) soft membership of ligand atom types

        Returns:
            torch.Tensor: scalar activity
        """
        x = x.to(self.device)
        c = c.to(self.device)

        # Pairwise squared distances (N, M)
        diff = x[:, None, :] - self.P[None, :, :]   # (N, M, 3)
        d2 = (diff ** 2).sum(-1)                      # (N, M), gradient-safe at d=0


        # Match ligand atom types to hotspot types (N, M)
        c_match = c.detach() @ self.TYPE.T   # typing defines TYPE, not a force; gradient flows via distance only

        # angular term: atom should approach the hotspot ALONG its H-bond axis.
        # axis self.D[m] points from pocket heavy atom outward to the target point;
        # the ligand atom should sit on that axis -> vector (x_i - p_m) aligned with D_m.
        diff_hp = x[:, None, :] - self.P[None, :, :]          # (N, M, 3): from hotspot to atom
        diff_norm = torch.sqrt((diff_hp ** 2).sum(-1, keepdim=True) + 1e-6)  # softened norm, grad-safe at d=0
        u = diff_hp / diff_norm                                # (N, M, 3) unit
        D_norm = self.D.norm(dim=-1, keepdim=True)             # (M, 1)
        has_axis = (D_norm.squeeze(-1) > 1e-6).float()         # (M,) 1 if directional, 0 if sphere
        D_unit = self.D / D_norm.clamp_min(1e-6)                # (M, 3) axis points +D into pocket (ligand approach side) 
        cos_theta = (u * D_unit[None, :, :]).sum(-1)           # (N, M), angle atom-axis
        # angular(θ) = exp(κ(cosθ − 1)); =1 when aligned, decays off-axis.
        # for sphere hotspots (no axis) angular=1 (disabled) via has_axis mask.
        angular = torch.exp(self.kappa * (cos_theta - 1.0))    # (N, M)
        angular = angular * has_axis[None, :] + (1.0 - has_axis[None, :])  # sphere -> 1
        # gate angular -> 1 at small d (angle is meaningless when atom sits on the point)
        d_gate = torch.sigmoid((d2 - 0.25) / 0.1)              # ~0 for d<0.5A, ~1 for d>0.5A
        angular = angular * d_gate + (1.0 - d_gate)           # near d=0 -> angular=1
        raw = c_match * torch.exp(-self.beta * d2) * angular   # (N, M), now direction-aware

        # Softmax normalization over atoms for each hotspot (N, M)
        # Anti-collapse over atoms (dim=0): one hotspot can't be farmed by many atoms.
        # NOTE: one-sided — does NOT penalize one atom satisfying many hotspots.
        # That mirror case (per-atom cap = "satisfy some, not all") is TODO.
        norm = torch.softmax(-d2.detach(), dim=0)

        # Weighted sum over hotspots and atoms
        weighted = norm * raw * self.W  # broadcast W over N

        S_act = weighted.sum()  # scalar
        return S_act


def activity_gradient(
    field: PharmacophoreField, x: torch.Tensor, c: torch.Tensor
) -> torch.Tensor:
    """
    Compute ∇_x S_act via autograd.

    Args:
        field (PharmacophoreField): field instance
        x (torch.Tensor): (N, 3) ligand atom coordinates (requires_grad=True)
        c (torch.Tensor): (N, T) ligand atom type memberships

    Returns:
        torch.Tensor: gradient shape (N, 3)
    """
    S_act = field(x, c)
    grad = torch.autograd.grad(S_act, x, retain_graph=False)[0]
    return grad


# --------------------------------------------------------------------------- #
# Self‑test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Test PharmacophoreField activity.")
    parser.add_argument(
        "--hotspots",
        type=str,
        default=None,
        help="Path to hotspots JSON file. If omitted, synthetic hotspots are used.",
    )
    args = parser.parse_args()

    # Prepare hotspots
    if args.hotspots and os.path.isfile(args.hotspots):
        hotspots_path = args.hotspots
    else:
        synthetic_hotspots = [
            {
                "position": [0.0, 0.0, 0.0],
                "ptype": "acceptor",
                "weight": 1.0,
                "source": "synthetic",
                "direction": None,
            },
            {
                "position": [5.0, 0.0, 0.0],
                "ptype": "donor",
                "weight": 1.0,
                "source": "synthetic",
                "direction": None,
            },
            {
                "position": [0.0, 5.0, 0.0],
                "ptype": "hydrophobic",
                "weight": 1.0,
                "source": "synthetic",
                "direction": None,
            },
        ]
        tmp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(synthetic_hotspots, tmp)
        tmp.close()
        hotspots_path = tmp.name
        print(f"Using synthetic hotspots at {hotspots_path}")

    device = "cpu"
    field = PharmacophoreField(hotspots_path, device=device)

    # Ligand atoms
    N = 10
    hotspot_centroid = field.P.mean(dim=0).clone().detach()
    x = hotspot_centroid + torch.randn(N, 3, dtype=torch.float32)
    # put atom 0 deliberately ~1 Å from the first acceptor hotspot (in its force range)
    acc_idx = (field.TYPE[:, TYPE_ORDER.index("acceptor")] > 0.5).nonzero()[0, 0]
    x[0] = field.P[acc_idx] + 0.3 * torch.randn(3)   # ~0.3-0.5 Å от hotspot
    x = x.clone().detach().requires_grad_(True)

    # c: one‑hot acceptor for atom 0, others uniform
    T = len(TYPE_ORDER)
    c = torch.full((N, T), 1.0 / T, dtype=torch.float32)
    acceptor_idx = TYPE_ORDER.index("acceptor")
    c[0, acceptor_idx] = 1.0

    # Compute activity
    S_act = field(x, c)
    print("S_act:", S_act.item())

    # Gradient
    grad = activity_gradient(field, x, c)
    print("grad shape:", grad.shape)
    print("grad norm:", grad.norm().item())

    # Assertions
    assert S_act.item() > 0, "S_act should be positive"
    assert not torch.isnan(grad).any(), "Gradient contains NaN"
    assert grad.norm().item() > 0, "Gradient norm should be non-zero"

    # Check direction for atom 0
    acceptor_mask = field.TYPE[:, acceptor_idx] > 0.5
    if acceptor_mask.sum() == 0:
        print("No acceptor hotspots defined.")
        sys.exit(0)
    acceptor_positions = field.P[acceptor_mask]
    d2_before = torch.cdist(x[0:1], acceptor_positions, p=2).min()
    lr = 0.1
    x_new = x.clone()
    x_new[0] = x_new[0] + lr * grad[0]
    d2_after = torch.cdist(x_new[0:1], acceptor_positions, p=2).min()
    print("Distance before:", d2_before.item(), "after:", d2_after.item())
    assert d2_after < d2_before, "Atom 0 did not move closer to acceptor hotspot"

    # --- Finding 2: capture-radius probe (NOT an assert — documents the dead zone) ---
    # Sharp Gaussian (beta=2.5 -> sigma~0.45 A) means activity is a finishing force:
    # an atom far from any hotspot feels almost no pull until beta-annealing (TODO in 2.6).
    acc_col = TYPE_ORDER.index("acceptor")
    acc_hot = field.P[(field.TYPE[:, acc_col] > 0.5)][0]
    for dist in [0.5, 1.0, 1.5, 2.0, 3.0]:
        probe = (acc_hot + torch.tensor([dist, 0.0, 0.0])).unsqueeze(0).clone().requires_grad_(True)
        c_probe = torch.zeros(1, len(TYPE_ORDER)); c_probe[0, acc_col] = 1.0
        gp = activity_gradient(field, probe, c_probe)
        print(f"  capture radius: d={dist:.1f} A  ->  |grad|={gp.norm().item():.2e}")



    # angular term: at SAME distance, on-axis must beat off-axis
    m0 = 0
    p0 = field.P[m0]
    axis0 = field.D[m0]
    if axis0.norm() > 1e-6:
        axis0 = axis0 / axis0.norm()
        # build a vector strictly perpendicular to the axis (Gram-Schmidt)
        ref = torch.tensor([1.0, 0.0, 0.0]) if abs(axis0[0]) < 0.9 else torch.tensor([0.0, 1.0, 0.0])
        perp = ref - (ref @ axis0) * axis0
        perp = perp / perp.norm()
        r = 1.0
        on_plus  = (p0 + r * axis0).unsqueeze(0).clone().requires_grad_(True)   # +axis side
        on_minus = (p0 - r * axis0).unsqueeze(0).clone().requires_grad_(True)   # -axis side
        off_axis = (p0 + r * perp).unsqueeze(0).clone().requires_grad_(True)    # perpendicular
        c1 = torch.zeros(1, len(TYPE_ORDER)); c1[0, int(field.TYPE[m0].argmax())] = 1.0
        s_plus  = field(on_plus,  c1).item()
        s_minus = field(on_minus, c1).item()
        s_off   = field(off_axis, c1).item()
        print(f"  angular: +axis S={s_plus:.4f}  -axis S={s_minus:.4f}  off S={s_off:.4f}")

        # STRICT: must distinguish the TWO sides of the axis, not just axis-vs-perpendicular.
        # Geometry: position = apex + 2.9*D, so +D points INTO the pocket (away from protein),
        # -D points back toward the protein atom. Ligand approaches FROM the pocket side (+D).
        # Correct cone must reward +axis (pocket side) and penalize -axis (through-protein).
        print(f"  sides: +axis={s_plus:.4f}  perp={s_off:.4f}  -axis={s_minus:.4f}")
        assert s_plus > s_off, "+axis (pocket side) must beat perpendicular"
        assert s_off > s_minus, "perpendicular must beat -axis (through-protein side)"
        # i.e. strict ordering  +axis > perp > -axis  confirms axis points into pocket


    print("All tests passed.")

    # Clean up synthetic file
    if not args.hotspots:
        os.unlink(hotspots_path)

