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
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)

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

        # Convert to tensors
        self.P = torch.tensor(positions, dtype=torch.float32, device=self.device)  # (M,3)
        self.W = torch.tensor(weights, dtype=torch.float32, device=self.device)    # (M,)

        M = len(positions)
        T = len(TYPE_ORDER)
        type_onehot = torch.zeros((M, T), dtype=torch.float32, device=self.device)
        type_onehot[torch.arange(M), torch.tensor(type_indices, device=self.device)] = 1.0
        self.TYPE = type_onehot  # (M,T)

        self.beta = torch.tensor(betas, dtype=torch.float32, device=self.device)  # (M,)

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
        d2 = torch.cdist(x, self.P, p=2) ** 2

        # Match ligand atom types to hotspot types (N, M)
        c_match = c @ self.TYPE.T

        # Raw Gaussian overlap (N, M)
        raw = c_match * torch.exp(-self.beta * d2)

        # Softmax normalization over atoms for each hotspot (N, M)
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

    print("All tests passed.")

    # Clean up synthetic file
    if not args.hotspots:
        os.unlink(hotspots_path)

