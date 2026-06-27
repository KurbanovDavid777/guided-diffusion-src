# targetdiff/guidance_v2/steric.py
"""
Differentiable steric repulsion for ligand atoms.
"""

from __future__ import annotations

import torch

__all__ = ["steric_penalty", "steric_gradient"]


def steric_penalty(
    x: torch.Tensor, sigma: float = 0.9
) -> torch.Tensor:
    """
    Compute the steric penalty S_ster(x) = Σ_{i<j} exp(-||x_i - x_j||^2 / sigma^2).

    Parameters
    ----------
    x : torch.Tensor
        (N, 3) coordinates of ligand atoms. Requires grad.
    sigma : float, default 2.7
        Repulsion range in Å.

    Returns
    -------
    torch.Tensor
        Scalar steric penalty.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Pairwise squared distances (N, N)
    diff = x[:, None, :] - x[None, :, :]          # (N, N, 3)
    d2 = (diff ** 2).sum(-1)

    # Upper triangular indices (i < j)
    N = x.size(0)
    iu = torch.triu_indices(N, N, offset=1, device=x.device)   # device-safe
    penalty = torch.exp(-d2[iu[0], iu[1]] / (sigma ** 2)).sum()

    # Compute penalty
    penalty = torch.exp(-d2[iu[0], iu[1]] / (sigma**2)).sum()
    return penalty


def steric_gradient(
    x: torch.Tensor, sigma: float = 0.9
) -> torch.Tensor:
    """
    Compute the gradient ∇_x S_ster via autograd.

    Parameters
    ----------
    x : torch.Tensor
        (N, 3) coordinates of ligand atoms. Requires grad.
    sigma : float, default 2.7
        Repulsion range in Å.

    Returns
    -------
    torch.Tensor
        (N, 3) gradient tensor.
    """
    penalty = steric_penalty(x, sigma)
    grad = torch.autograd.grad(penalty, x, create_graph=False)[0]
    return grad


if __name__ == "__main__":
    # Self-test
    torch.manual_seed(0)
    # Define 6 atoms: 3 pairs with distances 1.0, 3.0, 6.0 Å
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # atom 0
            [0.0, 0.0, 1.0],  # atom 1 (close to 0)
            [5.0, 0.0, 0.0],  # atom 2
            [5.0, 0.0, 3.0],  # atom 3 (normal to 2)
            [10.0, 0.0, 0.0],  # atom 4
            [10.0, 0.0, 6.0],  # atom 5 (far from 4)
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    sigma = 0.9
    penalty = steric_penalty(coords, sigma)
    grad = steric_gradient(coords, sigma)

    print("Penalty shape:", penalty.shape, "value:", penalty.item())
    print("Gradient shape:", grad.shape)
    print("Gradient norm:", grad.norm().item())

    # Check that penalty is positive and finite
    assert penalty.item() > 0.0
    assert torch.isfinite(penalty).all()

    # Check gradient shape and finiteness
    assert grad.shape == coords.shape
    assert torch.isfinite(grad).all()
    assert not torch.allclose(grad, torch.zeros_like(grad))

    # Repulsion test: move atoms opposite to gradient
    lr = 0.01
    with torch.no_grad():
        coords_new = coords - lr * grad

    # Distances before and after for the close pair (0,1)
    dist_before = torch.norm(coords[0] - coords[1]).item()
    dist_after = torch.norm(coords_new[0] - coords_new[1]).item()
    print(f"Distance before: {dist_before:.4f} Å")
    print(f"Distance after  : {dist_after:.4f} Å")
    assert dist_after > dist_before, "Repulsion did not increase distance"


    # Finding 4: real-overlap regime — clash must dominate bond (sigma=0.9 separates them)
    clash = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    gc = steric_gradient(clash)        # default sigma=0.9
    print(f"  clash d=0.5: |grad|={gc.norm().item():.3f}")   # should be large (~1.28)

    bond = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
    gb = steric_gradient(bond)         # default sigma=0.9
    print(f"  bond d=1.5:  |grad|={gb.norm().item():.3f}")    # should be small (~0.32)

    # both must exist BEFORE the comparison
    assert gc.norm().item() > 2.0 * gb.norm().item(), "clash must dominate bond (sigma separates them)"

    print("All tests passed.")

