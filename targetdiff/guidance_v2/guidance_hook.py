# targetdiff/guidance_v2/guidance_hook.py
"""
Guidance hook for TargetDiff sampler: assembles the geometry-native force
  dE = w(t) * [ lambda_act * grad_S_act  -  lambda_ster * grad_S_ster ]
with beta-annealing over diffusion time (fixes activity dead-zone, Finding 2).
Novelty (C3) is NOT included yet — activity + steric only.
"""

from __future__ import annotations
from typing import List, Optional

import torch

from targetdiff.guidance_v2.activity import PharmacophoreField, activity_gradient, TYPE_ORDER
from targetdiff.guidance_v2.atom_typing import soft_atom_types
from targetdiff.guidance_v2.steric import steric_gradient

_Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}


def atomic_numbers_to_symbols(atomic_nums: List[int]) -> List[str]:
    """Map atomic numbers to element symbols; unknown -> 'C' (safe hydrophobic default)."""
    return [_Z_TO_SYMBOL.get(int(z), "C") for z in atomic_nums]


def beta_scale_from_sigma(sigma_t: float, sigma_max: float,
                          scale_min: float = 0.04, scale_max: float = 1.0) -> float:
    """Beta-annealing schedule. High sigma_t (early, noisy) -> small scale (wide Gaussian,
    long-range guidance). Low sigma_t (late) -> scale_max (sharp, precise placement).
    scale_min=0.04 makes beta ~25x wider early (capture radius ~2.5 A instead of 0.45 A)."""
    frac = max(0.0, min(1.0, sigma_t / sigma_max))          # in [0,1]
    return scale_max - (scale_max - scale_min) * frac        # sigma big -> scale small


def compute_guidance_force(
    pos: torch.Tensor,                 # (N,3) x0_pred, requires_grad set inside
    elements: List[str],               # length N element symbols (fixed over steps)
    field: PharmacophoreField,
    sigma_t: float,                    # noise level of current step
    sigma_max: float,                  # max sigma (for normalization)
    lambda_act: float = 1.0,
    lambda_ster: float = 1.0,
    tau: float = 0.5,
) -> torch.Tensor:                     # (N,3) force to ADD to ligand_pos
    """Assemble dE = w(t) [ lambda_act * grad_S_act - lambda_ster * grad_S_ster ].
    Sign: +act (attraction), -ster (repulsion). w(t)=sigma_t^2."""
    with torch.enable_grad():
        pos = pos.detach().requires_grad_(True)

        # beta-annealing: widen the activity Gaussian on noisy early steps
        field.set_beta_scale(beta_scale_from_sigma(sigma_t, sigma_max))

        # real typing (not a proxy) — differentiable in pos
        c = soft_atom_types(pos, elements, tau=tau)            # (N,T)

        g_act = activity_gradient(field, pos, c)               # (N,3) attraction
        g_ster = steric_gradient(pos)                          # (N,3) repulsion (Buckingham)

    w_t = float(sigma_t) ** 2                              # w(t) ~ sigma_t^2
    force = w_t * (lambda_act * g_act - lambda_ster * g_ster)
    return force.detach()
