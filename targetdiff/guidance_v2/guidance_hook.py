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
                          scale_min: float = 0.01, scale_max: float = 1.0) -> float:
    """Beta-annealing schedule. High sigma_t (early, noisy) -> small scale (wide Gaussian,
    long-range guidance, reaches ~6 A). Low sigma_t (late) -> scale_max=1.0 (sharp beta,
    precise placement, force peak at 0.5-1.5 A). scale_min=0.01 reaches 6 A early;
    full sweep 0.01 -> 1.0 covers both ends of the radius problem."""
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

        # beta-annealing: wide Gaussian early (long-range ~6 A), sharp late (precise placement)
        field.set_beta_scale(beta_scale_from_sigma(sigma_t, sigma_max))

        # real typing (not a proxy) — differentiable in pos
        c = soft_atom_types(pos, elements, tau=tau)            # (N,T)

        g_act = activity_gradient(field, pos, c)               # (N,3) attraction
        g_ster = steric_gradient(pos)                          # (N,3) repulsion (Buckingham)

    # w(t) rising toward the end (Variant B): SAFETY VALVE, not cosmetics.
    # Early (sigma_t high): small w -> suppress the wide-radius force where typing is
    #   unreliable (CN on noisy coords lies). Late (sigma_t low): large w -> full force
    #   where types are trustworthy and the well is sharp. frac = sigma_t/sigma_max in [0,1].
    frac = max(0.0, min(1.0, float(sigma_t) / float(sigma_max)))
    w_t = 1.0 - frac                                       # sigma high (early) -> w~0; sigma low (late) -> w~1
    force = w_t * (lambda_act * g_act - lambda_ster * g_ster)

    return force.detach()
