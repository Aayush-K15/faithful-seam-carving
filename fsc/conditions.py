"""
fsc.conditions - every method in the ablation study, behind one signature:

    run(name, image, labels, target_width, target_height=None, params=None) -> CarveResult

The engine and its settings are identical across seam-carving conditions;
only the named components change.

Group                 Name            What changes
--------------------  --------------  ---------------------------------------------------
Baselines             scale           uniform scaling
                      crop            centre crop
                      sc_backward     vanilla seam carving (Avidan & Shamir 2007 energy)
                      sc_forward      forward energy (Rubinstein et al. 2008) = BASE
Original three fixes  lines           BASE + Canny/Hough line penalty
                      saliency        BASE + spectral-residual saliency
                      knee            BASE + cost-knee adaptive stop (finish: scale)
                      fixes_all       BASE + lines + saliency + knee
Integrity (ours)      static_mask     BASE + constant cost on object pixels (the usual fix)
                      budget          BASE + per-object budget (no gap term, no guard)
                      budget_gap      BASE + budget + gap protection (no guard)
                      faithful        BASE + budget + gaps + integrity guard (finish: scale)
                      faithful_crop   same, finish: crop keeping the most object pixels
                      faithful_all    faithful + lines + saliency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import baselines
from .engine import CarveResult, resize
from .fixes import CostKneeStop, line_mask, spectral_residual_saliency
from .integrity import InstanceIntegrity, object_mask


@dataclass
class Params:
    line_weight: float = 2.0      # cost added on protected line pixels
    saliency_weight: float = 1.0  # cost added per unit of saliency
    static_weight: float = 1.0    # static object-mask cost (same as the budget's base cost)
    lam: float = 1.0              # budget: base cost of an object pixel
    gamma: float = 8.0            # budget: growth with fraction deleted
    gap_px: int = 3
    gap_cost: float = 50.0
    max_loss: float = 0.5         # guard threshold
    knee: dict = field(default_factory=dict)   # CostKneeStop kwargs


GROUPS = {
    "Baselines": ["scale", "crop", "sc_backward", "sc_forward"],
    "Original fixes": ["lines", "saliency", "knee", "fixes_all"],
    "Integrity (ours)": ["static_mask", "budget", "budget_gap", "faithful", "faithful_crop", "faithful_all"],
}
ALL = [c for g in GROUPS.values() for c in g]

LABELS = {
    "scale": "Scaling", "crop": "Centre crop", "sc_backward": "Seam carving (backward)",
    "sc_forward": "Seam carving (forward)", "lines": "+ lines", "saliency": "+ saliency",
    "knee": "+ cost-knee stop", "fixes_all": "+ all three fixes", "static_mask": "+ static masks",
    "budget": "+ budget", "budget_gap": "+ budget + gaps", "faithful": "Faithful (ours)",
    "faithful_crop": "Faithful, crop finish", "faithful_all": "Faithful + lines + saliency",
}


def run(name: str, image: np.ndarray, labels: np.ndarray, target_width: int,
        target_height: Optional[int] = None, params: Optional[Params] = None) -> CarveResult:
    p = params or Params()
    th = target_height

    if name == "scale":
        return baselines.scale(image, target_width, th)
    if name == "crop":
        return baselines.crop(image, target_width, th)
    if name == "sc_backward":
        return resize(image, target_width, th, energy="backward")
    if name == "sc_forward":
        return resize(image, target_width, th)

    static = np.zeros(image.shape[:2])
    if name in ("lines", "fixes_all", "faithful_all"):
        static = static + p.line_weight * line_mask(image)
    if name in ("saliency", "fixes_all", "faithful_all"):
        static = static + p.saliency_weight * spectral_residual_saliency(image)
    if name == "static_mask":
        static = static + p.static_weight * object_mask(labels)
    pix = static if static.any() else None

    if name in ("lines", "saliency", "static_mask"):
        return resize(image, target_width, th, pixel_energy=pix)
    if name in ("knee", "fixes_all"):
        return resize(image, target_width, th, pixel_energy=pix, stop_fn=CostKneeStop(**p.knee))

    if name in ("budget", "budget_gap", "faithful", "faithful_crop", "faithful_all"):
        ctrl = InstanceIntegrity(labels, lam=p.lam, gamma=p.gamma, gap_px=p.gap_px,
                                 gap_cost=p.gap_cost if name != "budget" else 0.0,
                                 max_loss=p.max_loss)
        guard: Optional[Callable] = ctrl.guard if name.startswith("faithful") else None
        finish = "crop" if name == "faithful_crop" else "scale"
        importance = object_mask(labels) if finish == "crop" else None
        res = resize(image, target_width, th, pixel_energy=pix, energy_fn=ctrl.energy,
                     guard_fn=guard, finish=finish, crop_importance=importance)
        res.refusals = dict(ctrl.refusals)  # type: ignore[attr-defined]
        return res

    raise ValueError(f"unknown condition {name!r}; choose from {ALL}")
