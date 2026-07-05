"""Stub for the partner's missing diffusion_data module.

The real module was not shared; these stubs let my_opt_claude_v2.py import
cleanly. Without a loadable checkpoint the optimizer falls back to its
heuristic seed, which per partner/README_claude.md is not the quality
bottleneck (a ground-truth seed scores the same through the legalizer).
"""


def build_condition(*args, **kwargs):
    raise RuntimeError("diffusion_data stub: real module not available")


def fp_sol_to_z0(*args, **kwargs):
    raise RuntimeError("diffusion_data stub: real module not available")
