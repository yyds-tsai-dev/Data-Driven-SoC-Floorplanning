#!/usr/bin/env python3
"""Submission optimizer wrapper.

The best local validation path is the frame-first post-repair optimizer:
it keeps the stable relative-order legalizer, repairs MIB dimensions from
fixed/preplaced references, then snaps boundary and cluster components with
guarded acceptance. Keeping this wrapper thin avoids diverging from the
tested helper implementation.
"""

from my_optimizer_frame_first_best import MyOptimizer as _FrameFirstOptimizer


class MyOptimizer(_FrameFirstOptimizer):
    """Official entry point loaded by iccad2026_evaluate.py."""

    pass
