"""Slack-redistribution refiner (Phase 1: translate-only, dims/bbox frozen).

Gated by FLOORSET_SLACK_REFINE (default off). See
docs/design/slack_refiner_spec.md for the full algorithm.
"""

from .api import refine_layout

__all__ = ["refine_layout"]
