"""Public orchestration helpers for the possession-analysis pipeline.

Notebooks and scripts should call this module rather than rebuilding feature,
scenario, anchor, or decoder composition themselves. The lower-level modules
remain independently testable and contain the domain calculations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from possession.baseline_pipeline import ClosestTeamRuleModel
from possession.calculations.features import build_engineered_features
from possession.models.pipelines import PipelineSpec, apply_pipeline_spec
from possession.temporal_rules import load_rules


def build_evidence_table(
    model: ClosestTeamRuleModel | None = None,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the canonical supplied-plus-engineered frame evidence table."""
    return build_engineered_features(model=model, rules=rules)


def detect_scenarios(
    features: pd.DataFrame,
    rules: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a copy with scenario confidence and episode evidence attached."""
    from possession.decoder_experiments import add_scenario_episode_features, scenario_confidence_scores

    scored = scenario_confidence_scores(features.copy())
    return add_scenario_episode_features(scored)


def run_decision_pipeline(
    features: pd.DataFrame | None = None,
    *,
    model: ClosestTeamRuleModel | None = None,
    rules: dict[str, Any] | None = None,
    spec: PipelineSpec | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build evidence and apply one declared anchor/transition/decoder spec.

    Returns the complete feature table and one frame-aligned H/A/D/X label array.
    Ground truth is never read by this helper.
    """
    resolved_rules = load_rules() if rules is None else rules
    evidence = build_evidence_table(model=model, rules=resolved_rules) if features is None else features.copy()
    evidence = detect_scenarios(evidence, resolved_rules)
    resolved_spec = PipelineSpec(name="protected_aligned_reception") if spec is None else spec
    labels = apply_pipeline_spec(evidence, resolved_spec, resolved_rules)
    return evidence, labels


__all__ = ["PipelineSpec", "build_evidence_table", "detect_scenarios", "run_decision_pipeline"]
