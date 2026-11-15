"""Evaluate the short-occlusion ball-flight feature in isolation."""

from pathlib import Path

import pandas as pd

from possession.baseline_pipeline import ClosestTeamRuleModel, graded_mask
from possession.decoder_experiments import (
    _remove_short_dead_runs,
    classify_scenarios,
    dead_ball_boundary_evidence,
    restart_provisional_labels,
    scenario_gated_labels,
    bridge_same_team_missing_ball_trajectory,
    scenario_confidence_scores,
    duration_aware_scenario_decode,
    add_scenario_episode_features,
    bounded_bidirectional_scenario_resolver,
    apply_switch_confirmation,
    apply_switch_confirmation_hysteresis,
    resolve_cross_team_bidirectional_switch,
    resolve_reception_switch_inside_interval,
    reception_transition_alignment_audit,
    resolve_aligned_reception_switch,
    apply_stable_reception_episode_switch,
    restart_control_boundary_labels,
    hierarchical_restart_fill_labels,
)
from possession.scenario_decoder import decode_scenario_labels
from possession.temporal_rules import (
    add_anchor_features,
    add_reachability_features,
    add_special_anchor_features,
    add_boundary_anchor_features,
    add_temporal_features,
    add_transition_forecast_features,
    add_switch_confirmation_features,
    add_reception_outcome_features,
    add_reception_geometry_features,
    add_reception_signal_features,
    add_first_time_pass_features,
    add_reception_episode_features,
    load_rules,
)
from possession.baseline_pipeline import slot_team_map
from games import asset_paths
from starter.load_smart_tagging import load as load_smart_tagging
from possession.pipeline import build_evidence_table as build_engineered_features


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rules = load_rules()
    model = ClosestTeamRuleModel()
    temporal = build_engineered_features(model=model, rules=rules)

    interpolated = temporal.copy()
    interpolated["recent_flight_evidence"] = temporal["recent_flight_evidence_interpolated"]
    interpolated["ball_flight_candidate"] = temporal["ball_flight_candidate_interpolated"]
    labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
    )
    scenario_labels, _ = decode_scenario_labels(temporal, rules)
    dead_support = (
        temporal["graded"].to_numpy(bool)
        & temporal["smart_set_piece"].to_numpy(bool)
        & (scenario_labels == "D")
        & temporal["scenario_context"].isin(["DEAD", "UNKNOWN", "OCCLUDED"]).to_numpy(bool)
    )
    labels[dead_support] = "D"
    labels = _remove_short_dead_runs(labels, minimum_run=25)
    output = ROOT / "evaluation" / "results" / "baseline_scenario_bidirectional_interpolated_flight_ensemble.csv"
    model.write_submission(output, labels)
    print(output)

    # Candidate: use only agreeing same-team anchors to fill a physically
    # coherent missing-ball bridge. Keep separate until public evaluation shows
    # that it improves both frame and spell quality.
    trajectory_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
    )
    trajectory_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_team_missing_ball_trajectory.csv"
    model.write_submission(trajectory_bridge_output, trajectory_bridge_labels)
    print(trajectory_bridge_output)

    coordinate_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
        require_player_path=True,
    )
    coordinate_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_team_coordinate_path_bridge.csv"
    model.write_submission(coordinate_bridge_output, coordinate_bridge_labels)
    print(coordinate_bridge_output)

    # Diagnostic candidate: only bridge a missing-ball gap when the same
    # temporary player path remains speed-compatible with the ball. A sharp
    # change is deliberately excluded from this candidate because it is more
    # consistent with a pass, deflection, or tackle than dribble continuity.
    dribble_bridge_labels = bridge_same_team_missing_ball_trajectory(
        temporal,
        labels,
        maximum_gap_seconds=float(rules["timeline"]["long_ball_max_seconds"]),
        require_coherent_trajectory=True,
        require_receiver_run=True,
        require_player_path=True,
        require_same_player_dribble=True,
    )
    dribble_bridge_output = ROOT / "evaluation" / "results" / "baseline_same_player_dribble_gap_bridge.csv"
    model.write_submission(dribble_bridge_output, dribble_bridge_labels)
    print(dribble_bridge_output)

    restart_labels = restart_provisional_labels(
        temporal,
        labels,
        minimum_active_run=int(rules["restart"]["minimum_active_run_frames"]),
        minimum_dead_run=int(rules["restart"]["minimum_dead_run_frames"]),
        control_run=8,
        post_restart_hold_frames=int(rules["restart"]["post_restart_hold_frames"]),
        post_restart_hold_frames_by_type={
            str(key): int(value)
            for key, value in rules["restart"].get(
                "post_restart_hold_frames_by_type", {}
            ).items()
        },
    )
    restart_output = (
        ROOT
        / "evaluation"
        / "results"
        / "baseline_restart_type_specific_interpolated_flight.csv"
    )
    model.write_submission(restart_output, restart_labels)
    print(restart_output)

    dead_ball_labels = dead_ball_boundary_evidence(
        temporal,
        labels,
        low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
        low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
        minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
        maximum_out_of_bounds_bridge_frames=int(
            rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]
        ),
    )
    dead_ball_output = (
        ROOT
        / "evaluation"
        / "results"
        / "baseline_dead_ball_boundary_low_motion.csv"
    )
    model.write_submission(dead_ball_output, dead_ball_labels)
    print(dead_ball_output)

    switch_variants = {
        "baseline_switch_confirmation": {"minimum_score": 0.65, "minimum_run_frames": 5},
        "baseline_switch_confirmation_strict": {"minimum_score": 0.75, "minimum_run_frames": 5},
    }
    for variant_name, variant_kwargs in switch_variants.items():
        switch_labels = apply_switch_confirmation(
            temporal,
            dead_ball_labels,
            **variant_kwargs,
        )
        switch_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(switch_output, switch_labels)
        print(switch_output)

    hysteresis_variants = {
        "baseline_switch_confirmation_anchor_guard": {"merge_spikes": False},
        "baseline_switch_confirmation_anchor_guard_merge": {"merge_spikes": True},
    }
    for variant_name, variant_kwargs in hysteresis_variants.items():
        guarded_labels = apply_switch_confirmation_hysteresis(
            temporal,
            dead_ball_labels,
            minimum_score=0.65,
            candidate_run_frames=5,
            minimum_anchor_run_frames=8,
            anchor_lookahead_frames=75,
            **variant_kwargs,
        )
        guarded_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(guarded_output, guarded_labels)
        print(guarded_output)

    cross_team_labels = resolve_cross_team_bidirectional_switch(
        temporal,
        dead_ball_labels,
        max_anchor_gap_frames=75,
        minimum_anchor_confidence=0.85,
        minimum_switch_score=0.65,
        minimum_switch_run_frames=5,
    )
    cross_team_output = (
        ROOT / "evaluation" / "results" / "baseline_cross_team_bidirectional_switch.csv"
    )
    model.write_submission(cross_team_output, cross_team_labels)
    print(cross_team_output)

    reception_switch_variants = {
        "baseline_reception_switch_evidence": {
            "minimum_reception_score": 0.55,
            "minimum_post_control_frames": 3,
            "require_strict_reception": False,
        },
        "baseline_reception_switch_persistent": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "require_strict_reception": False,
        },
        "baseline_reception_switch_strict": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "require_strict_reception": True,
        },
    }
    for variant_name, variant_kwargs in reception_switch_variants.items():
        reception_switch_labels = resolve_reception_switch_inside_interval(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        reception_switch_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(reception_switch_output, reception_switch_labels)
        print(reception_switch_output)

    enhanced_reception_switch_variants = {
        "baseline_enhanced_reception_switch": {
            "minimum_reception_score": 0.55,
            "minimum_post_control_frames": 3,
            "use_enhanced_reception_signal": True,
        },
        "baseline_enhanced_reception_switch_persistent": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "use_enhanced_reception_signal": True,
        },
        "baseline_enhanced_reception_switch_sustained": {
            "minimum_reception_score": 0.65,
            "minimum_post_control_frames": 8,
            "use_enhanced_reception_signal": True,
            "require_sustained_reception_signal": True,
        },
    }
    for variant_name, variant_kwargs in enhanced_reception_switch_variants.items():
        enhanced_labels = resolve_reception_switch_inside_interval(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        enhanced_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(enhanced_output, enhanced_labels)
        print(enhanced_output)

    episode_variants = {
        "baseline_stable_reception_episode_switch_comfort": (3, 0.90, 0.0),
        "baseline_stable_reception_episode_switch_comfort_strict": (8, 0.95, 0.0),
        "baseline_stable_reception_episode_switch_post_control": (8, 0.95, 0.5),
    }
    for variant_name, (minimum_episode_frames, minimum_receiver_confidence, minimum_receiver_near_ball_fraction) in episode_variants.items():
        episode_labels = apply_stable_reception_episode_switch(
            temporal,
            dead_ball_labels,
            minimum_episode_frames=minimum_episode_frames,
            minimum_stability=0.75,
            minimum_receiver_confidence=minimum_receiver_confidence,
            minimum_receiver_near_ball_fraction=minimum_receiver_near_ball_fraction,
        )
        episode_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(episode_output, episode_labels)
        print(episode_output)

    restart_boundary_variants = {
        "baseline_restart_hinted_control_boundary": {
            "use_tracking_fallback": False,
            "minimum_control_run": 8,
        },
        "baseline_restart_tracking_control_boundary": {
            "use_tracking_fallback": True,
            "minimum_control_run": 8,
        },
        "baseline_restart_tracking_control_boundary_strict": {
            "use_tracking_fallback": True,
            "minimum_control_run": 12,
        },
    }
    for variant_name, variant_kwargs in restart_boundary_variants.items():
        boundary_labels = restart_control_boundary_labels(
            temporal,
            dead_ball_labels,
            maximum_live_search_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        boundary_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(boundary_output, boundary_labels)
        print(boundary_output)

    hierarchy_variants = {
        "baseline_hierarchical_restart_fill_hinted": {"use_tracking_fallback": False},
        "baseline_hierarchical_restart_fill_tracking": {"use_tracking_fallback": True},
    }
    for variant_name, variant_kwargs in hierarchy_variants.items():
        hierarchy_labels = hierarchical_restart_fill_labels(
            temporal,
            dead_ball_labels,
            minimum_control_run=8,
            maximum_live_search_frames=75,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        hierarchy_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(hierarchy_output, hierarchy_labels)
        print(hierarchy_output)

    reception_variants = {
        "baseline_reception_path_agreement": {
            "require_receiver_path_agreement": True,
        },
        "baseline_reception_receiver_persistence": {
            "minimum_receiver_persistence_frames": 5,
        },
        "baseline_reception_path_persistence": {
            "require_receiver_path_agreement": True,
            "minimum_receiver_persistence_frames": 5,
        },
    }
    for variant_name, variant_kwargs in reception_variants.items():
        raw_variant = scenario_gated_labels(
            interpolated,
            allow_receiver_switch=True,
            allow_bidirectional_bridge=True,
            receiver_horizon_frames=75,
            maximum_internal_spike=75,
            **variant_kwargs,
        )
        raw_variant = _remove_short_dead_runs(raw_variant, minimum_run=25)
        variant_labels = dead_ball_boundary_evidence(
            temporal,
            raw_variant,
            low_activity_median_mps=float(rules["dead_ball_evidence"]["low_activity_median_mps"]),
            low_activity_p90_mps=float(rules["dead_ball_evidence"]["low_activity_p90_mps"]),
            minimum_low_motion_frames=int(rules["dead_ball_evidence"]["minimum_low_motion_frames"]),
            maximum_out_of_bounds_bridge_frames=int(
                rules["dead_ball_evidence"]["maximum_out_of_bounds_bridge_frames"]
            ),
        )
        variant_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(variant_output, variant_labels)
        print(variant_output)

    strict_reception_labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        maximum_internal_spike=75,
        require_strict_reception=True,
    )
    strict_reception_output = (
        ROOT / "evaluation" / "results" / "baseline_strict_reception_guard.csv"
    )
    model.write_submission(strict_reception_output, strict_reception_labels)
    print(strict_reception_output)

    # Diagnostic candidate: use the anchor-only future-touch forecast as an
    # additional veto/confirmation for a receiver switch. The forecast never
    # labels a frame by itself; it only authorises a switch already supported
    # by the bounded reception logic.
    forecast_confirmed_labels = scenario_gated_labels(
        interpolated,
        allow_receiver_switch=True,
        allow_bidirectional_bridge=True,
        receiver_horizon_frames=75,
        require_anchor_forecast_confirmation=True,
        maximum_internal_spike=75,
    )
    forecast_confirmed_labels = _remove_short_dead_runs(
        forecast_confirmed_labels, minimum_run=25
    )
    forecast_confirmed_output = (
        ROOT / "evaluation" / "results" / "baseline_anchor_forecast_confirmation.csv"
    )
    model.write_submission(forecast_confirmed_output, forecast_confirmed_labels)
    print(forecast_confirmed_output)

    # Candidate: repair only suspicious scenario intervals when two bounded,
    # independent stable-control anchors agree on the same team.
    resolver_variants = {
        "baseline_bounded_bidirectional_short": {
            "max_anchor_gap_frames": 25,
            "minimum_interval_frames": 1,
        },
        "baseline_bounded_bidirectional_long_ball": {
            "max_anchor_gap_frames": 75,
            "minimum_interval_frames": 26,
            "long_ball_only": True,
        },
    }
    for variant_name, variant_kwargs in resolver_variants.items():
        candidate = bounded_bidirectional_scenario_resolver(
            temporal,
            dead_ball_labels,
            minimum_anchor_confidence=0.85,
            **variant_kwargs,
        )
        candidate_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(candidate_output, candidate)
        print(candidate_output)

    # Stage 1: expose scenario confidence without changing the official labels.
    scored_features = scenario_confidence_scores(interpolated)
    confidence_columns = [
        "frame", "scenario_context", "scenario_context_causal", "scenario_episode_id",
        "scenario_episode_length_frames", "scenario_episode_position_frames",
        "scenario_episode_fraction", "scenario_episode_start", "scenario_episode_end",
        "scenario_state", "scenario_confidence", "scenario_margin",
        "state_score_H_CONTROL", "state_score_A_CONTROL", "state_score_DEAD_BALL",
        "state_score_BALL_IN_FLIGHT", "state_score_CONTESTED", "state_score_OCCLUDED",
        "state_score_RECEPTION", "reception_score_H", "reception_score_A",
        "post_control_score_H", "post_control_score_A",
        "directional_reception_score_H", "directional_reception_score_A",
        "contest_score", "decoder_eligible",
    ]
    confidence_output = ROOT / "evaluation" / "reports" / "scenario_confidence_scores.csv"
    confidence_output.parent.mkdir(parents=True, exist_ok=True)
    scored_features.loc[:, [column for column in confidence_columns if column in scored_features]].to_csv(
        confidence_output, index=False
    )
    print(confidence_output)

    # Evidence audit for missing-ball intervals. These columns are diagnostic
    # only: they make the dribble/pass/deflection/tackle interpretation
    # inspectable without converting an occlusion into a hand-written label.
    missing_ball_columns = [
        "frame", "ball_gap_id", "ball_gap_frames", "ball_gap_pre_speed_mps",
        "ball_gap_post_speed_mps", "ball_gap_speed_change_mps",
        "ball_gap_sharp_acceleration_risk", "gap_player_path_supported",
        "gap_player_path_team", "gap_player_speed_before_mps",
        "gap_player_speed_after_mps", "gap_player_ball_speed_match_mps",
        "gap_same_player_dribble_candidate", "gap_tackle_risk_candidate",
        "ball_direction_change_rad", "closest_distance_m",
        "candidate_opponent_density_5m", "opponent_closing_speed_mps",
        "scenario_context",
    ]
    missing_ball_audit_output = ROOT / "evaluation" / "reports" / "missing_ball_evidence_audit.csv"
    scored_features.loc[
        :, [column for column in missing_ball_columns if column in scored_features]
    ].loc[
        scored_features["ball_gap_id"].notna()
        if "ball_gap_id" in scored_features
        else scored_features["ball_x"].isna() | scored_features["ball_y"].isna()
    ].to_csv(missing_ball_audit_output, index=False)
    print(missing_ball_audit_output)

    scenario_summary = (
        scored_features.groupby("scenario_context_causal", dropna=False)
        .agg(
            frames=("frame", "size"),
            episodes=("scenario_episode_id", "nunique"),
            median_episode_frames=("scenario_episode_length_frames", "median"),
            decoder_eligible_frames=("decoder_eligible", "sum"),
        )
        .reset_index()
        .sort_values(["frames", "scenario_context_causal"], ascending=[False, True])
    )
    scenario_summary_output = ROOT / "evaluation" / "reports" / "scenario_summary.csv"
    scenario_summary.to_csv(scenario_summary_output, index=False)
    print(scenario_summary_output)

    reception_audit = reception_transition_alignment_audit(temporal)
    reception_audit_output = ROOT / "evaluation" / "reports" / "reception_transition_alignment.csv"
    reception_audit.to_csv(reception_audit_output, index=False)
    reception_audit_summary = pd.DataFrame(
        {
            "metric": [
                "candidate_frames",
                "opposite_anchor_pairs",
                "candidate_matches_after_team",
                "opposite_and_matching_candidates",
            ],
            "value": [
                len(reception_audit),
                int(reception_audit["opposite_anchor_pair"].sum()) if not reception_audit.empty else 0,
                int(reception_audit["candidate_matches_after_team"].sum()) if not reception_audit.empty else 0,
                int((reception_audit["opposite_anchor_pair"] & reception_audit["candidate_matches_after_team"]).sum()) if not reception_audit.empty else 0,
            ],
        }
    )
    reception_audit_summary_output = ROOT / "evaluation" / "reports" / "reception_transition_alignment_summary.csv"
    reception_audit_summary.to_csv(reception_audit_summary_output, index=False)
    print(reception_audit_output)
    print(reception_audit_summary_output)

    aligned_switch_variants = {
        "baseline_aligned_reception_switch": {"use_sustained_signal": False},
        "baseline_aligned_sustained_reception_switch": {"use_sustained_signal": True},
        # Smart Tagging team metadata is a weak directional cue. It is only
        # allowed to support a switch already backed by opposite anchors and
        # reception geometry; it cannot create a label by itself.
        "baseline_aligned_reception_smart_team_context": {
            "use_sustained_signal": False,
            "use_team_context": True,
        },
        "baseline_aligned_physics_veto": {
            "use_sustained_signal": False,
            "use_physics_veto": True,
        },
    }
    for variant_name, variant_kwargs in aligned_switch_variants.items():
        aligned_labels = resolve_aligned_reception_switch(
            temporal,
            dead_ball_labels,
            max_anchor_gap_frames=75,
            minimum_anchor_confidence=0.85,
            minimum_reception_score=0.55,
            **variant_kwargs,
        )
        aligned_output = ROOT / "evaluation" / "results" / f"{variant_name}.csv"
        model.write_submission(aligned_output, aligned_labels)
        print(aligned_output)

    # Stage 2: conservative duration-aware decoding on top of those scores.
    hsmm_labels = duration_aware_scenario_decode(
        scored_features,
        dead_ball_labels,
        minimum_team_run=8,
        switch_penalty=0.35,
        base_evidence_weight=0.75,
    )
    hsmm_output = ROOT / "evaluation" / "results" / "baseline_scenario_confidence_duration.csv"
    model.write_submission(hsmm_output, hsmm_labels)
    print(hsmm_output)

    # Future-looking reception decoding is intentionally not part of the
    # active runner. The implementation remains available as a documented
    # research branch for an explicitly offline smoother/live system with
    # permitted latency; it must not enter a causal submission unnoticed.


if __name__ == "__main__":
    main()

