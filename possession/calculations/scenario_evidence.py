"""Reusable scenario-window evidence for diagnostics and model features.

The functions in this module do not reconstruct a hidden ball path.  They link the
observed player slots across a scenario window and retain missing positions as NaN.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _slot_columns(features: pd.DataFrame) -> list[int]:
    slots = []
    for column in features.columns:
        if column.endswith("_x") and column[:-2].isdigit():
            slot = int(column[:-2])
            if f"{slot}_y" in features:
                slots.append(slot)
    return sorted(set(slots))


def build_scenario_evidence(
    features: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    team_by_slot: Mapping[int, str] | None = None,
    *,
    radius_m: float = 12.0,
    pitch_length_m: float = 105.3,
    pitch_width_m: float = 68.0,
) -> dict[str, pd.DataFrame | dict]:
    """Return linked player trajectories and summary evidence for one window.

    A slot is *significant* when it is observed within ``radius_m`` of an observed
    ball position at least once in the window.  All rows for that slot are then
    retained, including missing player positions.  This makes the output suitable
    for animation and for downstream observations without pretending that a slot
    is continuously tracked or that the ball was observed between samples.
    """
    frame = pd.to_numeric(features["frame"], errors="coerce")
    window = features.loc[frame.between(int(start_frame), int(end_frame))].copy()
    if window.empty:
        empty = pd.DataFrame()
        return {"frame_table": empty, "trajectory_table": empty, "summary": {}}

    slots = _slot_columns(window)
    bx = pd.to_numeric(window["ball_x"], errors="coerce").to_numpy(float)
    by = pd.to_numeric(window["ball_y"], errors="coerce").to_numpy(float)
    ball_valid = np.isfinite(bx) & np.isfinite(by)
    distances: dict[int, np.ndarray] = {}
    significant: list[int] = []
    summary_rows: list[dict] = []

    for slot in slots:
        px = pd.to_numeric(window[f"{slot}_x"], errors="coerce").to_numpy(float)
        py = pd.to_numeric(window[f"{slot}_y"], errors="coerce").to_numpy(float)
        distance = np.hypot((px - bx) * pitch_length_m, (py - by) * pitch_width_m)
        distance[~ball_valid] = np.nan
        distances[slot] = distance
        near = np.isfinite(distance) & (distance <= radius_m)
        if not near.any():
            continue
        significant.append(slot)
        team = (team_by_slot or {}).get(slot, "?")
        summary_rows.append({
            "slot": slot,
            "team": team,
            "min_distance_m": float(np.nanmin(distance)),
            "frames_near_ball": int(near.sum()),
            "first_near_frame": int(window.loc[near, "frame"].iloc[0]),
            "last_near_frame": int(window.loc[near, "frame"].iloc[-1]),
            "visible_fraction": float(window[f"{slot}_x"].notna().mean()),
        })

    frame_table = window[[c for c in [
        "frame", "match_clock_s", "ball_x", "ball_y", "scenario_context",
        "scenario_episode_id", "closest_team", "closest_distance_m",
        "ball_speed_derived_mps", "ball_acceleration_mps2",
    ] if c in window]].copy()
    trajectory_rows = []
    for index, row in window.iterrows():
        for slot in significant:
            x = pd.to_numeric(row.get(f"{slot}_x"), errors="coerce")
            y = pd.to_numeric(row.get(f"{slot}_y"), errors="coerce")
            trajectory_rows.append({
                "frame": int(row["frame"]),
                "match_clock_s": row.get("match_clock_s", np.nan),
                "slot": slot,
                "team": (team_by_slot or {}).get(slot, "?"),
                "x_m": float(x * pitch_length_m) if pd.notna(x) else np.nan,
                "y_m": float(y * pitch_width_m) if pd.notna(y) else np.nan,
                "distance_to_ball_m": distances[slot][window.index.get_loc(index)],
                "visible": bool(pd.notna(x) and pd.notna(y)),
            })
    trajectory_table = pd.DataFrame(trajectory_rows)
    summary = {
        "start_frame": int(window["frame"].min()),
        "end_frame": int(window["frame"].max()),
        "radius_m": float(radius_m),
        "significant_slots": significant,
        "significant_player_count": len(significant),
        "home_significant_count": sum((team_by_slot or {}).get(s) == "H" for s in significant),
        "away_significant_count": sum((team_by_slot or {}).get(s) == "A" for s in significant),
    }
    return {
        "frame_table": frame_table,
        "trajectory_table": trajectory_table,
        "summary_table": pd.DataFrame(summary_rows),
        "summary": summary,
    }


def add_scenario_evidence_features(
    features: pd.DataFrame,
    team_by_slot: Mapping[int, str],
    *,
    radius_m: float = 12.0,
) -> pd.DataFrame:
    """Add per-frame significant-player evidence for each derived scenario episode.

    These are observations consumed by models; they are not possession labels.
    The same scenario and anchor detectors are therefore available to every model.
    """
    out = features.copy()
    defaults = {
        "scenario_significant_player_count": 0,
        "scenario_home_significant_count": 0,
        "scenario_away_significant_count": 0,
        "scenario_near_ball_home_count": 0,
        "scenario_near_ball_away_count": 0,
    }
    for column, value in defaults.items():
        out[column] = value
    if "scenario_episode_id" not in out:
        return out

    # Distances are computed once for the full table.  Rebuilding a dataframe for
    # every short episode is both unnecessary and very costly on a full match.
    slots = _slot_columns(out)
    bx = pd.to_numeric(out["ball_x"], errors="coerce").to_numpy(float)
    by = pd.to_numeric(out["ball_y"], errors="coerce").to_numpy(float)
    ball_valid = np.isfinite(bx) & np.isfinite(by)
    distance_matrix = np.full((len(out), len(slots)), np.nan, dtype=float)
    for column, slot in enumerate(slots):
        px = pd.to_numeric(out[f"{slot}_x"], errors="coerce").to_numpy(float)
        py = pd.to_numeric(out[f"{slot}_y"], errors="coerce").to_numpy(float)
        distance_matrix[:, column] = np.hypot(
            (px - bx) * 105.3, (py - by) * 68.0
        )
    distance_matrix[~ball_valid, :] = np.nan

    for _, episode in out.groupby("scenario_episode_id", sort=False):
        if episode.empty:
            continue
        positions = out.index.get_indexer(episode.index)
        episode_distances = distance_matrix[positions]
        if len(slots):
            # Inf fill avoids noisy all-NaN warnings for halftime/dead frames.
            episode_min = np.min(
                np.where(np.isfinite(episode_distances), episode_distances, np.inf),
                axis=0,
            )
            significant_columns = np.where(episode_min <= radius_m)[0]
        else:
            significant_columns = np.array([], dtype=int)
        significant = [slots[column] for column in significant_columns]
        out.loc[episode.index, "scenario_significant_player_count"] = len(significant)
        home_columns = [column for column in significant_columns if team_by_slot.get(slots[column]) == "H"]
        away_columns = [column for column in significant_columns if team_by_slot.get(slots[column]) == "A"]
        if home_columns:
            out.loc[episode.index, "scenario_near_ball_home_count"] = (
                np.isfinite(episode_distances[:, home_columns])
                & (episode_distances[:, home_columns] <= radius_m)
            ).sum(axis=1)
        if away_columns:
            out.loc[episode.index, "scenario_near_ball_away_count"] = (
                np.isfinite(episode_distances[:, away_columns])
                & (episode_distances[:, away_columns] <= radius_m)
            ).sum(axis=1)
        out.loc[episode.index, "scenario_home_significant_count"] = len(home_columns)
        out.loc[episode.index, "scenario_away_significant_count"] = len(away_columns)
    out["scenario_player_density_near_ball"] = (
        out["scenario_near_ball_home_count"] + out["scenario_near_ball_away_count"]
    )
    out["scenario_receiver_path_support_H"] = (
        out["scenario_near_ball_home_count"] > 0
    )
    out["scenario_receiver_path_support_A"] = (
        out["scenario_near_ball_away_count"] > 0
    )
    return out
