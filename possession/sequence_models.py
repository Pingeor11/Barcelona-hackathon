"""Small, inspectable sequence models seeded by algorithmic anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from possession.temporal_rules import load_rules
from possession.decoder_experiments import confidence_multiscale_labels, _merge_internal_spikes


STATES = ("H", "A", "D", "X")
FEATURES = (
    "team_distance_margin_m",
    "closest_distance_m",
    "ball_speed_derived_mps",
    "candidate_pressure_m",
)


def _value(rules: dict[str, Any], *keys: str) -> Any:
    value: Any = rules
    for key in keys:
        value = value[key]
    return value


def _feature_matrix(features: pd.DataFrame) -> np.ndarray:
    matrix = np.column_stack([
        pd.to_numeric(features.get(column, pd.Series(np.nan, index=features.index)), errors="coerce").to_numpy(float)
        for column in FEATURES
    ])
    return matrix


@dataclass
class AnchorInitializedHMM:
    """Diagonal-Gaussian HMM initialized from algorithmic anchor evidence.

    This is not trained on the public ground truth. H/A emissions are seeded
    from stable algorithmic anchors, D from Smart Tagging set-piece context,
    and X from the remaining observations. It is an honest first HMM baseline;
    EM can refine these parameters later.
    """

    rules: dict[str, Any] | None = None

    def fit(self, features: pd.DataFrame) -> "AnchorInitializedHMM":
        self.rules = load_rules() if self.rules is None else self.rules
        x = _feature_matrix(features)
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
        masks = {
            "H": anchor == "H",
            "A": anchor == "A",
            "D": set_piece,
            "X": ~(anchor == "H") & ~(anchor == "A") & ~set_piece,
        }
        self.means = np.zeros((len(STATES), x.shape[1]), dtype=float)
        self.scales = np.ones((len(STATES), x.shape[1]), dtype=float)
        global_median = np.nanmedian(x, axis=0)
        global_scale = np.nanmedian(np.abs(x - global_median), axis=0) * 1.4826
        global_scale = np.where(np.isfinite(global_scale) & (global_scale > 1e-6), global_scale, 1.0)
        for state_index, state in enumerate(STATES):
            subset = x[masks[state]]
            for feature_index in range(x.shape[1]):
                values = subset[:, feature_index] if len(subset) else np.array([])
                values = values[np.isfinite(values)]
                if len(values):
                    self.means[state_index, feature_index] = np.median(values)
                    scale = np.median(np.abs(values - self.means[state_index, feature_index])) * 1.4826
                    self.scales[state_index, feature_index] = max(float(scale), 0.25)
                else:
                    self.means[state_index, feature_index] = global_median[feature_index]
                    self.scales[state_index, feature_index] = global_scale[feature_index]
        self.global_median = global_median
        self.transition_logp = self._transition_log_probabilities()
        self.start_logp = np.log(np.full(len(STATES), 1.0 / len(STATES)))
        return self

    def _transition_log_probabilities(self) -> np.ndarray:
        self_p = float(_value(self.rules, "hmm", "self_transition"))
        switch_p = float(_value(self.rules, "hmm", "team_switch_transition"))
        dead_p = float(_value(self.rules, "hmm", "dead_ball_transition"))
        matrix = np.full((len(STATES), len(STATES)), 1e-8, dtype=float)
        h, a, d, unknown = range(4)
        matrix[h, h] = self_p
        matrix[a, a] = self_p
        matrix[h, a] = switch_p
        matrix[a, h] = switch_p
        matrix[h, d] = dead_p
        matrix[a, d] = dead_p
        matrix[d, d] = self_p
        matrix[d, h] = (1 - self_p) / 2
        matrix[d, a] = (1 - self_p) / 2
        matrix[unknown, :] = 1 / len(STATES)
        matrix[:, unknown] = np.maximum(matrix[:, unknown], 0.002)
        matrix /= matrix.sum(axis=1, keepdims=True)
        return np.log(matrix)

    def _emission_logp(self, features: pd.DataFrame) -> np.ndarray:
        x = _feature_matrix(features)
        filled = np.where(np.isfinite(x), x, self.global_median)
        residual = (filled[:, None, :] - self.means[None, :, :]) / self.scales[None, :, :]
        logp = -0.5 * np.sum(residual**2 + 2 * np.log(self.scales[None, :, :]), axis=2)
        set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
        logp[set_piece, :] -= 2.0
        logp[set_piece, STATES.index("D")] += 12.0
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        for state_index, state in enumerate(("H", "A")):
            logp[anchor == state, :] -= 1.5
            logp[anchor == state, state_index] += 8.0
        return logp

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "means"):
            self.fit(features)
        emission = self._emission_logp(features)
        n_frames, n_states = emission.shape
        score = np.empty((n_frames, n_states), dtype=float)
        back = np.zeros((n_frames, n_states), dtype=np.int8)
        score[0] = self.start_logp + emission[0]
        for frame in range(1, n_frames):
            candidates = score[frame - 1][:, None] + self.transition_logp
            back[frame] = np.argmax(candidates, axis=0)
            score[frame] = candidates[back[frame], np.arange(n_states)] + emission[frame]
        state_index = int(np.argmax(score[-1]))
        path = np.empty(n_frames, dtype=np.int8)
        for frame in range(n_frames - 1, -1, -1):
            path[frame] = state_index
            state_index = int(back[frame, state_index]) if frame else state_index
        labels = np.asarray(STATES, dtype="<U1")[path]
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        labels[~graded] = "X"
        return labels


@dataclass
class AnchorInitializedHSMM:
    """Duration-aware wrapper around the algorithmic anchor evidence.

    This first HSMM-style experiment does not fit hidden states from human
    labels. It applies explicit semi-Markov duration constraints to the HMM
    path: very short H/A spikes are treated as observations/noise, and short
    unknown gaps between the same team are bridged. The constraints are kept
    in YAML so this remains an inspectable experiment rather than an implicit
    smoothing trick.
    """

    rules: dict[str, Any] | None = None

    def fit(self, features: pd.DataFrame) -> "AnchorInitializedHSMM":
        self.rules = load_rules() if self.rules is None else self.rules
        self.hmm = AnchorInitializedHMM(self.rules).fit(features)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "hmm"):
            self.fit(features)
        labels = self.hmm.predict(features).astype(object)
        max_spike = int(_value(self.rules, "decoder", "max_team_spike_frames"))
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)

        # A short team run surrounded by the other team is not allowed to
        # create a new possession spell by itself.
        start = 0
        while start < len(labels):
            end = start + 1
            while end < len(labels) and labels[end] == labels[start]:
                end += 1
            if (
                labels[start] in {"H", "A"}
                and end - start <= max_spike
                and start > 0
                and end < len(labels)
                and labels[start - 1] == labels[end]
            ):
                labels[start:end] = labels[start - 1]
            start = end

        # A short unobserved interval between matching team states is one
        # semi-Markov spell, unless a set-piece context interrupts it.
        start = 0
        while start < len(labels):
            if labels[start] != "X":
                start += 1
                continue
            end = start + 1
            while end < len(labels) and labels[end] == "X":
                end += 1
            gap = end - start
            if (
                gap <= max_spike
                and start > 0
                and end < len(labels)
                and labels[start - 1] in {"H", "A"}
                and labels[start - 1] == labels[end]
                and not set_piece[start:end].any()
            ):
                labels[start:end] = labels[start - 1]
            start = end

        # Hard evidence always wins over the learned path.
        labels[set_piece & graded] = "D"
        for team in ("H", "A"):
            labels[(anchor == team) & graded] = team
        labels[~graded] = "X"
        return labels.astype("<U1")


@dataclass
class BayesianAnchorHMM:
    """MAP-style HMM using anchor transition counts with a Dirichlet prior.

    This is deliberately a diagnostic model.  It estimates transition
    probabilities from algorithmic anchor evidence, not from public labels.
    The posterior mean prevents sparse anchor counts from producing zero
    transitions.  Viterbi then decodes the most probable complete path.
    """

    rules: dict[str, Any] | None = None
    dirichlet_alpha: float = 1.0

    def fit(self, features: pd.DataFrame) -> "BayesianAnchorHMM":
        self.rules = load_rules() if self.rules is None else self.rules
        self.base = AnchorInitializedHMM(self.rules).fit(features)
        from possession.decoder_experiments import confidence_multiscale_labels

        seed = confidence_multiscale_labels(
            features,
            short_window=50,
            long_window=125,
            disagreement_margin_m=2.5,
            minimum_switch_run=50,
            maximum_internal_spike=50,
            dead_ball_mode="tagged_slow",
        )
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        restart = features.get("restart_confirmed_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
        trusted = np.isin(anchor, ["H", "A"]) | restart
        trusted &= np.isin(seed, list(STATES))
        counts = np.full((len(STATES), len(STATES)), self.dirichlet_alpha, dtype=float)
        state_index = {state: index for index, state in enumerate(STATES)}
        for frame in range(1, len(seed)):
            if trusted[frame - 1] and trusted[frame]:
                counts[state_index[seed[frame - 1]], state_index[seed[frame]]] += 1.0
        self.transition_logp = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.start_logp = np.log(np.full(len(STATES), 1.0 / len(STATES)))
        return self

    def _emission_logp(self, features: pd.DataFrame) -> np.ndarray:
        emission = self.base._emission_logp(features)
        from possession.decoder_experiments import confidence_multiscale_labels

        seed = confidence_multiscale_labels(
            features,
            short_window=50,
            long_window=125,
            disagreement_margin_m=2.5,
            minimum_switch_run=50,
            maximum_internal_spike=50,
            dead_ball_mode="tagged_slow",
        )
        for state_index, state in enumerate(STATES):
            emission[seed == state, state_index] += 3.0
        return emission

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "transition_logp"):
            self.fit(features)
        emission = self._emission_logp(features)
        n_frames, n_states = emission.shape
        score = np.empty((n_frames, n_states), dtype=float)
        back = np.zeros((n_frames, n_states), dtype=np.int8)
        score[0] = self.start_logp + emission[0]
        for frame in range(1, n_frames):
            candidates = score[frame - 1][:, None] + self.transition_logp
            back[frame] = np.argmax(candidates, axis=0)
            score[frame] = candidates[back[frame], np.arange(n_states)] + emission[frame]
        state_index = int(np.argmax(score[-1]))
        path = np.empty(n_frames, dtype=np.int8)
        for frame in range(n_frames - 1, -1, -1):
            path[frame] = state_index
            state_index = int(back[frame, state_index]) if frame else state_index
        labels = np.asarray(STATES, dtype="<U1")[path]
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        labels[~graded] = "X"
        return labels


@dataclass
class RuleConstrainedViterbi:
    """Use Viterbi only to smooth H/A evidence while rules remain authoritative.

    Unlike the first HMM, this model does not let an ``X`` emission dominate
    graded play.  The rule decoder supplies a soft seed, continuous distance
    margin supplies the observation score, and Smart Tagging/confirmed anchors
    are applied as hard constraints after decoding.
    """

    rules: dict[str, Any] | None = None
    seed_weight: float = 2.5
    self_transition: float = 0.995

    def fit(self, features: pd.DataFrame) -> "RuleConstrainedViterbi":
        self.rules = load_rules() if self.rules is None else self.rules
        from possession.decoder_experiments import confidence_multiscale_labels

        self.seed = confidence_multiscale_labels(
            features,
            short_window=50,
            long_window=125,
            disagreement_margin_m=2.5,
            minimum_switch_run=50,
            maximum_internal_spike=50,
            dead_ball_mode="tagged_slow",
        )
        switch = max(1e-6, 1.0 - self.self_transition)
        transition = np.full((3, 3), switch / 2.0, dtype=float)
        np.fill_diagonal(transition, self.self_transition)
        self.transition_logp = np.log(transition / transition.sum(axis=1, keepdims=True))
        self.start_logp = np.log(np.full(3, 1.0 / 3.0))
        return self

    def _emission(self, features: pd.DataFrame) -> np.ndarray:
        margin = pd.to_numeric(features["team_distance_margin_m"], errors="coerce").to_numpy(float)
        margin = np.nan_to_num(margin, nan=0.0)
        score = np.column_stack([margin / 2.0, -margin / 2.0, np.zeros(len(features))])
        for state_index, state in enumerate(("H", "A", "D")):
            score[self.seed == state, state_index] += self.seed_weight
        speed = pd.to_numeric(features.get("ball_speed_derived_mps"), errors="coerce").to_numpy(float)
        dead = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool).copy()
        dead &= np.nan_to_num(speed, nan=0.0) <= 1.0
        score[dead, 2] += 4.0
        score[dead, :2] -= 2.0
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        score[anchor == "H", 0] += 3.0
        score[anchor == "A", 1] += 3.0
        return score

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "transition_logp"):
            self.fit(features)
        emission = self._emission(features)
        n_frames, n_states = emission.shape
        score = np.empty((n_frames, n_states), dtype=float)
        back = np.zeros((n_frames, n_states), dtype=np.int8)
        score[0] = self.start_logp + emission[0]
        for frame in range(1, n_frames):
            candidates = score[frame - 1][:, None] + self.transition_logp
            back[frame] = np.argmax(candidates, axis=0)
            score[frame] = candidates[back[frame], np.arange(n_states)] + emission[frame]
        state = int(np.argmax(score[-1]))
        path = np.empty(n_frames, dtype=np.int8)
        for frame in range(n_frames - 1, -1, -1):
            path[frame] = state
            state = int(back[frame, state]) if frame else state
        labels = np.asarray(("H", "A", "D"), dtype="<U1")[path]
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        labels[~graded] = "X"
        return labels


@dataclass
class SelectiveRuleViterbi:
    """Use Viterbi only where the rule decoder is genuinely uncertain.

    The adaptive rule decoder remains the primary decision-maker.  Viterbi is
    allowed to alter only ambiguous graded frames; high-confidence geometric
    anchors, restart-confirmed anchors, Smart Tagging dead-ball frames, and
    non-graded frames are protected.  This is the intended combination of
    symbolic evidence and temporal modelling for the current data regime.
    """

    rules: dict[str, Any] | None = None
    minimum_margin_m: float = 2.5
    minimum_closest_distance_m: float = 2.5
    minimum_persistence: float = 0.80

    def fit(self, features: pd.DataFrame) -> "SelectiveRuleViterbi":
        self.rules = load_rules() if self.rules is None else self.rules
        from possession.decoder_experiments import confidence_multiscale_labels

        self.rule_labels = confidence_multiscale_labels(
            features,
            short_window=50,
            long_window=125,
            disagreement_margin_m=2.5,
            minimum_switch_run=50,
            maximum_internal_spike=50,
            dead_ball_mode="tagged_slow",
        )
        self.viterbi = RuleConstrainedViterbi(self.rules).fit(features)
        return self

    def _stable_mask(self, features: pd.DataFrame) -> np.ndarray:
        margin = pd.to_numeric(features["team_distance_margin_m"], errors="coerce").to_numpy(float)
        closest = pd.to_numeric(features["closest_distance_m"], errors="coerce").to_numpy(float)
        persistence = np.maximum(
            pd.to_numeric(features.get("home_persistence"), errors="coerce").to_numpy(float),
            pd.to_numeric(features.get("away_persistence"), errors="coerce").to_numpy(float),
        )
        stable = (
            np.isfinite(margin)
            & (np.abs(margin) >= self.minimum_margin_m)
            & np.isfinite(closest)
            & (closest <= self.minimum_closest_distance_m)
            & (persistence >= self.minimum_persistence)
        )
        stable |= features.get("anchor_confident", pd.Series(False, index=features.index)).to_numpy(bool)
        stable |= features.get("restart_confirmed_anchor", pd.Series(False, index=features.index)).to_numpy(bool)
        return stable

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "rule_labels"):
            self.fit(features)
        candidate = self.viterbi.predict(features)
        labels = self.rule_labels.copy()
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        stable = self._stable_mask(features)
        ambiguous = graded & ~stable
        labels[ambiguous] = candidate[ambiguous]

        # Hard evidence is restored after the selective temporal pass.
        set_piece = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)
        labels[graded & set_piece] = "D"
        anchor = features.get("anchor_team", pd.Series(None, index=features.index)).to_numpy(object)
        for team in ("H", "A"):
            labels[graded & (anchor == team)] = team
        labels[~graded] = "X"
        return labels.astype("<U1")


@dataclass
class ScenarioHSMM:
    """Duration-aware scenario decoder built around the adaptive rule path.

    The model treats marked ball-flight intervals as one semi-Markov episode.
    During that episode, nearest-player crossings are not accepted as a
    possession switch by themselves.  The previous controlled team is carried
    through the flight unless bounded receiver evidence supports a handoff.
    This is a constrained scenario model, not a claim that hidden states have
    been learned from human labels.
    """

    rules: dict[str, Any] | None = None
    receiver_margin_s: float = 0.25
    minimum_flight_speed_mps: float | None = None
    use_restart_boundaries: bool = False
    minimum_restart_persistence: float = 0.60

    def fit(self, features: pd.DataFrame) -> "ScenarioHSMM":
        self.rules = load_rules() if self.rules is None else self.rules
        self.seed = confidence_multiscale_labels(
            features,
            short_window=50,
            long_window=125,
            disagreement_margin_m=2.5,
            minimum_switch_run=50,
            maximum_internal_spike=50,
            dead_ball_mode="tagged_slow",
        )
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "seed"):
            self.fit(features)
        labels = self.seed.copy().astype(object)
        flight = features.get("ball_flight_candidate", pd.Series(False, index=features.index)).to_numpy(bool).copy()
        genuine_reception = features.get(
            "genuine_reception_anchor", pd.Series(False, index=features.index)
        ).to_numpy(bool)
        speed = pd.to_numeric(
            features.get("ball_speed_derived_mps", pd.Series(np.nan, index=features.index)),
            errors="coerce",
        ).to_numpy(float)
        minimum_speed = self.minimum_flight_speed_mps
        if minimum_speed is None:
            minimum_speed = float(_value(self.rules, "bridging", "minimum_ball_speed_mps"))
        flight &= speed >= minimum_speed
        # The reception frame terminates the flight state; its rule label is
        # retained so the receiving team can begin the next controlled spell.
        flight &= ~genuine_reception
        graded = features.get("graded", pd.Series(True, index=features.index)).to_numpy(bool)
        likely = features.get("likely_receiver_team", pd.Series(None, index=features.index)).to_numpy(object)
        margin = pd.to_numeric(
            features.get("arrival_time_margin_s", pd.Series(np.nan, index=features.index)),
            errors="coerce",
        ).to_numpy(float)
        restart = features.get("smart_set_piece", pd.Series(False, index=features.index)).to_numpy(bool)

        if self.use_restart_boundaries:
            end_candidate = features.get(
                "restart_end_candidate", pd.Series(False, index=features.index)
            ).to_numpy(bool)
            hint = features.get(
                "restart_team_hint", pd.Series(None, index=features.index)
            ).to_numpy(object)
            anchor = features.get(
                "anchor_team", pd.Series(None, index=features.index)
            ).to_numpy(object)
            home_future = pd.to_numeric(
                features.get("home_future_anchor_persistence", pd.Series(0.0, index=features.index)),
                errors="coerce",
            ).to_numpy(float)
            away_future = pd.to_numeric(
                features.get("away_future_anchor_persistence", pd.Series(0.0, index=features.index)),
                errors="coerce",
            ).to_numpy(float)
            max_handoff = int(_value(self.rules, "bridging", "long_ball_max_frames"))
            for boundary in np.flatnonzero(end_candidate):
                team = hint[boundary]
                if team not in {"H", "A"}:
                    continue
                stop = min(len(labels), boundary + 1 + max_handoff)
                persistence = np.where(
                    team == "H",
                    home_future[boundary + 1 : stop],
                    away_future[boundary + 1 : stop],
                )
                future = np.flatnonzero(
                    (anchor[boundary + 1 : stop] == team)
                    & (persistence >= self.minimum_restart_persistence)
                )
                if len(future):
                    receiver = boundary + 1 + int(future[0])
                    # The tagged restart team owns the interval after the
                    # restart until sustained tracking control is observed.
                    labels[boundary + 1 : receiver + 1] = team

        start = 0
        while start < len(labels):
            if not flight[start] or not graded[start]:
                start += 1
                continue
            end = start + 1
            while end < len(labels) and flight[end] and graded[end]:
                end += 1
            previous = labels[start - 1] if start else None
            following = labels[end] if end < len(labels) else None
            if previous not in {"H", "A"}:
                start = end
                continue

            # A flight belongs to the previous controlled team unless a
            # receiver has a clear arrival-time advantage near its end.
            labels[start:end] = previous
            tail_start = max(start, end - 25)
            tail_likely = likely[tail_start:end]
            tail_margin = margin[tail_start:end]
            receiver_votes = []
            for team in ("H", "A"):
                count = int(np.sum(tail_likely == team))
                if count and team != previous:
                    receiver_votes.append((count, team))
            receiver_votes.sort(reverse=True)
            if receiver_votes:
                count, receiver = receiver_votes[0]
                receiver_margin = np.nanmedian(np.abs(tail_margin[np.isfinite(tail_margin)]))
                supported = count >= max(3, min(25, end - tail_start) // 3)
                supported &= np.isfinite(receiver_margin) and receiver_margin >= self.receiver_margin_s
                supported &= following == receiver or np.any(likely[end:min(len(labels), end + 25)] == receiver)
                if supported and not restart[start:end].any():
                    # Keep the previous team during flight; the following
                    # controlled state carries the actual reception switch.
                    pass
            start = end

        labels = _merge_internal_spikes(labels, 50)
        labels[restart & graded] = "D"
        labels[~graded] = "X"
        return labels.astype("<U1")
