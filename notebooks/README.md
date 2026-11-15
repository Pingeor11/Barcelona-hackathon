# Possession Lab submission path

Read the notebooks in this order:

0. [00_load_data.ipynb](./00_load_data.ipynb) — locate the problem-repository data directory and load the shared data objects.
1. [01_eda.ipynb](./01_eda.ipynb) — data, public five-minute scope, Smart Tagging value and limits.
2. [02_feature_engineering.ipynb](./02_feature_engineering.ipynb) — geometry, speed, acceleration, pressure, reachability and missing-ball evidence.
3. [03_scenario_engineering.ipynb](./03_scenario_engineering.ipynb) — dead ball, control, flight, reception, long-ball and contested scenarios.
4. [04_anchors_rulebook.ipynb](./04_anchors_rulebook.ipynb) — observable evidence for easy H/A/D anchors.
5. [05_transition_markers.ipynb](./05_transition_markers.ipynb) — transition windows and unresolved contest evidence.
6. [06_viterbi_model.ipynb](./06_viterbi_model.ipynb) — sequence decoding after evidence engineering.
7. [07_run_pipeline.ipynb](./07_run_pipeline.ipynb) — reproducible pipeline execution.
8. [08_experimental_results.ipynb](./08_experimental_results.ipynb) — comparison of models, scenarios and rejected experiments.

Core conclusion: the observations are useful; mapping them to confidence and
H/A/D labels is the bottleneck. No fixed D hold duration is used.

## Appendix

The [appendix/](./appendix/) folder contains supporting definitions, the
organiser-facing evidence report, the rule-book detail, retired experiments,
and the sample-data restart audit. It is not required for the main walkthrough.

## Helper-first execution

The notebooks call the public pipeline helpers rather than reimplementing calculations. The pipeline is deliberately only three calls: features, scenarios, then model:

```python
from possession.pipeline import build_evidence_table, detect_scenarios, run_decision_pipeline, select_model

features = build_evidence_table()
features = detect_scenarios(features)
model = select_model("protected_aligned_reception")
features, labels = run_decision_pipeline(features, spec=model)
```

Plots are audit views over the same feature table. A visual observation is not a hand-written label.

## Implementation

The notebook does not duplicate the detailed theory. Reusable functions live in:

- [calculations/](../possession/calculations/) — one canonical `build_engineered_features()` composition;
- [scenarios/](../possession/scenarios/) — public scenario, anchor, reception, and boundary API;
- [models/](../possession/models/) — public submission-model and decoder API;
- [baseline_pipeline.py](../possession/baseline_pipeline.py) — loading, auditing, frame time, distances, and submission writing;
- [temporal_rules.py](../possession/temporal_rules.py) — temporal features, pressure, reachability, anchors, receptions, and scenarios;
- [decoder_experiments.py](../possession/decoder_experiments.py) — confidence evidence, bounded resolvers, and decoders;
- [scenario_decoder.py](../possession/scenario_decoder.py) — scenario-level decoding helpers;
- [broadcast_alignment.py](../possession/broadcast_alignment.py) — generic broadcast/tactical alignment audit;
- [../config/possession_rules.yaml](../config/possession_rules.yaml) — configuration-backed thresholds and windows.

The root modules remain as compatibility implementations for existing scripts.
New code should import through the three packages so every model receives the
same complete engineered feature table.
