# Barcelona hackathon: ball possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label every frame of Barcelona v Athletic as `H` / `A` / `D`.

## Public helper API

The pipeline is intentionally layered. Use the public façade in `possession/pipeline.py` from notebooks and scripts:

- `build_evidence_table()` — load the supplied data and calculate the complete frame-level feature table;
- `detect_scenarios()` — attach scenario confidence and episode evidence;
- `run_decision_pipeline()` — apply the configured anchor, transition, and decoder policy;
- `PipelineSpec` — declare the decision policy without duplicating implementation code.

Lower-level calculations remain in `possession/calculations/`, scenario rules in `possession/scenarios/`, and model composition in `possession/models/`. The notebooks are presentation and audit clients of these helpers; they should not become a second implementation of the pipeline.

## Modular implementation

The notebooks call the canonical runner; the runner composes reusable functions:

- `ClosestTeamRuleModel.build_features()` — load data, audit empty columns, calculate frame time and player/ball distances;
- `add_temporal_features()` — lagged, rolling, persistence, ball movement, and pressure signals;
- `add_reachability_features()` — closing speed, projected path, arrival time, and arrival margin;
- `add_anchor_features()` and `add_special_anchor_features()` — stable H/A and restart evidence;
- `apply_anchor_hierarchy()` — choose restart control, pressure control, stable control, or no anchor in configured order;
- `PipelineSpec` and `apply_pipeline_spec()` — choose anchors, scenarios, transitions, decoders, and protection rules per benchmark;
- `add_reception_signal_features()` — local minima, deceleration, persistence, and receiver agreement;
- `classify_scenarios()` and `build_scenario_episodes()` — scenario context and connected trajectories;
- `dead_ball_boundary_evidence()` — protect D using Smart Tagging plus physical activity evidence;
- `resolve_aligned_reception_switch()` — bounded bidirectional reception-based switch resolution;
- `run_evaluation.py` — display public sanity metrics without fitting on public labels.

Thresholds and windows are configuration-backed in `config/possession_rules.yaml`. Decoder alternatives are registered in `possession/models/pipelines.py`: `none`, `viterbi`, `selective_viterbi`, `hmm`, `hsmm`, `bayesian_hmm`, and `duration_viterbi`.

## Loading the data

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_load_data.ipynb
```

- `possession/load_data.py` reads the match into numpy/pandas: ball and player positions in metres, team per track, Smart Tagging, homography and the public 5-minute ground truth. Row `i` is video frame `i`, the same as a submission row.
- `notebooks/01_load_data.ipynb` walks through it with plots.


## Open Bundesliga data (for training)

7 Bundesliga matches with per-frame ball possession and in-play flags: the IDSSE dataset (Bassek et al., *Scientific Data* 2025, CC-BY 4.0).

- Download: https://huggingface.co/datasets/pysport/idsse-data
- Load with [kloppy](https://github.com/PySport/kloppy): `kloppy.sportec.load_open_tracking_data(match_id=...)`
- Paper: https://doi.org/10.1038/s41597-025-04505-y

## Credits

`games.py`, `starter/` and `evaluation/gt/` are from the challenge repository, MIT licence (see `LICENSE-more-than-a-hack`).
