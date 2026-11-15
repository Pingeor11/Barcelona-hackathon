# 08 — Experimental results

- The executed benchmark contains the rule variants for restart handling, missing-ball bridges, long-ball resolution, reception persistence, physics vetoes, and transition confirmation.
- Best observed public-window candidate in this run: `baseline_aligned_physics_veto` / equivalent aligned reception-switch variants.
- Public-window result: Macro-F1 `0.773604`; Spell-F1 mean `0.567742`; H F1 `0.858487`; A F1 `0.830460`; D F1 `0.631866`.
- Transition F1 at the 25-frame tolerance: `0.333333`.
- The results show that frame-level H/A classification is stronger than transition timing; transition confirmation remains the main bottleneck.
- The conservative conclusion is to retain all engineered signals, but promote labels only when path, kinematics, team context, and subsequent control agree.
- Ground truth is used for evaluation and reporting only, not as a feature or labelling rule.

