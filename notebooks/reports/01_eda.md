# 01 — Data and signal audit

- **Data available:** tactical ATD tracking, Smart Tagging, homography, metadata, broadcast assets, and a public five-minute H/A/D audit interval.
- **Coordinate system:** player and ball positions are pitch coordinates in metres; frame index is the submission index.
- **Useful signals:** ball position/speed, player position/speed, missingness, team assignment, Smart Tagging context, and ground-truth comparison.
- **Key limitation:** Smart Tagging describes match context; it is evidence, not possession ground truth.
- **Missing-ball interpretation:** missing observations are retained as uncertainty intervals, not silently converted into possession.
- **Presentation rule:** every football-pitch plot shows observed ball path, player paths, and scenario/timeline evidence together.

