# 03 — Scenario engineering

- Scenarios are evidence bundles, not labels: `DEAD`, `RESTART_LIVE_BOUNDARY`, `BALL_IN_FLIGHT`, `RECEPTION_CANDIDATE`, `CONTESTED`, `PRESSURED_CONTROL`, `CONTROLLED`, and `OCCLUDED`.
- A football-pitch view shows the ball trajectory, player paths, nearest-player geometry, and local pressure/density.
- The lower timeline shows scenario observations and the public H/A/D bar when available.
- Ground truth is displayed only as a post-hoc audit; it does not enter feature construction or transition decisions.
- A nearest player is not automatically the receiver: the path must intersect the ball trajectory and be supported by subsequent control evidence.
- The transition case is kept provisional when both teams remain close or the ball remains in flight.

