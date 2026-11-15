# 02 — Feature engineering

## Observation → feature

- Ball movement is represented by position-derived speed, acceleration, direction, curvature, travelled distance, and missing spans.
- A kick candidate is associated with a sharp positive speed change; a control candidate with deceleration or a local speed minimum near a player path.
- A direction change is retained as ambiguous evidence: it may be a pass, deflection, miscontrol, tackle, or first-time pass.
- Player density is measured around the ball and at the projected landing area.
- Closing speed and arrival-time margin describe whether players are converging on the ball and who can plausibly arrive first.
- Team distance margin, pressure, passing-space availability, and subsequent control describe whether a receiver can be anchored.

## Scenario use

- **Short action:** near-player path agreement plus low pressure supports controlled reception.
- **Long ball:** distance travelled, landing speed, trajectory coherence, density, and approach speed are evaluated together.
- **Contested:** high density and competing arrival times keep the state unresolved.
- **Missing ball:** endpoint geometry, acceleration continuity, player-path agreement, and reappearance evidence determine whether a bridge is plausible.

The notebook includes the pitch trajectories, speed/acceleration panels, missing-span overlays, kick-distance plots, long-ball studies, restart audits, and transition timelines.

