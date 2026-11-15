# 06 — Sequence decoding

- Sequence decoding regularises frame-level evidence into football-consistent spells.
- Anchor evidence initializes likely H/A/D states; scenario evidence controls how readily a state may change.
- Ball flight, contest, missing observations, and restart boundaries are treated as uncertainty rather than forced possession.
- The decoder is evaluated against the same public interval as the rule-based candidates.
- The purpose is temporal coherence: fewer isolated flips, clearer spells, and transitions aligned with credible reception evidence.

