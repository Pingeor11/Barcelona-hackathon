# Ball possession from automatic tracking — More than a Hack 2027

Labels every frame of the tactical video H (home in possession), A (away in possession) or
D (dead ball) with a hidden Markov model decoded by Viterbi. Method and rules: see the technical
document submitted with this repository.

## Reproduce the submission

1. Put the official data bundle in `data/` (see `data/README.md`). The match data is not, and must
   not be, in this repository.
2. Install and run (Python 3.9 or newer):

       python3 -m pip install -r solution/requirements.txt
       python3 solution/possession_model.py <team_name>.csv
       python3 evaluation/validate_submission.py <team_name>.csv

The run takes about a minute and is deterministic: running it again produces the identical file.

Optional checks (no ground truth needed): `python3 solution/compare.py <team_name>.csv` prints
whole-match possession counts, lengths and agreement with the automatic tags.

## Files

| file | what it does |
|---|---|
| `possession_model.py` | the model: per-frame control and dead-ball clues, Viterbi decoding. All settings are in `PARAMS` at the top |
| `features.py` | per-frame measurements: nearest player of each team to the ball, player and ball speeds, ball outside the pitch |
| `load.py` | reads the tracking file into arrays (cached as `cache_tactical.npz` next to the data) |
| `tag_check.py` | turns Smart Tagging team phases into a weak per-frame hint |
| `compare.py` | whole-match sanity report |

`games.py`, `starter/` and `evaluation/` come from the organisers' repository (MIT licence).
