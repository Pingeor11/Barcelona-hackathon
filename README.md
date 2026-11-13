# Barcelona hackathon: ball possession

Our entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label every frame of Barcelona v Athletic as `H` / `A` / `D`.

## Loading the data

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_load_data.ipynb
```

- `possession/load_data.py` reads the match into numpy/pandas: ball and player positions in metres, team per track, Smart Tagging, homography and the public 5-minute ground truth. Row `i` is video frame `i`, the same as a submission row.
- `notebooks/01_load_data.ipynb` walks through it with plots.

The match data isn't in this repo; it comes from the challenge enrolment. Put it in `data/` (with `tactical/` and `broadcast/` inside), next to the repo in `../data`, or point `MTAH_DATA_DIR` at it. `data/` and `cache/` are gitignored. Never commit match data.

## Open Bundesliga data (for training)

7 Bundesliga matches with per-frame ball possession and in-play flags: the IDSSE dataset (Bassek et al., *Scientific Data* 2025, CC-BY 4.0).

- Download: https://huggingface.co/datasets/pysport/idsse-data
- Load with [kloppy](https://github.com/PySport/kloppy): `kloppy.sportec.load_open_tracking_data(match_id=...)`
- Paper: https://doi.org/10.1038/s41597-025-04505-y

## Credits

`games.py`, `starter/` and `evaluation/gt/` are from the challenge repository, MIT licence (see `LICENSE-more-than-a-hack`).
