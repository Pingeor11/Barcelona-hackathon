import numpy as np
import pandas as pd

# Load raw tracking files (skipping the 3 Metrica header rows)
df_home = pd.read_csv('Sample_Game_1_RawTrackingData_Home_Team.csv', skiprows=3, header=None)
df_away = pd.read_csv('Sample_Game_1_RawTrackingData_Away_Team.csv', skiprows=3, header=None)

# Verify frame alignment
assert (df_home[0] == df_away[0]).all(), "Period mismatch"
assert (df_home[1] == df_away[1]).all(), "Frame mismatch"

# Build exact 193-column schema
target_cols = [
    'period_id', 'timestamp', 'frame_id',
    'ball_state', 'ball_owning_team_id',
    'ball_x', 'ball_y', 'ball_z', 'ball_speed'
]
for i in range(1, 47):
    target_cols.extend([f'{i}_x', f'{i}_y', f'{i}_d', f'{i}_s'])

n = len(df_home)
data = {col: np.full(n, np.nan, dtype=object) for col in target_cols}

# Populate metadata & ball coordinates
data['period_id'] = df_home[0].to_numpy()
data['timestamp'] = df_home[2].to_numpy()
data['frame_id'] = df_home[1].to_numpy()
data['ball_x'] = df_home[31].to_numpy(dtype=np.float64)
data['ball_y'] = df_home[32].to_numpy(dtype=np.float64)

# Populate Home players (shirt numbers in row 1 of Home CSV)
home_players = [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]
for idx, p_num in enumerate(home_players):
    data[f'{p_num}_x'] = df_home[3 + idx * 2].to_numpy(dtype=np.float64)
    data[f'{p_num}_y'] = df_home[4 + idx * 2].to_numpy(dtype=np.float64)

# Populate Away players (shirt numbers in row 1 of Away CSV)
away_players = [25, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26]
for idx, p_num in enumerate(away_players):
    data[f'{p_num}_x'] = df_away[3 + idx * 2].to_numpy(dtype=np.float64)
    data[f'{p_num}_y'] = df_away[4 + idx * 2].to_numpy(dtype=np.float64)

# Assemble and save combined CSV
out_df = pd.DataFrame(data, columns=target_cols)
out_df.to_csv('Sample_Game_1_Combined_TrackingData.csv', index=False)
print(f"Combined CSV saved with shape: {out_df.shape}")