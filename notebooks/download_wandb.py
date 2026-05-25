import wandb
import pandas as pd
import json
import os

api = wandb.Api()
project = 'cfw20-stanford-university/rlad-noise-control'

print("Fetching runs...")
runs = api.runs(project)

# Find runs by looking at all runs and picking the ones with most steps
track_a_candidates = []
track_b_candidates = []

for r in runs:
    steps = r.summary.get('_step', 0)
    if r.name == 'qwen3-1p7b-gsm8k-grpo-clean':
        track_a_candidates.append((r, steps))
    elif r.name == 'qwen3-1p7b-gsm8k-grpo-mixed':
        track_b_candidates.append((r, steps))

print(f"Track A candidates: {[(r.name, steps) for r, steps in track_a_candidates]}")
print(f"Track B candidates: {[(r.name, steps) for r, steps in track_b_candidates]}")

# Pick the ones with the most steps
track_a_run = max(track_a_candidates, key=lambda x: x[1])[0] if track_a_candidates else None
track_b_run = max(track_b_candidates, key=lambda x: x[1])[0] if track_b_candidates else None

print(f"\nSelected Track A: {track_a_run.name if track_a_run else 'NOT FOUND'} (steps={track_a_run.summary.get('_step', 'N/A') if track_a_run else 'N/A'})")
print(f"Selected Track B: {track_b_run.name if track_b_run else 'NOT FOUND'} (steps={track_b_run.summary.get('_step', 'N/A') if track_b_run else 'N/A'})")

if track_a_run and track_b_run:
    # Download histories
    print("\nDownloading Track A history...")
    # Track A was resumed twice, so we combine the histories of the relevant runs to get the full 0-400 trajectory.
    run_ids = ['go4zdpgb', 'a72jwsb8', 'fgw6prw4']
    hist_a_list = []
    for rid in run_ids:
        try:
            r_obj = api.run(f"{project}/{rid}")
            h = r_obj.history(samples=10000)
            hist_a_list.append(h)
            print(f"  Loaded run {rid}: {len(h)} rows")
        except Exception as e:
            print(f"  Warning: failed to load run {rid}: {e}")
    if hist_a_list:
        hist_a = pd.concat(hist_a_list)
        hist_a = hist_a.sort_values('_step').drop_duplicates(subset=['_step'], keep='last')
    else:
        hist_a = track_a_run.history(samples=10000)
    hist_a.to_csv('track_a_history.csv', index=False)
    print(f"Track A: {len(hist_a)} rows saved to track_a_history.csv")
    
    print("\nDownloading Track B history...")
    hist_b = track_b_run.history(samples=10000)
    hist_b.to_csv('track_b_history.csv', index=False)
    print(f"Track B: {len(hist_b)} rows saved to track_b_history.csv")
    
    # Also download summary metrics
    summary_a = dict(track_a_run.summary)
    summary_b = dict(track_b_run.summary)
    
    with open('track_a_summary.json', 'w') as f:
        json.dump(summary_a, f, indent=2, default=str)
    with open('track_b_summary.json', 'w') as f:
        json.dump(summary_b, f, indent=2, default=str)
    
    print("\nAll data downloaded successfully!")
else:
    print("Could not find both runs.")
