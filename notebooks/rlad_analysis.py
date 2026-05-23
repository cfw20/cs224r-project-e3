import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
import os

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Load data
print("Loading data...")
df_a = pd.read_csv('track_a_history.csv')
df_b = pd.read_csv('track_b_history.csv')

with open('track_a_summary.json') as f:
    summary_a = json.load(f)
with open('track_b_summary.json') as f:
    summary_b = json.load(f)

print(f"Track A: {len(df_a)} rows, steps: {df_a['_step'].min():.0f} - {df_a['_step'].max():.0f}")
print(f"Track B: {len(df_b)} rows, steps: {df_b['_step'].min():.0f} - {df_b['_step'].max():.0f}")

# Filter to common step range
max_step = min(df_a['_step'].max(), df_b['_step'].max())
df_a_aligned = df_a[df_a['_step'] <= max_step].copy()
df_b_aligned = df_b[df_b['_step'] <= max_step].copy()

print(f"\nAligned to common step range: 0 - {max_step:.0f}")

# ============================================================================
# 1. Validation Accuracy Comparison
# ============================================================================
print("\n" + "="*70)
print("1. VALIDATION ACCURACY COMPARISON")
print("="*70)

val_a = df_a_aligned[df_a_aligned['val/openai/gsm8k/reward/mean'].notna()]
val_b = df_b_aligned[df_b_aligned['val/openai/gsm8k/reward/mean'].notna()]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(val_a['_step'], val_a['val/openai/gsm8k/reward/mean'], 'o-', 
        label='Track A (Clean)', linewidth=2, markersize=6, color='#e74c3c')
ax.plot(val_b['_step'], val_b['val/openai/gsm8k/reward/mean'], 's-', 
        label='Track B (Mixed)', linewidth=2, markersize=6, color='#2ecc71')

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Validation Accuracy (Reward Mean)', fontsize=12)
ax.set_title('Validation Accuracy: Track A (Clean) vs Track B (Mixed)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add annotations for final values
if len(val_a) > 0:
    ax.annotate(f"{val_a['val/openai/gsm8k/reward/mean'].iloc[-1]:.3f}", 
                xy=(val_a['_step'].iloc[-1], val_a['val/openai/gsm8k/reward/mean'].iloc[-1]),
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#e74c3c', alpha=0.3))

if len(val_b) > 0:
    ax.annotate(f"{val_b['val/openai/gsm8k/reward/mean'].iloc[-1]:.3f}", 
                xy=(val_b['_step'].iloc[-1], val_b['val/openai/gsm8k/reward/mean'].iloc[-1]),
                xytext=(10, -20), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#2ecc71', alpha=0.3))

plt.tight_layout()
plt.savefig('validation_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: validation_accuracy_comparison.png")

# Print step-by-step comparison
print("\nValidation Accuracy at Common Steps:")
print(f"{'Step':<8} {'Track A':<12} {'Track B':<12} {'Delta (B-A)':<15}")
print("-" * 50)
for step in sorted(set(val_a['_step']) & set(val_b['_step'])):
    a_val = val_a[val_a['_step'] == step]['val/openai/gsm8k/reward/mean'].values
    b_val = val_b[val_b['_step'] == step]['val/openai/gsm8k/reward/mean'].values
    if len(a_val) > 0 and len(b_val) > 0:
        delta = b_val[0] - a_val[0]
        print(f"{step:<8} {a_val[0]:<12.4f} {b_val[0]:<12.4f} {delta:<+15.4f}")

# ============================================================================
# 2. Timing Efficiency
# ============================================================================
print("\n" + "="*70)
print("2. TIMING EFFICIENCY ANALYSIS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

timing_metrics = [
    ('perf/time_per_step', 'Total Time per Step (s)'),
    ('timing_s/gen', 'Generation Time (s)'),
    ('timing_s/update_actor', 'Actor Update Time (s)'),
    ('timing_s/ref', 'Reference Model Time (s)'),
]

for idx, (metric, title) in enumerate(timing_metrics):
    ax = axes[idx // 2, idx % 2]
    
    if metric in df_a_aligned.columns:
        data_a = df_a_aligned[df_a_aligned[metric].notna()]
        ax.plot(data_a['_step'], data_a[metric], 'o-', label='Track A', alpha=0.7, color='#e74c3c')
    
    if metric in df_b_aligned.columns:
        data_b = df_b_aligned[df_b_aligned[metric].notna()]
        ax.plot(data_b['_step'], data_b[metric], 's-', label='Track B', alpha=0.7, color='#2ecc71')
    
    ax.set_xlabel('Step')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Timing Breakdown: Track A vs Track B', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('timing_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: timing_comparison.png")

# Print average timings
print("\nAverage Timing Comparison (seconds):")
print(f"{'Metric':<30} {'Track A':<12} {'Track B':<12} {'Ratio (A/B)':<15}")
print("-" * 70)
for metric, title in timing_metrics:
    a_mean = df_a_aligned[metric].mean() if metric in df_a_aligned.columns else np.nan
    b_mean = df_b_aligned[metric].mean() if metric in df_b_aligned.columns else np.nan
    ratio = a_mean / b_mean if b_mean > 0 else np.nan
    print(f"{title:<30} {a_mean:<12.2f} {b_mean:<12.2f} {ratio:<15.2f}")

# ============================================================================
# 3. Response Length Dynamics
# ============================================================================
print("\n" + "="*70)
print("3. RESPONSE LENGTH DYNAMICS")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Mean response length
ax = axes[0]
if 'response_length/mean' in df_a_aligned.columns:
    ax.plot(df_a_aligned['_step'], df_a_aligned['response_length/mean'], 
            'o-', label='Track A', alpha=0.7, color='#e74c3c')
if 'response_length/mean' in df_b_aligned.columns:
    ax.plot(df_b_aligned['_step'], df_b_aligned['response_length/mean'], 
            's-', label='Track B', alpha=0.7, color='#2ecc71')
ax.set_xlabel('Step')
ax.set_ylabel('Mean Response Length (tokens)')
ax.set_title('Mean Response Length Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

# Clip ratio
ax = axes[1]
if 'response_length/clip_ratio' in df_a_aligned.columns:
    ax.plot(df_a_aligned['_step'], df_a_aligned['response_length/clip_ratio'], 
            'o-', label='Track A', alpha=0.7, color='#e74c3c')
if 'response_length/clip_ratio' in df_b_aligned.columns:
    ax.plot(df_b_aligned['_step'], df_b_aligned['response_length/clip_ratio'], 
            's-', label='Track B', alpha=0.7, color='#2ecc71')
ax.set_xlabel('Step')
ax.set_ylabel('Clip Ratio (% at max length)')
ax.set_title('Response Length Clip Ratio Over Training')
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle('Response Length Dynamics: Track A vs Track B', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('response_length_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: response_length_comparison.png")

# Print key statistics
print("\nResponse Length Statistics (at final common step):")
common_steps = sorted(set(df_a_aligned['_step']) & set(df_b_aligned['_step']))
if common_steps:
    last_step = max(common_steps)
    a_last = df_a_aligned[df_a_aligned['_step'] == last_step]
    b_last = df_b_aligned[df_b_aligned['_step'] == last_step]
    
    print(f"Last common step: {last_step}")
    if 'response_length/mean' in a_last.columns:
        print(f"  Track A mean length: {a_last['response_length/mean'].values[0]:.1f} tokens")
    if 'response_length/mean' in b_last.columns:
        print(f"  Track B mean length: {b_last['response_length/mean'].values[0]:.1f} tokens")
    if 'response_length/clip_ratio' in a_last.columns:
        print(f"  Track A clip ratio: {a_last['response_length/clip_ratio'].values[0]:.1%}")
    if 'response_length/clip_ratio' in b_last.columns:
        print(f"  Track B clip ratio: {b_last['response_length/clip_ratio'].values[0]:.1%}")

# ============================================================================
# 4. Training Dynamics (Rewards, Advantages, Entropy)
# ============================================================================
print("\n" + "="*70)
print("4. TRAINING DYNAMICS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('critic/rewards/mean', 'Mean Training Reward'),
    ('critic/advantages/mean', 'Mean Advantage'),
    ('actor/entropy', 'Policy Entropy'),
    ('actor/pg_loss', 'Policy Gradient Loss'),
]

for idx, (metric, title) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    
    if metric in df_a_aligned.columns:
        data_a = df_a_aligned[df_a_aligned[metric].notna()]
        ax.plot(data_a['_step'], data_a[metric], 'o-', label='Track A', 
                alpha=0.7, markersize=4, color='#e74c3c')
    
    if metric in df_b_aligned.columns:
        data_b = df_b_aligned[df_b_aligned[metric].notna()]
        ax.plot(data_b['_step'], data_b[metric], 's-', label='Track B', 
                alpha=0.7, markersize=4, color='#2ecc71')
    
    ax.set_xlabel('Step')
    ax.set_ylabel(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Training Dynamics: Track A vs Track B', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('training_dynamics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: training_dynamics.png")

# ============================================================================
# 5. Memory Utilization
# ============================================================================
print("\n" + "="*70)
print("5. GPU MEMORY UTILIZATION")
print("="*70)

fig, ax = plt.subplots(figsize=(10, 5))

if 'perf/max_memory_allocated_gb' in df_a_aligned.columns:
    ax.plot(df_a_aligned['_step'], df_a_aligned['perf/max_memory_allocated_gb'], 
            'o-', label='Track A', alpha=0.7, color='#e74c3c')
if 'perf/max_memory_allocated_gb' in df_b_aligned.columns:
    ax.plot(df_b_aligned['_step'], df_b_aligned['perf/max_memory_allocated_gb'], 
            's-', label='Track B', alpha=0.7, color='#2ecc71')

ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='GPU Limit (80GB)')
ax.set_xlabel('Step')
ax.set_ylabel('Peak GPU Memory (GB)')
ax.set_title('GPU Memory Utilization: Track A vs Track B')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('memory_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: memory_comparison.png")

# Print memory stats
print("\nGPU Memory Statistics:")
if 'perf/max_memory_allocated_gb' in df_a_aligned.columns:
    print(f"  Track A peak: {df_a_aligned['perf/max_memory_allocated_gb'].max():.2f} GB")
if 'perf/max_memory_allocated_gb' in df_b_aligned.columns:
    print(f"  Track B peak: {df_b_aligned['perf/max_memory_allocated_gb'].max():.2f} GB")

# ============================================================================
# 6. Step-wise Accuracy Gap
# ============================================================================
print("\n" + "="*70)
print("6. STEP-WISE ACCURACY GAP")
print("="*70)

common_val_steps = sorted(set(val_a['_step']) & set(val_b['_step']))

gaps = []
for step in common_val_steps:
    a_val = val_a[val_a['_step'] == step]['val/openai/gsm8k/reward/mean'].values[0]
    b_val = val_b[val_b['_step'] == step]['val/openai/gsm8k/reward/mean'].values[0]
    gaps.append({'step': step, 'track_a': a_val, 'track_b': b_val, 'gap': b_val - a_val})

gap_df = pd.DataFrame(gaps)

# Plot gap
fig, ax = plt.subplots(figsize=(10, 5))

colors = ['green' if g > 0 else 'red' for g in gap_df['gap']]
ax.bar(gap_df['step'], gap_df['gap'], color=colors, alpha=0.7, width=8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Validation Step')
ax.set_ylabel('Accuracy Gap (Track B - Track A)')
ax.set_title('Step-wise Accuracy Gap: Track B vs Track A', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for _, row in gap_df.iterrows():
    ax.text(row['step'], row['gap'], f"{row['gap']:+.3f}", 
            ha='center', va='bottom' if row['gap'] > 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_gap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: accuracy_gap.png")

print("\nAccuracy Gap Summary:")
print(gap_df.to_string(index=False))
print(f"\nAverage gap: {gap_df['gap'].mean():+.4f}")
print(f"Max gap: {gap_df['gap'].max():+.4f} at step {gap_df.loc[gap_df['gap'].idxmax(), 'step']:.0f}")
print(f"Min gap: {gap_df['gap'].min():+.4f} at step {gap_df.loc[gap_df['gap'].idxmin(), 'step']:.0f}")

# ============================================================================
# 7. Final Summary Table
# ============================================================================
print("\n" + "="*70)
print("7. FINAL SUMMARY TABLE")
print("="*70)

summary_data = {
    'Metric': [
        'Total Steps',
        'Final Validation Accuracy',
        'Mean Response Length (final)',
        'Clip Ratio (final)',
        'Avg Step Time (s)',
        'Peak GPU Memory (GB)',
    ],
    'Track A (Clean)': [
        f"{df_a['_step'].max():.0f}",
        f"{val_a['val/openai/gsm8k/reward/mean'].iloc[-1]:.4f}" if len(val_a) > 0 else 'N/A',
        f"{df_a_aligned['response_length/mean'].iloc[-1]:.1f}" if 'response_length/mean' in df_a_aligned.columns else 'N/A',
        f"{df_a_aligned['response_length/clip_ratio'].iloc[-1]:.1%}" if 'response_length/clip_ratio' in df_a_aligned.columns else 'N/A',
        f"{df_a_aligned['perf/time_per_step'].mean():.1f}" if 'perf/time_per_step' in df_a_aligned.columns else 'N/A',
        f"{df_a_aligned['perf/max_memory_allocated_gb'].max():.2f}" if 'perf/max_memory_allocated_gb' in df_a_aligned.columns else 'N/A',
    ],
    'Track B (Mixed)': [
        f"{df_b['_step'].max():.0f}",
        f"{val_b['val/openai/gsm8k/reward/mean'].iloc[-1]:.4f}" if len(val_b) > 0 else 'N/A',
        f"{df_b_aligned['response_length/mean'].iloc[-1]:.1f}" if 'response_length/mean' in df_b_aligned.columns else 'N/A',
        f"{df_b_aligned['response_length/clip_ratio'].iloc[-1]:.1%}" if 'response_length/clip_ratio' in df_b_aligned.columns else 'N/A',
        f"{df_b_aligned['perf/time_per_step'].mean():.1f}" if 'perf/time_per_step' in df_b_aligned.columns else 'N/A',
        f"{df_b_aligned['perf/max_memory_allocated_gb'].max():.2f}" if 'perf/max_memory_allocated_gb' in df_b_aligned.columns else 'N/A',
    ],
}

summary_table = pd.DataFrame(summary_data)
print("\n=== FINAL SUMMARY TABLE ===")
print(summary_table.to_string(index=False))
summary_table.to_csv('summary_table.csv', index=False)
print("\nSummary saved to summary_table.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("  - validation_accuracy_comparison.png")
print("  - timing_comparison.png")
print("  - response_length_comparison.png")
print("  - training_dynamics.png")
print("  - memory_comparison.png")
print("  - accuracy_gap.png")
print("  - summary_table.csv")
