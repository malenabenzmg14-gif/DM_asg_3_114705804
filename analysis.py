import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#Import everything from the real solution
from solution import (
    RANDOM_STATE, SEQ_LEN, NUM_CLASSES, RAW_COLS,
    TRAIN_DIRS, TEST_DIRS,
    seed_everything, find_dir,
    load_train,
    add_channels, normalize_channels,
    extract_stat_features,
    generate_kernels, rocket_transform,
    build_models, hard_vote,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

# Config
# Matches N_KERNELS in solution.py so that every figure (incl. the pipeline
# diagram and feature-dimension labels) reflects the exact configuration used
# by the final, submitted pipeline — keeping report and code consistent.
N_KERNELS_ANALYSIS = 5000
N_SPLITS = 5

OUT = "report_figures"
os.makedirs(OUT, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

# HELPERS

def save(name):
    plt.savefig(
        os.path.join(OUT, f"{name}.pdf"),
        bbox_inches="tight"
    )
    plt.close()

    print(f"  Saved {name}.pdf")


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

# Load raw training data (same function as solution.py)

print("=" * 60)
print("STEP 1: Loading training data")
print("=" * 60)

seed_everything(RANDOM_STATE)
train_dir = find_dir(TRAIN_DIRS)

X_raw, y, groups = load_train(train_dir)
print(f"  Loaded {len(y)} samples, shape {X_raw.shape}")
print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

#Build full feature matrix (same pipeline as solution.py)

print("\n" + "=" * 60)
print("STEP 2: Building feature matrix (same pipeline as solution.py)")
print("=" * 60)

print("  Adding derived channels...")
X_seq = add_channels(X_raw)

# For normalisation we use a dummy test set (copy of train) — only train
# statistics matter; the scaler is fitted on train only.
X_seq_norm, _ = normalize_channels(X_seq, X_seq.copy())

print("  Extracting statistical features...")
X_stat = extract_stat_features(X_seq_norm)

print(f"  Generating {N_KERNELS_ANALYSIS} ROCKET kernels...")
kernels = generate_kernels(N_KERNELS_ANALYSIS, X_seq_norm.shape[2], seed=RANDOM_STATE)

print("  Applying ROCKET transform (this takes a while)...")
X_rocket = rocket_transform(X_seq_norm, kernels)

X_full = np.hstack([X_rocket, X_stat]).astype(np.float32)
X_full[~np.isfinite(X_full)] = 0.0
print(f"  Final feature shape: {X_full.shape}")

#Ablation CV  (real numbers for the report)

print("\n" + "=" * 60)
print("STEP 3: Ablation cross-validation")
print("=" * 60)

splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                random_state=RANDOM_STATE)

# We evaluate each ablation stage on the same folds.
# Feature matrices at each stage:
X_raw_flat = X_raw.reshape(len(X_raw), -1).astype(np.float32)   # 6 raw features flattened
X_seq_flat = X_seq.reshape(len(X_seq), -1).astype(np.float32)   # 17 ch, no norm
X_norm_flat = X_seq_norm.reshape(len(X_seq_norm), -1).astype(np.float32)  # 17 ch, normalised

ablation_stages = {
    "Raw 6ch + Ridge":         X_raw_flat,
    "+ Channel engineering":   X_seq_flat,
    "+ Normalisation":         X_norm_flat,
    "+ Stat features":         np.hstack([X_norm_flat, X_stat]).astype(np.float32),
    "+ ROCKET (full)":         X_full,
}

ablation_results = {}  # stage -> list of fold F1s

for stage_name, X_stage in ablation_stages.items():
    scores = []
    for tr, va in splitter.split(X_stage, y, groups):
        clf = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10),
                              class_weight="balanced"),
        )
        clf.fit(X_stage[tr], y[tr])
        scores.append(macro_f1(y[va], clf.predict(X_stage[va])))
    ablation_results[stage_name] = scores
    print(f"  {stage_name:35s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Model comparison: Ridge vs LogReg vs ExtraTrees vs Ensemble (on full features)
print("\n  Model comparison on full feature set:")
model_results = {}

for fold_idx, (tr, va) in enumerate(splitter.split(X_full, y, groups)):
    ridge, logreg, et = build_models()
    ridge.fit(X_full[tr],  y[tr])
    logreg.fit(X_full[tr], y[tr])
    et.fit(X_full[tr],     y[tr])

    p_r = ridge.predict(X_full[va])
    p_l = logreg.predict(X_full[va])
    p_e = et.predict(X_full[va])
    p_v = hard_vote([p_r, p_l, p_e])

    for name, pred in [("Ridge", p_r), ("LogReg", p_l),
                       ("ExtraTrees", p_e), ("Ensemble", p_v)]:
        model_results.setdefault(name, []).append(macro_f1(y[va], pred))

for name, scores in model_results.items():
    print(f"  {name:12s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

#Generate all figures

print("\n" + "=" * 60)
print("STEP 4: Generating figures")
print("=" * 60)


#Fig 1: Class distribution
fig, ax = plt.subplots(figsize=(7, 3.8))
classes, counts = np.unique(y, return_counts=True)
bars = ax.bar(classes, counts, color=COLORS, edgecolor="white", zorder=3)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{cnt}\n({cnt/len(y)*100:.1f}%)", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_xticks(classes)
ax.set_xticklabels([f"Activity {c}" for c in classes], fontsize=10)
ax.set_ylabel("Number of samples", fontsize=11)
ax.set_title("Class Distribution in Training Set", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, alpha=0.4, zorder=0); ax.set_axisbelow(True)
ax.set_ylim(0, max(counts) * 1.18)
plt.tight_layout(); save("fig_class_dist")


#Fig 2: Per-class time series (mean magnitude)
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
t = np.arange(SEQ_LEN)
for cls in range(6):
    ax = axes[cls]
    X_cls = X_raw[y == cls]
    mag = np.sqrt(X_cls[:, :, 0] ** 2 + X_cls[:, :, 1] ** 2 + X_cls[:, :, 2] ** 2)
    ax.plot(t, mag.mean(axis=0), color=COLORS[cls], lw=1.8)
    ax.fill_between(t, mag.mean(0) - mag.std(0), mag.mean(0) + mag.std(0),
                    color=COLORS[cls], alpha=0.2)
    ax.set_title(f"Activity {cls}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=8); ax.set_ylabel(r"$|\mathbf{a}|$ (g)", fontsize=8)
    ax.yaxis.grid(True, alpha=0.3); ax.set_xlim(0, 299)
plt.suptitle("Mean Acceleration Magnitude per Activity (mean ± 1σ)",
             fontsize=12, fontweight="bold")
plt.tight_layout(); save("fig_timeseries")


#Fig 3: Boxplots mean channels
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col, lbl in zip(axes, [0, 1, 2], ["mean_x", "mean_y", "mean_z"]):
    bp = ax.boxplot([X_raw[y == c, :, col].flatten() for c in range(6)],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", markersize=1.5, alpha=0.3))
    for patch, col_ in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col_); patch.set_alpha(0.75)
    ax.set_xticklabels([f"Act.{c}" for c in range(6)], rotation=25,
                       ha="right", fontsize=8)
    ax.set_title(lbl.upper(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Acceleration (g)", fontsize=9)
    ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
plt.suptitle("Mean Acceleration per Axis and Activity", fontsize=12, fontweight="bold")
plt.tight_layout(); save("fig_boxplots")


#Fig 4: Boxplots std channels
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, col, lbl in zip(axes, [3, 4, 5], ["std_x", "std_y", "std_z"]):
    bp = ax.boxplot([X_raw[y == c, :, col].flatten() for c in range(6)],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2),
                    flierprops=dict(marker=".", markersize=1.5, alpha=0.3))
    for patch, col_ in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col_); patch.set_alpha(0.75)
    ax.set_xticklabels([f"Act.{c}" for c in range(6)], rotation=25,
                       ha="right", fontsize=8)
    ax.set_title(lbl.upper(), fontsize=11, fontweight="bold")
    ax.set_ylabel("Std. Deviation (g)", fontsize=9)
    ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
plt.suptitle("Acceleration Std. Deviation per Axis and Activity",
             fontsize=12, fontweight="bold")
plt.tight_layout(); save("fig_std_boxplots")


#Fig 5: Correlation matrix
import pandas as pd
feats_mean = X_raw.mean(axis=1)  # (N, 6)
corr = pd.DataFrame(feats_mean, columns=RAW_COLS).corr().values
fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(RAW_COLS, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(RAW_COLS, fontsize=9)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if abs(corr[i, j]) > 0.5 else "black")
ax.set_title("Feature Correlation Matrix (temporal means)", fontsize=10, fontweight="bold")
plt.tight_layout(); save("fig_corr")


#Fig 6: Temporal variability 
mag_all = np.sqrt(X_raw[:, :, 0]**2 + X_raw[:, :, 1]**2 + X_raw[:, :, 2]**2)
mag_std = mag_all.std(axis=1)
fig, ax = plt.subplots(figsize=(7, 4))
bp = ax.boxplot([mag_std[y == c] for c in range(6)], patch_artist=True,
                medianprops=dict(color="black", lw=2),
                flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch, col in zip(bp["boxes"], COLORS):
    patch.set_facecolor(col); patch.set_alpha(0.75)
ax.set_xticklabels([f"Activity {c}" for c in range(6)], fontsize=10)
ax.set_ylabel("Temporal std of |acceleration| (g)", fontsize=10)
ax.set_title("Temporal Variability of Acceleration Magnitude per Activity",
             fontsize=11, fontweight="bold")
ax.yaxis.grid(True, alpha=0.35); ax.set_axisbelow(True)
plt.tight_layout(); save("fig_temporal_std")


# Fig 7: Ablation bar chart (REAL CV values)
stage_names = list(ablation_results.keys())
stage_means = [np.mean(ablation_results[s]) for s in stage_names]
bar_colors  = ["#aaaaaa", "#7fbfff", "#4C72B0", "#DD8452", "#55A868"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(stage_names, stage_means, color=bar_colors,
              edgecolor="white", linewidth=0.8, zorder=3)
for bar, val in zip(bars, stage_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.axhline(0.1201, color="red",    lw=1.4, ls="--", label="Baseline-1 (0.1201)", zorder=4)
ax.axhline(0.6130, color="orange", lw=1.4, ls="--", label="Baseline-2 (0.6130)", zorder=4)
ax.axhline(0.7088, color="green",  lw=1.4, ls="--", label="Baseline-3 (0.7088)", zorder=4)
ax.set_ylabel("Macro F1-Score (5-fold group CV)", fontsize=11)
ax.set_title("Ablation Study: Cumulative Feature Engineering Impact\n(real cross-validation scores)",
             fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)
ax.set_ylim(0, min(1.0, max(stage_means) * 1.18))
ax.legend(fontsize=9, loc="upper left")
plt.tight_layout(); save("fig_ablation_full")


#Fig 8: Model comparison bar chart (REAL CV values)
model_names  = list(model_results.keys())
model_means  = [np.mean(model_results[m]) for m in model_names]
model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(6.5, 4))
bars = ax.bar(model_names, model_means, color=model_colors,
              edgecolor="white", linewidth=0.8, zorder=3, width=0.5)
for bar, val in zip(bars, model_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(0.7088, color="green", lw=1.6, ls="--",
           label="Baseline-3 (0.7088)", zorder=4)
ax.set_ylabel("Macro F1-Score (5-fold group CV)", fontsize=11)
ax.set_title("Individual vs. Ensemble Model Performance\n(real cross-validation scores)",
             fontsize=11, fontweight="bold")
ax.yaxis.grid(True, alpha=0.35, zorder=0); ax.set_axisbelow(True)
ax.set_ylim(max(0, min(model_means) * 0.93), min(1.0, max(model_means) * 1.08))
ax.legend(fontsize=10)
plt.tight_layout(); save("fig_model_comparison")


#Fig 9: Pipeline diagram
fig, ax = plt.subplots(figsize=(13, 2.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")

# Derive the ROCKET / concat feature-dimension labels directly from
# N_KERNELS_ANALYSIS (== N_KERNELS in solution.py) so the diagram can never
# drift out of sync with the actual submitted pipeline again.
_rocket_feats = N_KERNELS_ANALYSIS * 3
_total_feats = _rocket_feats + 810
_fmt_k = lambda n: f"{n/1000:g}k"

boxes = [
    (0.3,  "Raw CSV\n(300×6)"),
    (2.0,  "Channel\nEngineering\n(300×17)"),
    (3.8,  "Standard\nScaler\n(per ch.)"),
    (5.6,  "Statistical\nFeatures\n(810-dim)"),
    (7.4,  f"ROCKET\n{_fmt_k(N_KERNELS_ANALYSIS)} kernels\n({_fmt_k(_rocket_feats)} feat.)"),
    (9.2,  f"Concat\n({_fmt_k(_rocket_feats)}+810\n= {_fmt_k(_total_feats)})"),
    (11.0, "Ridge\n(balanced)\nfinal model"),
    (12.6, "submission\n_final.csv"),
]
fc_map = {"ROCKET": "#FFF3CD", "Ridge": "#D4EDDA", "submission": "#F8D7DA"}
for x, label in boxes:
    fc = next((v for k, v in fc_map.items() if k in label), "#E8F4FD")
    rect = mpatches.FancyBboxPatch((x - 0.75, 0.5), 1.4, 2.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=fc, edgecolor="#555", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x, 1.5, label, ha="center", va="center", fontsize=7.5, fontweight="bold")
for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i + 1][0] - 0.75, 1.5),
                xytext=(boxes[i][0] + 0.75, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
ax.set_title("End-to-End Processing Pipeline", fontsize=12, fontweight="bold", pad=8)
plt.tight_layout(); save("fig_pipeline")

# Print summary table for the report

print("\n" + "=" * 60)
print("SUMMARY  —  copy these into the report")
print("=" * 60)
print("\nAblation (Macro F1, 5-fold group CV):")
for stage, scores in ablation_results.items():
    print(f"  {stage:35s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print("\nModel comparison (Macro F1, 5-fold group CV):")
for name, scores in model_results.items():
    print(f"  {name:12s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

print(f"\nAll figures saved to:  {os.path.abspath(OUT)}/")
