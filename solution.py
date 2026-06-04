import os
import random
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from scipy.ndimage import convolve1d

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
SEQ_LEN = 300
NUM_CLASSES = 6

TRAIN_DIRS = ["train/train", "train"]
TEST_DIRS = ["test/test", "test"]
SAMPLE_SUBMISSION = "sample_submission.csv"

RAW_COLS = ["mean_x", "mean_y", "mean_z", "std_x", "std_y", "std_z"]

# Increased from 5000 → 10000 for stronger ROCKET features
N_KERNELS = 10000

# Set to True to run group-based cross-validation before final training
RUN_VALIDATION = True
N_SPLITS = 5


# ============================================================
# UTILS
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def find_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Could not find any of: {candidates}")


def validate_df(df, path, train=True):
    required = ["index", "file_id"] + RAW_COLS
    if train:
        required.append("label")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    if len(df) != SEQ_LEN:
        raise ValueError(f"{path}: expected {SEQ_LEN} rows, got {len(df)}")

    if df["file_id"].nunique() != 1:
        raise ValueError(f"{path}: multiple file_id values")


# ============================================================
# DATA LOADING
# ============================================================

def load_train(train_dir):
    X, y, groups = [], [], []

    users = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )

    for user in users:
        user_path = os.path.join(train_dir, user)

        for file in sorted(f for f in os.listdir(user_path) if f.endswith(".csv")):
            path = os.path.join(user_path, file)
            df = pd.read_csv(path)
            validate_df(df, path, train=True)
            df = df.sort_values("index").reset_index(drop=True)

            file_id = int(df["file_id"].iloc[0])
            filename_id = int(file.replace(".csv", ""))
            if file_id != filename_id:
                raise ValueError(f"{path}: file_id mismatch")

            if df["label"].nunique() != 1:
                raise ValueError(f"{path}: multiple labels in one file")

            label = int(df["label"].iloc[0])
            if label < 0 or label > 5:
                raise ValueError(f"{path}: invalid label {label}")

            X.append(df[RAW_COLS].values.astype(np.float32))
            y.append(label)
            groups.append(user)

    return np.array(X), np.array(y), np.array(groups)


def load_test(test_dir):
    X, ids = [], []

    users = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )

    for user in users:
        user_path = os.path.join(test_dir, user)

        for file in sorted(f for f in os.listdir(user_path) if f.endswith(".csv")):
            path = os.path.join(user_path, file)
            df = pd.read_csv(path)
            validate_df(df, path, train=False)
            df = df.sort_values("index").reset_index(drop=True)

            file_id = int(df["file_id"].iloc[0])
            filename_id = int(file.replace(".csv", ""))
            if file_id != filename_id:
                raise ValueError(f"{path}: file_id mismatch")

            X.append(df[RAW_COLS].values.astype(np.float32))
            ids.append(file_id)

    return np.array(X), np.array(ids)


# ============================================================
# CHANNEL ENGINEERING
# ============================================================

def add_channels(X):
    """
    Extends the 6 raw channels with motion-derived features.
    Input:  (n, 300, 6)
    Output: (n, 300, 17)
    """
    mx, my, mz = X[:, :, 0], X[:, :, 1], X[:, :, 2]
    sx, sy, sz = X[:, :, 3], X[:, :, 4], X[:, :, 5]

    mag     = np.sqrt(mx**2 + my**2 + mz**2)
    std_mag = np.sqrt(sx**2 + sy**2 + sz**2)

    dmx  = np.diff(mx,  axis=1, prepend=mx[:, :1])
    dmy  = np.diff(my,  axis=1, prepend=my[:, :1])
    dmz  = np.diff(mz,  axis=1, prepend=mz[:, :1])
    dmag = np.diff(mag, axis=1, prepend=mag[:, :1])
    ddmag = np.diff(dmag, axis=1, prepend=dmag[:, :1])

    energy   = mx**2 + my**2 + mz**2
    ratio_xy = mx / (np.abs(my) + 1e-6)
    ratio_xz = mx / (np.abs(mz) + 1e-6)
    ratio_yz = my / (np.abs(mz) + 1e-6)

    extra = np.stack(
        [mag, std_mag, dmx, dmy, dmz, dmag, ddmag,
         energy, ratio_xy, ratio_xz, ratio_yz],
        axis=2,
    )

    out = np.concatenate([X, extra], axis=2).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def normalize_channels(X_train, X_test):
    n_train, t, c = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.reshape(-1, c)).reshape(n_train, t, c)
    X_te = scaler.transform(X_test.reshape(-1, c)).reshape(n_test, t, c)

    return X_tr.astype(np.float32), X_te.astype(np.float32)


# ============================================================
# STATISTICAL FEATURES
# ============================================================

def safe_skew(x):
    v = stats.skew(x)
    return float(v) if np.isfinite(v) else 0.0


def safe_kurtosis(x):
    v = stats.kurtosis(x)
    return float(v) if np.isfinite(v) else 0.0


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    v = np.corrcoef(a, b)[0, 1]
    return float(v) if np.isfinite(v) else 0.0


def one_axis_stats(x):
    return [
        np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
        np.percentile(x, 5), np.percentile(x, 10),
        np.percentile(x, 25), np.percentile(x, 75),
        np.percentile(x, 90), np.percentile(x, 95),
        np.max(x) - np.min(x),
        np.var(x),
        safe_skew(x),
        safe_kurtosis(x),
        np.mean(np.abs(x - np.mean(x))),
        np.sqrt(np.mean(x**2)),
    ]


def extract_stat_features_one(seq):
    features = []

    # Per-channel global stats
    for c in range(seq.shape[1]):
        features.extend(one_axis_stats(seq[:, c]))

    # Segment-level means (temporal structure)
    for n_seg in [2, 3, 5, 10]:
        splits = np.array_split(np.arange(seq.shape[0]), n_seg)
        for c in range(seq.shape[1]):
            vals = [np.mean(seq[idx, c]) for idx in splits]
            features.extend(vals)
            features.append(vals[-1] - vals[0])   # trend
            features.append(max(vals) - min(vals)) # range across segments

    # Cross-channel correlations (first 10 channels)
    max_c = min(seq.shape[1], 10)
    for i in range(max_c):
        for j in range(i + 1, max_c):
            features.append(safe_corr(seq[:, i], seq[:, j]))

    arr = np.array(features, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def extract_stat_features(X):
    return np.vstack([extract_stat_features_one(x) for x in X]).astype(np.float32)


# ============================================================
# ROCKET
# ============================================================

def generate_kernels(n_kernels, n_channels, seed):
    rng = np.random.default_rng(seed)
    kernels = []

    for _ in range(n_kernels):
        length   = int(rng.choice([7, 9, 11, 13]))
        dilation = int(rng.choice([1, 2, 3, 4, 5, 7, 9, 12]))

        n_ch = int(rng.integers(1, min(n_channels, 5) + 1))
        channels = rng.choice(n_channels, size=n_ch, replace=False)

        weights = rng.normal(0, 1, size=(n_ch, length)).astype(np.float32)
        weights -= weights.mean(axis=1, keepdims=True)  # zero-mean (ROCKET style)

        bias = float(rng.uniform(-1.0, 1.0))
        kernels.append((channels, weights, dilation, bias))

    return kernels


def apply_one_kernel(X, channels, weights, dilation, bias):
    n = X.shape[0]
    conv_sum = np.zeros((n, X.shape[1]), dtype=np.float32)

    for local_i, ch in enumerate(channels):
        w = weights[local_i]
        if dilation > 1:
            dilated = np.zeros((len(w) - 1) * dilation + 1, dtype=np.float32)
            dilated[::dilation] = w
            w = dilated

        conv = convolve1d(
            X[:, :, ch], weights=w[::-1], axis=1, mode="constant", cval=0.0
        )
        conv_sum += conv.astype(np.float32)

    conv_sum += bias
    max_val  = np.max(conv_sum, axis=1)
    ppv      = np.mean(conv_sum > 0, axis=1)
    mean_pos = np.mean(np.maximum(conv_sum, 0), axis=1)
    return max_val, ppv, mean_pos


def rocket_transform(X, kernels):
    n = X.shape[0]
    out = np.zeros((n, len(kernels) * 3), dtype=np.float32)

    for i, (channels, weights, dilation, bias) in enumerate(kernels):
        mv, ppv, mp = apply_one_kernel(X, channels, weights, dilation, bias)
        out[:, 3*i]   = mv
        out[:, 3*i+1] = ppv
        out[:, 3*i+2] = mp

        if (i + 1) % 1000 == 0:
            print(f"  ROCKET: {i+1}/{len(kernels)} kernels done")

    out[~np.isfinite(out)] = 0.0
    return out


# ============================================================
# MODELS
# ============================================================

def build_models():
    """
    Returns three classifiers that complement each other.
    All use class_weight='balanced' to handle the heavy imbalance
    (classes 0+1 = 85% of data, class 4 = only 1.3%).
    """
    ridge = make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 13),
            class_weight="balanced",
        ),
    )

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.5,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            n_jobs=-1,
        ),
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return ridge, logreg, et


def hard_vote(list_of_preds):
    mat = np.vstack(list_of_preds).T  # (n_samples, n_models)
    return np.array(
        [np.argmax(np.bincount(row, minlength=NUM_CLASSES)) for row in mat]
    )


# ============================================================
# VALIDATION
# ============================================================

def run_validation(X_features, y, groups):
    """
    StratifiedGroupKFold: folds are split by user so the model
    is never evaluated on users it was trained on.
    """
    print(f"\nRunning {N_SPLITS}-fold group-stratified validation...")
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = {"Ridge": [], "LogReg": [], "ExtraTrees": [], "Vote": []}

    for fold, (tr, va) in enumerate(splitter.split(X_features, y, groups), start=1):
        X_tr, X_va = X_features[tr], X_features[va]
        y_tr, y_va = y[tr], y[va]

        ridge, logreg, et = build_models()

        ridge.fit(X_tr, y_tr)
        logreg.fit(X_tr, y_tr)
        et.fit(X_tr, y_tr)

        p_ridge  = ridge.predict(X_va)
        p_logreg = logreg.predict(X_va)
        p_et     = et.predict(X_va)
        p_vote   = hard_vote([p_ridge, p_logreg, p_et])

        scores["Ridge"].append(f1_score(y_va, p_ridge,  average="macro", zero_division=0))
        scores["LogReg"].append(f1_score(y_va, p_logreg, average="macro", zero_division=0))
        scores["ExtraTrees"].append(f1_score(y_va, p_et,  average="macro", zero_division=0))
        scores["Vote"].append(f1_score(y_va, p_vote, average="macro", zero_division=0))

        print(f"  Fold {fold}: Ridge={scores['Ridge'][-1]:.4f}  "
              f"LogReg={scores['LogReg'][-1]:.4f}  "
              f"ET={scores['ExtraTrees'][-1]:.4f}  "
              f"Vote={scores['Vote'][-1]:.4f}")

    print("\nValidation summary (macro F1):")
    for name, vals in scores.items():
        print(f"  {name:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# ============================================================
# SUBMISSION
# ============================================================

def save_submission(test_ids, labels, filename):
    sample = pd.read_csv(SAMPLE_SUBMISSION)
    sample_ids = sample["Id"].astype(int).to_numpy()
    test_ids   = np.array(test_ids, dtype=int)

    if set(sample_ids) != set(test_ids):
        raise ValueError("ID mismatch between sample_submission.csv and test files")

    pred_map = dict(zip(test_ids, labels))
    out = sample[["Id"]].copy()
    out["Label"] = [int(pred_map[i]) for i in sample_ids]

    assert out["Label"].between(0, 5).all(), "Invalid label detected"
    out.to_csv(filename, index=False)
    print(f"  Saved → {filename}")


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(RANDOM_STATE)

    # ── Load ──────────────────────────────────────────────────
    print("Loading data...")
    train_dir = find_dir(TRAIN_DIRS)
    test_dir  = find_dir(TEST_DIRS)

    X_train_raw, y_train, groups = load_train(train_dir)
    X_test_raw,  test_ids        = load_test(test_dir)

    print(f"Train: {X_train_raw.shape}  Test: {X_test_raw.shape}")
    print("Class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

    # ── Feature engineering ───────────────────────────────────
    print("\nAdding derived channels...")
    X_train_seq = add_channels(X_train_raw)
    X_test_seq  = add_channels(X_test_raw)

    print("Normalizing channels...")
    X_train_seq, X_test_seq = normalize_channels(X_train_seq, X_test_seq)

    print("Extracting statistical features...")
    X_train_stat = extract_stat_features(X_train_seq)
    X_test_stat  = extract_stat_features(X_test_seq)

    # ── ROCKET ────────────────────────────────────────────────
    print(f"\nGenerating {N_KERNELS} ROCKET kernels...")
    kernels = generate_kernels(N_KERNELS, X_train_seq.shape[2], seed=RANDOM_STATE)

    print("Transforming train with ROCKET...")
    X_train_rocket = rocket_transform(X_train_seq, kernels)

    print("Transforming test with ROCKET...")
    X_test_rocket = rocket_transform(X_test_seq, kernels)

    # ── Combine ───────────────────────────────────────────────
    print("\nCombining ROCKET + statistical features...")
    X_train = np.hstack([X_train_rocket, X_train_stat]).astype(np.float32)
    X_test  = np.hstack([X_test_rocket,  X_test_stat]).astype(np.float32)
    X_train[~np.isfinite(X_train)] = 0.0
    X_test[~np.isfinite(X_test)]   = 0.0

    print(f"Final feature shape — train: {X_train.shape}  test: {X_test.shape}")

    # ── Validation ────────────────────────────────────────────
    if RUN_VALIDATION:
        run_validation(X_train, y_train, groups)

    # ── Final training ────────────────────────────────────────
    print("\nTraining final models on full training set...")
    ridge, logreg, et = build_models()

    print("  Training Ridge...")
    ridge.fit(X_train, y_train)

    print("  Training Logistic Regression...")
    logreg.fit(X_train, y_train)

    print("  Training ExtraTrees...")
    et.fit(X_train, y_train)

    # ── Predict & save ────────────────────────────────────────
    print("\nPredicting test set...")
    p_ridge  = ridge.predict(X_test)
    p_logreg = logreg.predict(X_test)
    p_et     = et.predict(X_test)
    p_vote   = hard_vote([p_ridge, p_logreg, p_et])

    os.makedirs("submissions", exist_ok=True)
    print("\nSaving submissions...")
    save_submission(test_ids, p_ridge,  "submissions/submission_ridge.csv")
    save_submission(test_ids, p_logreg, "submissions/submission_logreg.csv")
    save_submission(test_ids, p_et,     "submissions/submission_extratrees.csv")
    save_submission(test_ids, p_vote,   "submissions/submission_vote.csv")

    print("\nDone.")
    print("Recommended upload order (start with vote, then best single):")
    print("  1. submissions/submission_vote.csv")
    print("  2. submissions/submission_logreg.csv")
    print("  3. submissions/submission_extratrees.csv")


if __name__ == "__main__":
    main()