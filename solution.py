import os
import random
import warnings
import numpy as np
import pandas as pd

from scipy import stats
from scipy.ndimage import convolve1d

from sklearn.preprocessing import StandardScaler, RobustScaler
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

# More kernels = potentially stronger but slower.
# 5000 is a good compromise on a MacBook.
N_KERNELS = 5000

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
        raise ValueError(f"{path}: expected 300 rows, got {len(df)}")

    if df["file_id"].nunique() != 1:
        raise ValueError(f"{path}: multiple file_id values")


# ============================================================
# DATA LOADING
# ============================================================

def load_train(train_dir):
    X, y, groups, file_ids = [], [], [], []

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
                raise ValueError(f"{path}: file_id does not match filename")

            if df["label"].nunique() != 1:
                raise ValueError(f"{path}: multiple labels")

            label = int(df["label"].iloc[0])

            if label < 0 or label > 5:
                raise ValueError(f"{path}: invalid label {label}")

            X.append(df[RAW_COLS].values.astype(np.float32))
            y.append(label)
            groups.append(user)
            file_ids.append(file_id)

    return np.array(X), np.array(y), np.array(groups), np.array(file_ids)


def load_test(test_dir):
    X, ids, test_groups = [], [], []

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
                raise ValueError(f"{path}: file_id does not match filename")

            X.append(df[RAW_COLS].values.astype(np.float32))
            ids.append(file_id)
            test_groups.append(user)

    return np.array(X), np.array(ids), np.array(test_groups)


# ============================================================
# CHANNEL ENGINEERING
# ============================================================

def add_channels(X):
    """
    Input:
        X shape = (n_samples, 300, 6)

    Output:
        X_new shape = (n_samples, 300, channels)

    We keep the original channels and add motion-oriented channels.
    """

    mx = X[:, :, 0]
    my = X[:, :, 1]
    mz = X[:, :, 2]

    sx = X[:, :, 3]
    sy = X[:, :, 4]
    sz = X[:, :, 5]

    mag = np.sqrt(mx ** 2 + my ** 2 + mz ** 2)
    std_mag = np.sqrt(sx ** 2 + sy ** 2 + sz ** 2)

    dmx = np.diff(mx, axis=1, prepend=mx[:, :1])
    dmy = np.diff(my, axis=1, prepend=my[:, :1])
    dmz = np.diff(mz, axis=1, prepend=mz[:, :1])
    dmag = np.diff(mag, axis=1, prepend=mag[:, :1])

    ddmag = np.diff(dmag, axis=1, prepend=dmag[:, :1])

    energy = mx ** 2 + my ** 2 + mz ** 2

    ratio_xy = mx / (np.abs(my) + 1e-6)
    ratio_xz = mx / (np.abs(mz) + 1e-6)
    ratio_yz = my / (np.abs(mz) + 1e-6)

    extra = np.stack(
        [
            mag,
            std_mag,
            dmx,
            dmy,
            dmz,
            dmag,
            ddmag,
            energy,
            ratio_xy,
            ratio_xz,
            ratio_yz,
        ],
        axis=2,
    )

    X_new = np.concatenate([X, extra], axis=2).astype(np.float32)

    X_new[~np.isfinite(X_new)] = 0.0

    return X_new


def normalize_sequence_channels(X_train, X_test):
    n_train, t, c = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, c)
    X_test_2d = X_test.reshape(-1, c)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, t, c)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, t, c)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


# ============================================================
# CLASSICAL STATISTICAL FEATURES
# ============================================================

def safe_skew(x):
    v = stats.skew(x)
    return v if np.isfinite(v) else 0.0


def safe_kurtosis(x):
    v = stats.kurtosis(x)
    return v if np.isfinite(v) else 0.0


def safe_corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    v = np.corrcoef(a, b)[0, 1]
    return v if np.isfinite(v) else 0.0


def one_axis_stats(x):
    return [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        np.percentile(x, 5),
        np.percentile(x, 10),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.percentile(x, 90),
        np.percentile(x, 95),
        np.max(x) - np.min(x),
        np.var(x),
        safe_skew(x),
        safe_kurtosis(x),
        np.mean(np.abs(x - np.mean(x))),
        np.sqrt(np.mean(x ** 2)),
    ]


def extract_stat_features_one(seq):
    features = []

    # seq shape: (300, channels)
    for c in range(seq.shape[1]):
        x = seq[:, c]
        features.extend(one_axis_stats(x))

    # segment statistics
    for segments in [2, 3, 5, 10]:
        splits = np.array_split(np.arange(seq.shape[0]), segments)

        for c in range(seq.shape[1]):
            vals = []
            for idx in splits:
                vals.append(np.mean(seq[idx, c]))

            features.extend(vals)
            features.append(vals[-1] - vals[0])
            features.append(max(vals) - min(vals))

    # correlations for original and important derived channels
    max_c = min(seq.shape[1], 10)
    for i in range(max_c):
        for j in range(i + 1, max_c):
            features.append(safe_corr(seq[:, i], seq[:, j]))

    features = np.array(features, dtype=np.float32)
    features[~np.isfinite(features)] = 0.0

    return features


def extract_stat_features(X):
    return np.vstack([extract_stat_features_one(x) for x in X]).astype(np.float32)


# ============================================================
# ROCKET-LIKE RANDOM CONVOLUTION FEATURES
# ============================================================

def generate_kernels(n_kernels, n_channels, seed):
    rng = np.random.default_rng(seed)

    kernels = []

    possible_lengths = [7, 9, 11, 13]
    possible_dilations = [1, 2, 3, 4, 5, 7, 9, 12]

    for _ in range(n_kernels):
        length = int(rng.choice(possible_lengths))
        dilation = int(rng.choice(possible_dilations))

        max_channels = min(n_channels, 5)
        n_used_channels = int(rng.integers(1, max_channels + 1))
        channels = rng.choice(n_channels, size=n_used_channels, replace=False)

        weights = rng.normal(0, 1, size=(n_used_channels, length)).astype(np.float32)

        # Important ROCKET-style normalization:
        # each kernel has zero mean weights
        weights = weights - weights.mean(axis=1, keepdims=True)

        bias = float(rng.uniform(-1.0, 1.0))

        kernels.append((channels, weights, dilation, bias))

    return kernels


def apply_one_kernel(X, channels, weights, dilation, bias):
    """
    X shape = (n_samples, time, channels)
    Output features:
        max convolution response
        proportion of positive values
        mean positive response
    """

    n = X.shape[0]
    conv_sum = np.zeros((n, X.shape[1]), dtype=np.float32)

    for local_i, ch in enumerate(channels):
        w = weights[local_i]

        if dilation > 1:
            dilated = np.zeros((len(w) - 1) * dilation + 1, dtype=np.float32)
            dilated[::dilation] = w
            w_use = dilated
        else:
            w_use = w

        conv = convolve1d(
            X[:, :, ch],
            weights=w_use[::-1],
            axis=1,
            mode="constant",
            cval=0.0,
        )

        conv_sum += conv.astype(np.float32)

    conv_sum += bias

    max_val = np.max(conv_sum, axis=1)
    ppv = np.mean(conv_sum > 0, axis=1)
    mean_pos = np.mean(np.maximum(conv_sum, 0), axis=1)

    return max_val, ppv, mean_pos


def rocket_transform(X, kernels, batch_print=500):
    n = X.shape[0]
    out = np.zeros((n, len(kernels) * 3), dtype=np.float32)

    for i, (channels, weights, dilation, bias) in enumerate(kernels):
        max_val, ppv, mean_pos = apply_one_kernel(X, channels, weights, dilation, bias)

        out[:, 3 * i] = max_val
        out[:, 3 * i + 1] = ppv
        out[:, 3 * i + 2] = mean_pos

        if (i + 1) % batch_print == 0:
            print(f"ROCKET kernels processed: {i + 1}/{len(kernels)}")

    out[~np.isfinite(out)] = 0.0
    return out


# ============================================================
# VALIDATION
# ============================================================

def run_validation(X_features, y, groups):
    print("\nRunning group-based validation...")

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    ridge_scores = []
    logreg_scores = []
    et_scores = []

    for fold, (tr, va) in enumerate(splitter.split(X_features, y, groups), start=1):
        print(f"\nFold {fold}/{N_SPLITS}")

        X_tr, X_va = X_features[tr], X_features[va]
        y_tr, y_va = y[tr], y[va]

        ridge = make_pipeline(
            StandardScaler(),
            RidgeClassifierCV(
                alphas=np.logspace(-3, 3, 13),
                class_weight="balanced",
            ),
        )

        ridge.fit(X_tr, y_tr)
        pred_ridge = ridge.predict(X_va)
        f1_ridge = f1_score(y_va, pred_ridge, average="macro", zero_division=0)
        ridge_scores.append(f1_ridge)

        print(f"Ridge ROCKET F1: {f1_ridge:.5f}")

        logreg = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.5,
                penalty="l2",
                solver="lbfgs",
                max_iter=800,
                class_weight="balanced",
                multi_class="auto",
                n_jobs=-1,
            ),
        )

        logreg.fit(X_tr, y_tr)
        pred_logreg = logreg.predict(X_va)
        f1_logreg = f1_score(y_va, pred_logreg, average="macro", zero_division=0)
        logreg_scores.append(f1_logreg)

        print(f"LogReg ROCKET F1: {f1_logreg:.5f}")

        et = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE + fold,
            n_jobs=-1,
        )

        et.fit(X_tr, y_tr)
        pred_et = et.predict(X_va)
        f1_et = f1_score(y_va, pred_et, average="macro", zero_division=0)
        et_scores.append(f1_et)

        print(f"ExtraTrees hybrid F1: {f1_et:.5f}")

    print("\nValidation summary:")
    print(f"Ridge:     {np.mean(ridge_scores):.5f} ± {np.std(ridge_scores):.5f}")
    print(f"LogReg:    {np.mean(logreg_scores):.5f} ± {np.std(logreg_scores):.5f}")
    print(f"ExtraTrees:{np.mean(et_scores):.5f} ± {np.std(et_scores):.5f}")


# ============================================================
# FINAL TRAINING
# ============================================================

def train_final_models(X_features, y):
    print("\nTraining final models on full training data...")

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
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            multi_class="auto",
            n_jobs=-1,
        ),
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=350,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training Ridge...")
    ridge.fit(X_features, y)

    print("Training Logistic Regression...")
    logreg.fit(X_features, y)

    print("Training ExtraTrees...")
    et.fit(X_features, y)

    print("Training RandomForest...")
    rf.fit(X_features, y)

    return ridge, logreg, et, rf


# ============================================================
# SUBMISSION
# ============================================================

def save_submission(test_ids, labels, filename):
    sample = pd.read_csv(SAMPLE_SUBMISSION)

    if list(sample.columns) != ["Id", "Label"]:
        raise ValueError("sample_submission.csv must have columns Id, Label")

    sample_ids = sample["Id"].astype(int).to_numpy()
    test_ids = test_ids.astype(int)

    if set(sample_ids) != set(test_ids):
        missing = sorted(set(sample_ids) - set(test_ids))[:10]
        extra = sorted(set(test_ids) - set(sample_ids))[:10]
        raise ValueError(f"ID mismatch. Missing={missing}, Extra={extra}")

    pred_map = dict(zip(test_ids, labels))

    out = sample[["Id"]].copy()
    out["Label"] = [int(pred_map[i]) for i in sample_ids]

    assert out["Label"].between(0, 5).all()
    assert len(out) == len(sample)

    out.to_csv(filename, index=False)
    print(f"Saved {filename}")


def hard_vote(preds):
    preds = np.vstack(preds).T
    final = []

    for row in preds:
        counts = np.bincount(row, minlength=NUM_CLASSES)
        final.append(np.argmax(counts))

    return np.array(final)


# ============================================================
# MAIN
# ============================================================

def main():
    seed_everything(RANDOM_STATE)

    print("Loading data...")

    train_dir = find_dir(TRAIN_DIRS)
    test_dir = find_dir(TEST_DIRS)

    X_train_raw, y_train, groups, train_ids = load_train(train_dir)
    X_test_raw, test_ids, test_groups = load_test(test_dir)

    print("Raw train shape:", X_train_raw.shape)
    print("Raw test shape:", X_test_raw.shape)
    print("Class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

    print("\nAdding derived channels...")
    X_train_seq = add_channels(X_train_raw)
    X_test_seq = add_channels(X_test_raw)

    print("Sequence train shape:", X_train_seq.shape)
    print("Sequence test shape:", X_test_seq.shape)

    print("\nNormalizing sequence channels...")
    X_train_seq, X_test_seq = normalize_sequence_channels(X_train_seq, X_test_seq)

    print("\nExtracting statistical features...")
    X_train_stat = extract_stat_features(X_train_seq)
    X_test_stat = extract_stat_features(X_test_seq)

    print("Stat train shape:", X_train_stat.shape)
    print("Stat test shape:", X_test_stat.shape)

    print("\nGenerating ROCKET kernels...")
    kernels = generate_kernels(
        n_kernels=N_KERNELS,
        n_channels=X_train_seq.shape[2],
        seed=RANDOM_STATE,
    )

    print("\nTransforming training data with ROCKET...")
    X_train_rocket = rocket_transform(X_train_seq, kernels)

    print("\nTransforming test data with ROCKET...")
    X_test_rocket = rocket_transform(X_test_seq, kernels)

    print("ROCKET train shape:", X_train_rocket.shape)
    print("ROCKET test shape:", X_test_rocket.shape)

    print("\nCombining ROCKET + statistical features...")
    X_train_features = np.hstack([X_train_rocket, X_train_stat]).astype(np.float32)
    X_test_features = np.hstack([X_test_rocket, X_test_stat]).astype(np.float32)

    X_train_features[~np.isfinite(X_train_features)] = 0.0
    X_test_features[~np.isfinite(X_test_features)] = 0.0

    print("Final train feature shape:", X_train_features.shape)
    print("Final test feature shape:", X_test_features.shape)

    if RUN_VALIDATION:
        run_validation(X_train_features, y_train, groups)

    ridge, logreg, et, rf = train_final_models(X_train_features, y_train)

    print("\nPredicting test data...")

    pred_ridge = ridge.predict(X_test_features)
    pred_logreg = logreg.predict(X_test_features)
    pred_et = et.predict(X_test_features)
    pred_rf = rf.predict(X_test_features)

    pred_vote = hard_vote([pred_ridge, pred_logreg, pred_et, pred_rf])

    save_submission(test_ids, pred_ridge, "submission_rocket_ridge.csv")
    save_submission(test_ids, pred_logreg, "submission_rocket_logreg.csv")
    save_submission(test_ids, pred_et, "submission_rocket_extratrees.csv")
    save_submission(test_ids, pred_rf, "submission_rocket_rf.csv")
    save_submission(test_ids, pred_vote, "submission_rocket_vote.csv")

    print("\nDone.")
    print("Recommended upload order:")
    print("1. submission_rocket_logreg.csv")
    print("2. submission_rocket_vote.csv")
    print("3. submission_rocket_ridge.csv")
    print("4. submission_rocket_extratrees.csv")


if __name__ == "__main__":
    main()