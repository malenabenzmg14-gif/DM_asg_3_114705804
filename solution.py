import os
import random
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from scipy.ndimage import convolve1d

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42
SEQ_LEN = 300
NUM_CLASSES = 6

TRAIN_DIRS = ["train/train", "train"]
TEST_DIRS = ["test/test", "test"]

SAMPLE_SUBMISSION = "sample_submission.csv"

RAW_COLS = [
    "mean_x",
    "mean_y",
    "mean_z",
    "std_x",
    "std_y",
    "std_z",
]

N_KERNELS = 5000
RUN_VALIDATION = False
OUTPUT_FILE = "submission_final.csv"


# =========================================================
# UTILITIES
# =========================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


def find_dir(candidates):
    for d in candidates:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Could not find any of: {candidates}")


# =========================================================
# DATA VALIDATION
# =========================================================

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


# =========================================================
# LOAD TRAIN
# =========================================================

def load_train(train_dir):
    X = []
    y = []
    groups = []

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
                raise ValueError(f"{path}: multiple labels")

            label = int(df["label"].iloc[0])

            if label < 0 or label > 5:
                raise ValueError(f"{path}: invalid label {label}")

            X.append(df[RAW_COLS].values.astype(np.float32))
            y.append(label)
            groups.append(user)

    return np.array(X), np.array(y), np.array(groups)


# =========================================================
# LOAD TEST
# =========================================================

def load_test(test_dir):
    X = []
    ids = []

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


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def add_channels(X):
    mx = X[:, :, 0]
    my = X[:, :, 1]
    mz = X[:, :, 2]

    sx = X[:, :, 3]
    sy = X[:, :, 4]
    sz = X[:, :, 5]

    mag = np.sqrt(mx**2 + my**2 + mz**2)
    std_mag = np.sqrt(sx**2 + sy**2 + sz**2)

    dmx = np.diff(mx, axis=1, prepend=mx[:, :1])
    dmy = np.diff(my, axis=1, prepend=my[:, :1])
    dmz = np.diff(mz, axis=1, prepend=mz[:, :1])
    dmag = np.diff(mag, axis=1, prepend=mag[:, :1])

    energy = mx**2 + my**2 + mz**2

    extra = np.stack(
        [
            mag,
            std_mag,
            dmx,
            dmy,
            dmz,
            dmag,
            energy,
        ],
        axis=2,
    )

    out = np.concatenate([X, extra], axis=2).astype(np.float32)
    out[~np.isfinite(out)] = 0.0

    return out


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_channels(X_train, X_test):
    n_train, t, c = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train.reshape(-1, c)
    ).reshape(n_train, t, c)

    X_test_scaled = scaler.transform(
        X_test.reshape(-1, c)
    ).reshape(n_test, t, c)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


# =========================================================
# SAFE STATS
# =========================================================

def safe_skew(x):
    v = stats.skew(x)
    return float(v) if np.isfinite(v) else 0.0


def safe_kurtosis(x):
    v = stats.kurtosis(x)
    return float(v) if np.isfinite(v) else 0.0


# =========================================================
# STAT FEATURES
# =========================================================

def one_axis_stats(x):
    return [
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        np.percentile(x, 10),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.percentile(x, 90),
        np.var(x),
        safe_skew(x),
        safe_kurtosis(x),
        np.sqrt(np.mean(x**2)),
    ]


def extract_stat_features_one(seq):
    features = []

    for c in range(seq.shape[1]):
        features.extend(one_axis_stats(seq[:, c]))

    for n_seg in [2, 5]:
        splits = np.array_split(np.arange(seq.shape[0]), n_seg)

        for c in range(seq.shape[1]):
            vals = [np.mean(seq[idx, c]) for idx in splits]
            features.extend(vals)
            features.append(vals[-1] - vals[0])

    arr = np.array(features, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0

    return arr


def extract_stat_features(X):
    return np.vstack([extract_stat_features_one(x) for x in X]).astype(np.float32)


# =========================================================
# ROCKET
# =========================================================

def generate_kernels(n_kernels, n_channels, seed):
    rng = np.random.default_rng(seed)
    kernels = []

    for _ in range(n_kernels):
        length = int(rng.choice([7, 9, 11]))
        dilation = int(rng.choice([1, 2, 3, 5]))

        n_ch = int(rng.integers(1, min(n_channels, 4) + 1))
        channels = rng.choice(n_channels, size=n_ch, replace=False)

        weights = rng.normal(0, 1, size=(n_ch, length)).astype(np.float32)
        weights -= weights.mean(axis=1, keepdims=True)

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
            X[:, :, ch],
            weights=w[::-1],
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


def rocket_transform(X, kernels):
    n = X.shape[0]

    out = np.zeros((n, len(kernels) * 3), dtype=np.float32)

    for i, (channels, weights, dilation, bias) in enumerate(kernels):
        mv, ppv, mp = apply_one_kernel(
            X,
            channels,
            weights,
            dilation,
            bias,
        )

        out[:, 3 * i] = mv
        out[:, 3 * i + 1] = ppv
        out[:, 3 * i + 2] = mp

        if (i + 1) % 500 == 0:
            print(f"ROCKET kernels processed: {i + 1}/{len(kernels)}")

    out[~np.isfinite(out)] = 0.0

    return out


# =========================================================
# MODEL
# =========================================================

def build_model():
    return make_pipeline(
        StandardScaler(),
        RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 13),
            class_weight="balanced",
        ),
    )


def hard_vote(preds):
    preds = np.asarray(preds)
    out = []

    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        out.append(vals[np.argmax(counts)])

    return np.array(out)


def build_models():
    ridge = build_model()

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return ridge, logreg, et


# =========================================================
# SAVE SUBMISSION
# =========================================================

def save_submission(test_ids, labels, filename):
    sample = pd.read_csv(SAMPLE_SUBMISSION)

    if list(sample.columns) != ["Id", "Label"]:
        raise ValueError("sample_submission.csv must have columns Id, Label")

    sample_ids = sample["Id"].astype(int).to_numpy()
    test_ids = test_ids.astype(int)

    if set(sample_ids) != set(test_ids):
        raise ValueError("ID mismatch between sample_submission and test files")

    pred_map = dict(zip(test_ids, labels))

    out = sample[["Id"]].copy()
    out["Label"] = [int(pred_map[i]) for i in sample_ids]

    if not out["Label"].between(0, 5).all():
        raise ValueError("Invalid label detected")

    out.to_csv(filename, index=False)
    print(f"\nSaved: {filename}")


# =========================================================
# MAIN
# =========================================================

def main():
    seed_everything(RANDOM_STATE)

    print("Loading data...")

    train_dir = find_dir(TRAIN_DIRS)
    test_dir = find_dir(TEST_DIRS)

    X_train_raw, y_train, _groups = load_train(train_dir)
    X_test_raw, test_ids = load_test(test_dir)

    print(f"Train: {X_train_raw.shape}")
    print(f"Test: {X_test_raw.shape}")

    print("\nAdding channels...")
    X_train_seq = add_channels(X_train_raw)
    X_test_seq = add_channels(X_test_raw)

    print("Normalizing channels...")
    X_train_seq, X_test_seq = normalize_channels(X_train_seq, X_test_seq)

    print("Extracting statistical features...")
    X_train_stat = extract_stat_features(X_train_seq)
    X_test_stat = extract_stat_features(X_test_seq)

    print(f"\nGenerating {N_KERNELS} ROCKET kernels...")
    kernels = generate_kernels(N_KERNELS, X_train_seq.shape[2], RANDOM_STATE)

    print("\nTransforming train with ROCKET...")
    X_train_rocket = rocket_transform(X_train_seq, kernels)

    print("\nTransforming test with ROCKET...")
    X_test_rocket = rocket_transform(X_test_seq, kernels)

    print("\nCombining features...")

    X_train = np.hstack([X_train_rocket, X_train_stat]).astype(np.float32)
    X_test = np.hstack([X_test_rocket, X_test_stat]).astype(np.float32)

    X_train[~np.isfinite(X_train)] = 0.0
    X_test[~np.isfinite(X_test)] = 0.0

    print(f"Final train shape: {X_train.shape}")
    print(f"Final test shape: {X_test.shape}")

    print("\nTraining Ridge model...")
    model = build_model()
    model.fit(X_train, y_train)

    print("\nPredicting test set...")
    pred = model.predict(X_test)

    save_submission(test_ids, pred, OUTPUT_FILE)

    print("\nUpload file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()