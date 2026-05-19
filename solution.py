import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

print("NYCU Data Mining Assignment 3 - Solution")

if not os.path.isdir("train/train") or not os.path.isdir("test/test"):
    print("Error: train/train/ or test/test/ not found")
    exit()

def extract_features(df):
    x = df['mean_x'].values
    y = df['mean_y'].values
    z = df['mean_z'].values
    features = []

    for axis in [x, y, z]:
        features.append(np.mean(axis))
        features.append(np.std(axis))
        features.append(np.min(axis))
        features.append(np.max(axis))
        features.append(np.median(axis))
        features.append(np.percentile(axis, 75))

    mag = np.sqrt(x**2 + y**2 + z**2)
    features.extend([
        np.mean(mag), np.std(mag), np.min(mag), np.max(mag),
        np.median(mag), np.sqrt(np.mean(x**2 + y**2 + z**2)), np.percentile(mag, 75)
    ])

    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    dmag = np.sqrt(dx**2 + dy**2 + dz**2)
    features.extend([
        np.mean(dmag), np.std(dmag), np.max(dmag), np.mean(np.abs(dx)),
        np.mean(np.abs(dy)), np.mean(np.abs(dz)), np.sum(dmag),
        np.percentile(dmag, 90), np.max(np.abs(dx))
    ])

    features.extend([
        np.sum(x**2), np.sum(y**2), np.sum(z**2), np.sum(mag**2),
        np.mean(x**2 + y**2 + z**2)
    ])

    return np.array(features)

print("Loading training data...")
X_train, y_train = [], []
for user_dir in sorted([d for d in os.listdir("train/train") if os.path.isdir(f"train/train/{d}")]):
    user_path = f"train/train/{user_dir}"
    for csv_file in sorted([f for f in os.listdir(user_path) if f.endswith(".csv")]):
        try:
            df = pd.read_csv(f"{user_path}/{csv_file}")
            X_train.append(extract_features(df))
            y_train.append(int(df['label'].iloc[0]))
        except:
            pass

X_train, y_train = np.array(X_train), np.array(y_train)
print(f"Loaded {len(X_train)} training samples")

print("Loading test data...")
X_test, test_ids = [], []
for user_dir in sorted([d for d in os.listdir("test/test") if os.path.isdir(f"test/test/{d}")]):
    user_path = f"test/test/{user_dir}"
    for csv_file in sorted([f for f in os.listdir(user_path) if f.endswith(".csv")]):
        try:
            df = pd.read_csv(f"{user_path}/{csv_file}")
            X_test.append(extract_features(df))
            test_ids.append(int(csv_file.replace(".csv", "")))
        except:
            pass

X_test, test_ids = np.array(X_test), np.array(test_ids)
print(f"Loaded {len(X_test)} test samples")

print("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Splitting data...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("Training model...")
model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1, verbose=0)
model.fit(X_tr, y_tr)

y_val_pred = model.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
print(f"Validation F1-Score: {f1_val:.4f}")

print("Generating predictions...")
y_test_pred = model.predict(X_test)

print("Saving submission...")
os.makedirs("submissions", exist_ok=True)
submission_df = pd.DataFrame({'Id': test_ids, 'Label': y_test_pred}).sort_values('Id').reset_index(drop=True)
submission_df.to_csv("submissions/submission.csv", index=False)

print(f"Submission saved: submissions/submission.csv ({len(submission_df)} predictions)")
print(f"First 10 rows:\n{submission_df.head(10)}")