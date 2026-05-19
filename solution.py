import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except:
    has_xgboost = False

if not os.path.isdir("train/train") or not os.path.isdir("test/test"):
    print("Error: train/train/ or test/test/ not found")
    exit()

def extract_features(df):
    x = df['mean_x'].values
    y = df['mean_y'].values
    z = df['mean_z'].values
    features = []

    for axis in [x, y, z]:
        features.extend([np.mean(axis), np.std(axis), np.min(axis), np.max(axis), np.median(axis),
                        np.percentile(axis, 75), np.percentile(axis, 25), np.max(axis) - np.min(axis),
                        np.var(axis)])

    mag = np.sqrt(x**2 + y**2 + z**2)
    features.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.median(mag),
                    np.sqrt(np.mean(x**2 + y**2 + z**2)), np.percentile(mag, 75),
                    np.var(mag)])

    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    dmag = np.sqrt(dx**2 + dy**2 + dz**2)
    features.extend([np.mean(dmag), np.std(dmag), np.max(dmag), np.mean(np.abs(dx)), np.mean(np.abs(dy)),
                    np.mean(np.abs(dz)), np.sum(dmag), np.percentile(dmag, 90), np.max(np.abs(dx)),
                    np.var(dx), np.var(dy), np.var(dz)])

    features.extend([np.sum(x**2), np.sum(y**2), np.sum(z**2), np.sum(mag**2), np.mean(x**2 + y**2 + z**2)])

    if len(x) > 1:
        first_half = len(x)//2
        features.extend([np.mean(x[:first_half]), np.mean(x[first_half:]), np.mean(y[:first_half]),
                        np.mean(y[first_half:]), np.mean(z[:first_half]), np.mean(z[first_half:]),
                        np.std(x[:first_half]), np.std(x[first_half:])])
    else:
        features.extend([0]*8)

    try:
        features.extend([np.corrcoef(x, y)[0,1] if len(x) > 1 else 0,
                        np.corrcoef(x, z)[0,1] if len(x) > 1 else 0,
                        np.corrcoef(y, z)[0,1] if len(x) > 1 else 0])
    except:
        features.extend([0, 0, 0])

    energy = x**2 + y**2 + z**2
    features.extend([np.mean(energy), np.std(energy), np.max(energy), np.min(energy), np.var(energy)])

    try:
        fft_x = np.abs(np.fft.fft(x))[:len(x)//2]
        fft_y = np.abs(np.fft.fft(y))[:len(y)//2]
        fft_z = np.abs(np.fft.fft(z))[:len(z)//2]
        features.extend([np.mean(fft_x), np.std(fft_x), np.max(fft_x), np.mean(fft_y),
                        np.std(fft_y), np.max(fft_y), np.mean(fft_z), np.std(fft_z), np.max(fft_z)])
    except:
        features.extend([0]*9)

    features = [f if np.isfinite(f) else 0 for f in features]
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
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Splitting data...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

print("Training models...")
rf = RandomForestClassifier(n_estimators=150, max_depth=25, min_samples_split=5, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
print(f"RF F1: {f1_score(y_val, rf.predict(X_val), average='macro', zero_division=0):.4f}")

gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=42)
gb.fit(X_tr, y_tr)
print(f"GB F1: {f1_score(y_val, gb.predict(X_val), average='macro', zero_division=0):.4f}")

svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm.fit(X_tr, y_tr)
print(f"SVM F1: {f1_score(y_val, svm.predict(X_val), average='macro', zero_division=0):.4f}")

if has_xgboost:
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_tr, y_tr)
    print(f"XGB F1: {f1_score(y_val, xgb.predict(X_val), average='macro', zero_division=0):.4f}")
    ensemble = 0.30*rf.predict_proba(X_test) + 0.25*gb.predict_proba(X_test) + 0.20*svm.predict_proba(X_test) + 0.25*xgb.predict_proba(X_test)
else:
    ensemble = 0.40*rf.predict_proba(X_test) + 0.35*gb.predict_proba(X_test) + 0.25*svm.predict_proba(X_test)

y_pred = np.argmax(ensemble, axis=1)

print("Saving submission...")
os.makedirs("submissions", exist_ok=True)
sub = pd.DataFrame({'Id': test_ids, 'Label': y_pred}).sort_values('Id').reset_index(drop=True)
sub.to_csv("submissions/submission.csv", index=False)
print(f"Submission saved: submissions/submission.csv ({len(sub)} predictions)")
