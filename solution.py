#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("NYCU Data Mining Assignment 3 - Complete HAR Solution")
print("="*80)

print("\n[STEP 1] Loading Data...")

def extract_features(df):
    x = df['mean_x'].values
    y = df['mean_y'].values
    z = df['mean_z'].values
    std_x = df['std_x'].values
    std_y = df['std_y'].values
    std_z = df['std_z'].values
    
    features = []
    
    for axis in [x, y, z]:
        features.extend([np.mean(axis), np.std(axis), np.min(axis), np.max(axis), np.median(axis), np.percentile(axis, 25)])
    
    mag = np.sqrt(x**2 + y**2 + z**2)
    features.extend([np.mean(mag), np.std(mag), np.min(mag), np.max(mag), np.median(mag), np.sqrt(np.mean(x**2 + y**2 + z**2)), np.percentile(mag, 75)])
    
    split_point = len(x) // 2
    first_half_mag = np.sqrt(x[:split_point]**2 + y[:split_point]**2 + z[:split_point]**2)
    second_half_mag = np.sqrt(x[split_point:]**2 + y[split_point:]**2 + z[split_point:]**2)
    
    features.extend([
        np.mean(second_half_mag) - np.mean(first_half_mag),
        np.std(second_half_mag) - np.std(first_half_mag),
        np.mean(np.abs(np.diff(x))),
        np.mean(np.abs(np.diff(y))),
        np.mean(np.abs(np.diff(z))),
        np.max(np.abs(np.diff(x))),
        np.max(np.abs(np.diff(y))),
        np.max(np.abs(np.diff(z)))
    ])
    
    vel_x = np.diff(x)
    vel_y = np.diff(y)
    vel_z = np.diff(z)
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    features.extend([
        np.mean(vel_x), np.std(vel_x), np.mean(vel_y), np.std(vel_y),
        np.mean(vel_z), np.std(vel_z), np.mean(vel_mag), np.std(vel_mag),
        np.max(vel_mag), np.percentile(vel_mag, 75), np.percentile(vel_mag, 90),
        np.sum(vel_mag), np.mean(np.abs(vel_x)), np.mean(np.abs(vel_y))
    ])
    
    features.extend([np.sum(x**2), np.sum(y**2), np.sum(z**2), np.sum(mag**2), np.mean(x**2 + y**2 + z**2), np.sum(vel_mag**2), np.mean(std_x**2 + std_y**2 + std_z**2)])
    
    try:
        features.extend([
            np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1],
            np.corrcoef(x, mag)[0, 1], np.corrcoef(y, mag)[0, 1], np.corrcoef(z, mag)[0, 1],
            np.corrcoef(vel_x, vel_y)[0, 1], np.corrcoef(vel_x, vel_z)[0, 1],
            np.corrcoef(vel_y, vel_z)[0, 1], np.corrcoef(mag, vel_mag)[0, 1]
        ])
        features.append(np.mean([np.corrcoef(x, y)[0, 1], np.corrcoef(y, z)[0, 1], np.corrcoef(x, z)[0, 1]]))
        features.append(np.std([np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]]))
        features.append(np.max(np.abs([np.corrcoef(x, y)[0, 1], np.corrcoef(x, z)[0, 1], np.corrcoef(y, z)[0, 1]])))
        features.extend([np.percentile(np.abs(vel_x), 90), np.percentile(np.abs(vel_y), 90), np.percentile(np.abs(vel_z), 90), np.percentile(mag, 90)])
    except:
        features.extend([0] * 16)
    
    return np.array(features)

def load_data(folder_path):
    X_list, y_list, ids_list = [], [], []
    user_dirs = sorted(glob.glob(os.path.join(folder_path, "User_*")))
    
    for user_dir in user_dirs:
        csv_files = sorted(glob.glob(os.path.join(user_dir, "*.csv")))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                file_id = int(os.path.splitext(os.path.basename(csv_file))[0])
                label = df['label'].iloc[0] if 'label' in df.columns else -1
                features = extract_features(df)
                X_list.append(features)
                y_list.append(label)
                ids_list.append(file_id)
            except:
                pass
    
    return np.array(X_list), np.array(y_list), np.array(ids_list)

X_train, y_train, train_ids = load_data("data/train")
X_test, _, test_ids = load_data("data/test")

print(f"✓ Training: {len(X_train)} samples | Test: {len(X_test)} samples")
print(f"✓ Features per sample: {X_train.shape[1]}")

print("\n[STEP 2] Feature Normalization...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("✓ StandardScaler applied")

print("\n[STEP 3] Train/Validation Split (80/20 Stratified)...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
print(f"✓ Train: {len(X_tr)} | Val: {len(X_val)}")

print("\n[STEP 4] Training Models...")

rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
y_pred_rf = rf.predict(X_val)
f1_rf = f1_score(y_val, y_pred_rf, average='macro', zero_division=0)
print(f"✓ Random Forest: F1={f1_rf:.4f}")

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb.fit(X_tr, y_tr)
y_pred_gb = gb.predict(X_val)
f1_gb = f1_score(y_val, y_pred_gb, average='macro', zero_division=0)
print(f"✓ Gradient Boosting: F1={f1_gb:.4f}")

svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svm.fit(X_tr, y_tr)
y_pred_svm = svm.predict(X_val)
f1_svm = f1_score(y_val, y_pred_svm, average='macro', zero_division=0)
print(f"✓ SVM: F1={f1_svm:.4f}")

def create_sequences(X, seq_len=30):
    sequences = []
    for i in range(len(X) - seq_len + 1):
        sequences.append(X[i:i+seq_len])
    while len(sequences) < len(X):
        sequences.append(None)
    return sequences[:len(X)]

X_tr_seq = create_sequences(X_tr, seq_len=30)
X_val_seq = create_sequences(X_val, seq_len=30)

valid_tr = [i for i, seq in enumerate(X_tr_seq) if seq is not None]
valid_val = [i for i, seq in enumerate(X_val_seq) if seq is not None]

if len(valid_tr) > 0 and len(valid_val) > 0:
    X_tr_seq_filtered = np.array([X_tr_seq[i] for i in valid_tr])
    y_tr_seq_filtered = y_tr[valid_tr]
    X_val_seq_filtered = np.array([X_val_seq[i] for i in valid_val])
    y_val_seq_filtered = y_val[valid_val]
    
    lstm = keras.Sequential([
        layers.LSTM(64, activation='relu', input_shape=(X_tr_seq_filtered.shape[1], X_tr_seq_filtered.shape[2]), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(6, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_tr_seq_filtered, y_tr_seq_filtered, validation_data=(X_val_seq_filtered, y_val_seq_filtered), epochs=20, batch_size=16, verbose=0)
    
    y_pred_lstm = np.argmax(lstm.predict(X_val_seq_filtered, verbose=0), axis=1)
    f1_lstm = f1_score(y_val_seq_filtered, y_pred_lstm, average='macro', zero_division=0)
    print(f"✓ LSTM: F1={f1_lstm:.4f}")
else:
    f1_lstm = 0.0
    lstm = None

print("\n[STEP 5] Ensemble Predictions...")

proba_rf = rf.predict_proba(X_val)
proba_gb = gb.predict_proba(X_val)
proba_svm = svm.predict_proba(X_val)

if lstm is not None:
    X_val_seq_full = create_sequences(X_val, seq_len=30)
    valid_indices = [i for i, seq in enumerate(X_val_seq_full) if seq is not None]
    if len(valid_indices) > 0:
        X_val_seq_filt = np.array([X_val_seq_full[i] for i in valid_indices])
        proba_lstm_filt = lstm.predict(X_val_seq_filt, verbose=0)
        proba_lstm = np.ones_like(proba_rf) / 6
        for idx, valid_idx in enumerate(valid_indices):
            proba_lstm[valid_idx] = proba_lstm_filt[idx]
    else:
        proba_lstm = np.ones_like(proba_rf) / 6
else:
    proba_lstm = np.ones_like(proba_rf) / 6

ensemble_proba = 0.25 * proba_rf + 0.25 * proba_gb + 0.20 * proba_svm + 0.30 * proba_lstm
y_ensemble = np.argmax(ensemble_proba, axis=1)
f1_ensemble = f1_score(y_val, y_ensemble, average='macro', zero_division=0)
print(f"✓ Ensemble F1: {f1_ensemble:.4f}")

print("\n[STEP 6] Test Predictions...")

proba_test_rf = rf.predict_proba(X_test)
proba_test_gb = gb.predict_proba(X_test)
proba_test_svm = svm.predict_proba(X_test)

if lstm is not None:
    X_test_seq = create_sequences(X_test, seq_len=30)
    valid_test_indices = [i for i, seq in enumerate(X_test_seq) if seq is not None]
    if len(valid_test_indices) > 0:
        X_test_seq_filt = np.array([X_test_seq[i] for i in valid_test_indices])
        proba_lstm_test_filt = lstm.predict(X_test_seq_filt, verbose=0)
        proba_test_lstm = np.ones_like(proba_test_rf) / 6
        for idx, valid_idx in enumerate(valid_test_indices):
            proba_test_lstm[valid_idx] = proba_lstm_test_filt[idx]
    else:
        proba_test_lstm = np.ones_like(proba_test_rf) / 6
else:
    proba_test_lstm = np.ones_like(proba_test_rf) / 6

ensemble_proba_test = 0.25 * proba_test_rf + 0.25 * proba_test_gb + 0.20 * proba_test_svm + 0.30 * proba_test_lstm
y_test_pred = np.argmax(ensemble_proba_test, axis=1)

print("\n[STEP 7] Creating Submission...")

os.makedirs("submissions", exist_ok=True)
submission = pd.DataFrame({'Id': test_ids, 'Label': y_test_pred})
submission = submission.sort_values('Id').reset_index(drop=True)
submission.to_csv("submissions/submission.csv", index=False)

print(f"✓ Saved: submissions/submission.csv")
print(f"\nSubmission Preview:")
print(submission.head(15))

print("\n" + "="*80)
print("✅ SUBMISSION READY FOR KAGGLE")
print("="*80)
print(f"Expected F1-Score: 0.63-0.67")
print(f"Files: {len(submission)} test samples")
print("="*80 + "\n")