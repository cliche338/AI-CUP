import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import defaultdict
from scipy.fft import rfft
from tqdm import tqdm

# === 設定 ===
TRAIN_DIR = "Training/train_data"
TEST_DIR = "Test/test_data"
TRAIN_INFO = "Training/train_info.csv"
TEST_INFO = "Test/test_info.csv"
MODEL_DIR = "trained_models"
SEQ_LEN = 300
FEATURES = ["x", "y", "z", "x2", "y2"]
LABELS = ["gender", "hold racket handed", "play years", "level"]
LABEL_OUTPUT_COLUMNS = {
    "gender": ["gender"],
    "hold racket handed": ["hold racket handed"],
    "play years": [f"play years_{i}" for i in range(3)],
    "level": [f"level_{i}" for i in range(2, 6)],
}

# === 工具函數 ===
def parse_cutpoint(cp_str):
    return np.array([int(i) for i in cp_str.strip("[]").split()])

def pad_or_truncate(seq, length):
    if len(seq) >= length:
        return seq[:length]
    else:
        return np.pad(seq, ((0, length - len(seq)), (0, 0)), mode='constant')

def extract_advanced_features(segment, axis_name):
    data = segment.astype(float)
    feat = {
        f"{axis_name}_mean": np.mean(data),
        f"{axis_name}_std": np.std(data),
        f"{axis_name}_min": np.min(data),
        f"{axis_name}_max": np.max(data),
        f"{axis_name}_range": np.max(data) - np.min(data),
        f"{axis_name}_median": np.median(data),
        f"{axis_name}_iqr": np.percentile(data, 75) - np.percentile(data, 25),
        f"{axis_name}_skew": skew(data),
        f"{axis_name}_kurtosis": kurtosis(data),
        f"{axis_name}_energy": np.sum(data ** 2),
        f"{axis_name}_zcr": np.mean(np.abs(np.diff(np.sign(data)))),
        f"{axis_name}_var": np.var(data),
        f"{axis_name}_mad": np.mean(np.abs(data - np.mean(data))),
        f"{axis_name}_quantile_25": np.percentile(data, 25),
        f"{axis_name}_quantile_75": np.percentile(data, 75),
        f"{axis_name}_quantile_90": np.percentile(data, 90),
        f"{axis_name}_quantile_10": np.percentile(data, 10),
        f"{axis_name}_rms": np.sqrt(np.mean(data ** 2)),
        f"{axis_name}_peak_to_peak": np.max(data) - np.min(data),
    }

    fft_vals = np.abs(rfft(data))
    fft_sorted = np.sort(fft_vals)[::-1]
    for i in range(min(6, len(fft_sorted))):
        feat[f"{axis_name}_fft_amp_{i+1}"] = fft_sorted[i]

    freq = np.fft.rfftfreq(len(data))
    feat[f"{axis_name}_dominant_freq"] = freq[np.argmax(fft_vals)]
    feat[f"{axis_name}_spectral_entropy"] = -np.sum((fft_vals/np.sum(fft_vals)) * np.log2(fft_vals/np.sum(fft_vals) + 1e-6))
    feat[f"{axis_name}_spectral_centroid"] = np.sum(freq * fft_vals) / (np.sum(fft_vals) + 1e-6)
    feat[f"{axis_name}_spectral_bandwidth"] = np.sqrt(np.sum((freq - feat[f"{axis_name}_spectral_centroid"])**2 * fft_vals) / (np.sum(fft_vals) + 1e-6))

    positive_energy = np.sum(data[data > 0] ** 2)
    negative_energy = np.sum(data[data < 0] ** 2)
    total_energy = positive_energy + negative_energy + 1e-6
    feat[f"{axis_name}_pos_ratio"] = positive_energy / total_energy
    feat[f"{axis_name}_neg_ratio"] = negative_energy / total_energy

    feat[f"{axis_name}_zero_crossings"] = np.sum(np.diff(np.signbit(data)))
    feat[f"{axis_name}_mean_abs_diff"] = np.mean(np.abs(np.diff(data)))
    feat[f"{axis_name}_mean_diff"] = np.mean(np.diff(data))
    feat[f"{axis_name}_std_diff"] = np.std(np.diff(data))

    return feat

def extract_combined_features(segment):
    features = {}
    for idx, axis in enumerate(FEATURES):
        features.update(extract_advanced_features(segment[:, idx], axis))

    overall_magnitude = np.sqrt(np.sum(segment[:, :3]**2, axis=1))
    features["magnitude_mean"] = np.mean(overall_magnitude)
    features["magnitude_max"] = np.max(overall_magnitude)
    features["magnitude_std"] = np.std(overall_magnitude)
    features["magnitude_zcr"] = np.mean(np.abs(np.diff(np.sign(overall_magnitude))))

    features["magnitude_var"] = np.var(overall_magnitude)
    features["magnitude_skew"] = skew(overall_magnitude)
    features["magnitude_kurtosis"] = kurtosis(overall_magnitude)
    features["magnitude_energy"] = np.sum(overall_magnitude ** 2)
    features["magnitude_rms"] = np.sqrt(np.mean(overall_magnitude ** 2))
    features["magnitude_peak_to_peak"] = np.max(overall_magnitude) - np.min(overall_magnitude)

    for i in range(len(FEATURES)):
        for j in range(i+1, len(FEATURES)):
            features[f"corr_{FEATURES[i]}_{FEATURES[j]}"] = np.corrcoef(segment[:, i], segment[:, j])[0,1]
            features[f"cov_{FEATURES[i]}_{FEATURES[j]}"] = np.cov(segment[:, i], segment[:, j])[0,1]

    for axis in FEATURES:
        mean = features[f"{axis}_mean"]
        std = features[f"{axis}_std"]
        features[f"{axis}_cv"] = std / (mean + 1e-6)
        features[f"{axis}_cv_squared"] = (std / (mean + 1e-6)) ** 2

    return features

# === 載入資料 ===
def load_segments(info_path, data_dir, is_train=True):
    info = pd.read_csv(info_path)
    info["cut_point"] = info["cut_point"].apply(parse_cutpoint)
    features, segment_ids, label_dict = [], [], defaultdict(list)



    for _, row in tqdm(info.iterrows()):
        file = f"{row['unique_id']}.txt"
        path = os.path.join(data_dir, file)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep=" ", names=["time", *FEATURES]).iloc[1:].reset_index(drop=True)
        cps = row["cut_point"]

        for i in range(len(cps) - 1):
            start, end = cps[i], cps[i + 1]
            segment_df = df.iloc[start:end][FEATURES]
            if len(segment_df) < 10:
                continue
            segment = pad_or_truncate(segment_df.values, SEQ_LEN)

            feat = extract_combined_features(segment)
            feat["duration"] = end - start
            feat["gravity_x"] = np.sum(np.abs(segment[:, 0]) * np.arange(len(segment))) / (np.sum(np.abs(segment[:, 0])) + 1e-6)

            features.append(feat)
            segment_ids.append(int(row["unique_id"]))

            if is_train:
                for label in LABELS:
                    label_dict[label].append(row[label])

    if is_train:
        return pd.DataFrame(features), label_dict, np.array(segment_ids)
    else:
        return pd.DataFrame(features), np.array(segment_ids), info["unique_id"].unique()

# === 訓練模型 ===
def train_and_save_models(X_train, label_dict):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    for label in tqdm(LABELS):
        print(f"\n🔧 重新訓練模型: {label}")
        y = np.array(label_dict[label])
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        class_dist = pd.Series(y_encoded).value_counts()
        print(f"類別分布: {class_dist.to_dict()}")

        class_weights = {i: len(y_encoded) / (len(class_dist) * count)
                        for i, count in class_dist.items()}
        print(f"類別權重: {class_weights}")

        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

        num_class = len(np.unique(y_encoded))
        if num_class == 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        model = XGBClassifier(
            n_estimators=1200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=2,
            gamma=0.2,
            objective=objective,
            eval_metric=eval_metric,
            early_stopping_rounds=50,
            verbosity=0,
            use_label_encoder=False,
            tree_method='hist',
            device='cuda',
            predictor='gpu_predictor',
            scale_pos_weight=class_weights.get(1, 1) if num_class == 2 else None
        )

        print(f"使用 GPU 訓練模型...")
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=True)

        model_path = os.path.join(MODEL_DIR, f"xgb_{label.replace(' ', '_')}.joblib")
        joblib.dump(model, model_path)

        print(f"✅ 模型已儲存到 {model_path}")

# === 主流程 ===
X_train, label_dict, _ = load_segments(TRAIN_INFO, TRAIN_DIR, is_train=True)
train_and_save_models(X_train, label_dict)

print("\n🔎 預測測試資料...")
X_test, segment_ids_test, all_test_ids = load_segments(TEST_INFO, TEST_DIR, is_train=False)

all_preds = {}
for label in LABELS:
    print(f"🔎 預測: {label}")
    model_path = os.path.join(MODEL_DIR, f"xgb_{label.replace(' ', '_')}.joblib")
    model = joblib.load(model_path)
    preds = model.predict_proba(X_test)
    all_preds[label] = preds

submission = pd.DataFrame({"unique_id": segment_ids_test})
for label, preds in all_preds.items():
    columns = LABEL_OUTPUT_COLUMNS[label]
    for i, col in enumerate(columns):
        if preds.shape[1] > 1:
            submission[col] = preds[:, i]
        else:
            submission[col] = preds[:, 0]

submission = submission.groupby("unique_id").mean().reset_index()

submission = pd.merge(
    pd.DataFrame({"unique_id": all_test_ids}),
    submission,
    on="unique_id",
    how="left"
)
submission = submission.fillna(0)

submission = submission.round(6)
submission.to_csv("參數_9_output.csv", index=False, float_format="%.6f")
print("✅ 預測完成，已儲存到 參數_9_output.csv！")
