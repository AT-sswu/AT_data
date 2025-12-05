import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.fft import fft, fftfreq
from scipy import stats
import os
import pickle

# ==================== 설정 ====================
DATA_FOLDER = '/Users/seohyeon/PycharmProjects/AT_data/data_v1'
SEQUENCE_LENGTH = 100
STEP_SIZE = 10
RANDOM_STATE = 42
SAMPLING_RATE = 100

# 라벨 매핑 (5개 클래스)
LABEL_MAP = {
    'lidar': 0,
    'motor': 1,
    'driving': 2,
    'lidar_driving': 3,
    'motor_driving': 4
}

print("=" * 60)
print("DNN-MLP용 특징 추출 데이터 준비 (5개 클래스)")
print("=" * 60)
print("\n특징 추출 방식:")
print("  - 시간 도메인: 평균, 표준편차, 최대/최소, 왜도, 첨도")
print("  - 주파수 도메인: FFT 주요 주파수, 스펙트럼 에너지")
print("  - 축 간 상관관계")

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] CSV 파일 로드...")

all_data = []
all_labels = []

csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

for filename in sorted(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)

    # 파일명에서 라벨 추출
    filename_lower = filename.lower()

    if 'lidar_driving' in filename_lower:
        label = 'lidar_driving'
    elif 'motor_driving' in filename_lower:
        label = 'motor_driving'
    elif 'lidar' in filename_lower:
        label = 'lidar'
    elif 'motor' in filename_lower:
        label = 'motor'
    elif 'driving' in filename_lower:
        label = 'driving'
    else:
        continue

    df = pd.read_csv(filepath)
    feature_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    if not all(col in df.columns for col in feature_columns):
        continue

    data = df[feature_columns].values
    print(f"  ✓ {filename}: {len(data):,}개 샘플 ({label})")

    all_data.append(data)
    all_labels.extend([label] * len(data))

X_raw = np.vstack(all_data)
y_labels = np.array(all_labels)

print(f"\n✓ 전체 데이터: {X_raw.shape}")
print(f"✓ 클래스별 샘플 수:")
for label in LABEL_MAP.keys():
    count = np.sum(y_labels == label)
    print(f"  - {label}: {count:,}개")

# ==================== 2. 특징 추출 함수 ====================
print("\n[2단계] 특징 추출 함수 정의...")


def extract_time_domain_features(signal):
    """
    시간 도메인 통계적 특징
    """
    features = []

    # 각 축별 통계
    for axis in range(signal.shape[1]):
        axis_data = signal[:, axis]

        features.extend([
            np.mean(axis_data),  # 평균
            np.std(axis_data),  # 표준편차
            np.max(axis_data),  # 최대값
            np.min(axis_data),  # 최소값
            np.ptp(axis_data),  # 범위 (max - min)
            stats.skew(axis_data),  # 왜도 (비대칭도)
            stats.kurtosis(axis_data),  # 첨도 (뾰족함)
            np.median(axis_data),  # 중앙값
            np.percentile(axis_data, 25),  # 1사분위수
            np.percentile(axis_data, 75),  # 3사분위수
        ])

    return features


def extract_frequency_domain_features(signal, sampling_rate):
    """
    주파수 도메인 특징 (FFT)
    """
    features = []

    # 각 축별 FFT
    for axis in range(signal.shape[1]):
        axis_data = signal[:, axis]

        # FFT
        n = len(axis_data)
        fft_values = fft(axis_data)
        fft_freqs = fftfreq(n, d=1 / sampling_rate)

        # 양의 주파수만
        positive_freqs = fft_freqs[:n // 2]
        positive_fft = np.abs(fft_values[:n // 2])

        # DC 성분 제외
        positive_fft[0] = 0

        # 주요 주파수
        dominant_idx = np.argmax(positive_fft)
        dominant_freq = positive_freqs[dominant_idx]
        dominant_magnitude = positive_fft[dominant_idx]

        # 스펙트럼 통계
        spectral_energy = np.sum(positive_fft ** 2)
        spectral_entropy = -np.sum((positive_fft / np.sum(positive_fft)) *
                                   np.log(positive_fft / np.sum(positive_fft) + 1e-10))

        features.extend([
            dominant_freq,  # 주요 주파수
            dominant_magnitude,  # 주요 주파수 크기
            spectral_energy,  # 스펙트럼 에너지
            spectral_entropy,  # 스펙트럼 엔트로피
            np.mean(positive_fft),  # 평균 스펙트럼
            np.std(positive_fft),  # 스펙트럼 표준편차
        ])

    return features


def extract_correlation_features(signal):
    """
    축 간 상관관계 특징
    """
    features = []

    # 6축 데이터의 상관 행렬
    corr_matrix = np.corrcoef(signal.T)

    # 상삼각 행렬 (중복 제거)
    for i in range(6):
        for j in range(i + 1, 6):
            features.append(corr_matrix[i, j])

    return features


def extract_all_features(sequence, sampling_rate):
    """
    모든 특징 추출
    """
    features = []

    # 시간 도메인 (6축 × 10개 = 60개)
    features.extend(extract_time_domain_features(sequence))

    # 주파수 도메인 (6축 × 6개 = 36개)
    features.extend(extract_frequency_domain_features(sequence, sampling_rate))

    # 상관관계 (15개 = C(6,2))
    features.extend(extract_correlation_features(sequence))

    # 전체 신호 크기
    magnitude = np.sqrt(np.sum(sequence ** 2, axis=1))
    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.max(magnitude),
    ])

    return np.array(features)


print(f"✓ 특징 추출 함수 준비 완료")

# ==================== 3. 시퀀스 생성 및 특징 추출 ====================
print("\n[3단계] 시퀀스 생성 및 특징 추출 중...")
print("(이 과정은 시간이 걸릴 수 있습니다...)")


# 주파수 레이블도 함께 생성
def extract_dominant_frequency(signal, sampling_rate):
    magnitude = np.sqrt(np.sum(signal ** 2, axis=1))
    n = len(magnitude)
    fft_values = fft(magnitude)
    fft_freqs = fftfreq(n, d=1 / sampling_rate)
    positive_freqs = fft_freqs[:n // 2]
    positive_fft = np.abs(fft_values[:n // 2])
    positive_fft[0] = 0
    dominant_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[dominant_idx]
    if dominant_freq < 1.0:
        dominant_freq = np.random.uniform(0.5, 2.0)
    return dominant_freq


feature_vectors = []
class_labels = []
frequency_labels = []

for label in LABEL_MAP.keys():
    class_indices = np.where(y_labels == label)[0]
    class_data = X_raw[class_indices]

    for i in range(0, len(class_data) - SEQUENCE_LENGTH + 1, STEP_SIZE):
        seq = class_data[i:i + SEQUENCE_LENGTH]

        # 특징 추출
        features = extract_all_features(seq, SAMPLING_RATE)
        feature_vectors.append(features)

        # 분류 레이블
        class_labels.append(LABEL_MAP[label])

        # 회귀 레이블 (주파수)
        freq = extract_dominant_frequency(seq, SAMPLING_RATE)
        frequency_labels.append(freq)

    print(f"  ✓ {label}: {len(class_data):,}개 → 특징 벡터 생성")

X_features = np.array(feature_vectors)
y_class = np.array(class_labels)
y_freq = np.array(frequency_labels)

print(f"\n✓ 특징 벡터 생성 완료")
print(f"  - 샘플 수: {len(X_features):,}개")
print(f"  - 특징 수: {X_features.shape[1]}개")
print(f"    • 시간 도메인: 60개 (6축 × 10 통계)")
print(f"    • 주파수 도메인: 36개 (6축 × 6 FFT)")
print(f"    • 상관관계: 15개 (축 간 상관)")
print(f"    • 전체 통계: 3개")
print(f"  - 분류 레이블: 5 클래스")
print(f"  - 회귀 레이블: {y_freq.min():.2f} ~ {y_freq.max():.2f} Hz")

# ==================== 4. 정규화 ====================
print("\n[4단계] 특징 정규화...")

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_features)

print(f"✓ 정규화 완료 (평균 0, 표준편차 1)")

# ==================== 5. 데이터 분할 ====================
print("\n[5단계] 데이터 분할 (Train 70%, Val 15%, Test 15%)...")

# 분류용 데이터 분할
X_train_cls, X_temp_cls, y_train_cls, y_temp_cls = train_test_split(
    X_normalized, y_class,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_class
)

X_val_cls, X_test_cls, y_val_cls, y_test_cls = train_test_split(
    X_temp_cls, y_temp_cls,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp_cls
)

# One-hot 인코딩 (5개 클래스)
y_train_cls_onehot = np.eye(5)[y_train_cls]
y_val_cls_onehot = np.eye(5)[y_val_cls]
y_test_cls_onehot = np.eye(5)[y_test_cls]

# 회귀용 데이터 분할
X_train_reg, X_temp_reg, y_train_reg, y_temp_reg = train_test_split(
    X_normalized, y_freq,
    test_size=0.3,
    random_state=RANDOM_STATE
)

X_val_reg, X_test_reg, y_val_reg, y_test_reg = train_test_split(
    X_temp_reg, y_temp_reg,
    test_size=0.5,
    random_state=RANDOM_STATE
)

print(f"\n✓ 분류 데이터:")
print(f"  - Train: {X_train_cls.shape[0]:,}개")
print(f"  - Val: {X_val_cls.shape[0]:,}개")
print(f"  - Test: {X_test_cls.shape[0]:,}개")

print(f"\n✓ 회귀 데이터:")
print(f"  - Train: {X_train_reg.shape[0]:,}개")
print(f"  - Val: {X_val_reg.shape[0]:,}개")
print(f"  - Test: {X_test_reg.shape[0]:,}개")

# ==================== 6. 데이터 저장 ====================
print("\n[6단계] 데이터 저장...")

# 분류용
classification_data = {
    'X_train': X_train_cls,
    'X_val': X_val_cls,
    'X_test': X_test_cls,
    'y_train': y_train_cls_onehot,
    'y_val': y_val_cls_onehot,
    'y_test': y_test_cls_onehot,
    'scaler': scaler,
    'feature_count': X_features.shape[1],
    'label_map': LABEL_MAP,
    'num_classes': 5,
    'task': 'classification'
}

with open('processed_data_dnn_classification.pkl', 'wb') as f:
    pickle.dump(classification_data, f)

print(f"✓ 저장: processed_data_dnn_classification.pkl")

# 회귀용
regression_data = {
    'X_train': X_train_reg,
    'X_val': X_val_reg,
    'X_test': X_test_reg,
    'y_train': y_train_reg,
    'y_val': y_val_reg,
    'y_test': y_test_reg,
    'scaler': scaler,
    'feature_count': X_features.shape[1],
    'label_map': LABEL_MAP,
    'task': 'regression'
}

with open('processed_data_dnn_regression.pkl', 'wb') as f:
    pickle.dump(regression_data, f)

print(f"✓ 저장: processed_data_dnn_regression.pkl")

# ==================== 7. 특징 중요도 시각화 ====================
print("\n[7단계] 특징 분포 시각화...")

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results_dnn', exist_ok=True)

# 특징별 분산 확인 (중요도 지표)
feature_variance = np.var(X_normalized, axis=0)
feature_importance = np.argsort(feature_variance)[::-1][:20]  # 상위 20개

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(range(20), feature_variance[feature_importance], color='steelblue',
       alpha=0.7, edgecolor='black')
ax.set_xlabel('특징 인덱스 (분산 기준 상위 20개)', fontsize=12, fontweight='bold')
ax.set_ylabel('분산', fontsize=12, fontweight='bold')
ax.set_title('특징 중요도 (분산 기준)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results_dnn/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_dnn/feature_importance.png")
plt.close()

# ==================== 8. 요약 ====================
print("\n" + "=" * 60)
print("DNN-MLP용 데이터 준비 완료!")
print("=" * 60)

print(f"\n 데이터셋 정보:")
print(f"  - 특징 벡터 크기: {X_features.shape[1]}개")
print(f"  - 샘플 수: {len(X_features):,}개")
print(f"  - 분류: 5 클래스 (lidar, motor, driving, lidar_driving, motor_driving)")
print(f"  - 회귀: 연속적 주파수 예측")
print(f"\n클래스별 샘플 분포:")
for label_name, label_id in LABEL_MAP.items():
    count = np.sum(y_class == label_id)
    print(f"  - {label_name}: {count:,}개 ({count / len(y_class) * 100:.1f}%)")