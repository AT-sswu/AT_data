"""
전통적 머신러닝용 상태 지표(Health Indicator) 추출
MATLAB Predictive Maintenance 방식 기반
- 시간 도메인 특징
- 주파수 도메인 특징 (FFT, 스펙트럼 분석)
- 통계적 모멘트
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# ==================== 설정 ====================
DATA_FOLDER = '/Users/seohyeon/PycharmProjects/AT_data/data_v1'
WINDOW_SIZE = 1000  # 윈도우 크기 (1초 @ 1000Hz)
OVERLAP = 500  # 50% 오버랩
SAMPLING_RATE = 100  # Hz
RANDOM_STATE = 42

print("=" * 70)
print("전통적 머신러닝용 상태 지표 추출")
print("MATLAB Predictive Maintenance 방식")
print("=" * 70)

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] 데이터 로드...")

all_data = []
all_labels = []
all_filenames = []

csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

# 5개 클래스 라벨 매핑
label_map = {
    'lidar': 0,
    'motor': 1,
    'driving': 2,
    'lidar_driving': 3,
    'motor_driving': 4
}

for filename in sorted(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)

    # 파일명에서 라벨 추출 (복합 라벨을 먼저 체크)
    if 'lidar_driving' in filename.lower():
        label = label_map['lidar_driving']
        label_name = 'lidar_driving'
    elif 'motor_driving' in filename.lower():
        label = label_map['motor_driving']
        label_name = 'motor_driving'
    elif 'lidar' in filename.lower():
        label = label_map['lidar']
        label_name = 'lidar'
    elif 'motor' in filename.lower():
        label = label_map['motor']
        label_name = 'motor'
    elif 'driving' in filename.lower():
        label = label_map['driving']
        label_name = 'driving'
    else:
        continue

    df = pd.read_csv(filepath)
    feature_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    if not all(col in df.columns for col in feature_columns):
        continue

    data = df[feature_columns].values
    print(f"  ✓ {filename}: {len(data):,}개 샘플 ({label_name})")

    all_data.append(data)
    all_labels.append(label)
    all_filenames.append(filename)

print(f"\n✓ 총 {len(all_data)}개 파일 로드")
print(f"✓ 클래스별 파일 수:")
for label_name, label_id in label_map.items():
    count = sum(1 for label in all_labels if label == label_id)
    print(f"  - {label_name.capitalize()}: {count}개")

# ==================== 2. 상태 지표 함수 정의 ====================
print("\n[2단계] 상태 지표 추출 함수 정의...")


def extract_time_domain_features(segment):
    """
    시간 도메인 상태 지표
    - 평균, RMS, 표준편차, 왜도, 첨도
    - Peak-to-Peak, Crest Factor, Shape Factor
    """
    features = {}

    for i, axis_name in enumerate(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']):
        axis_data = segment[:, i]

        # 기본 통계
        mean_val = np.mean(axis_data)
        std_val = np.std(axis_data)
        rms_val = np.sqrt(np.mean(axis_data ** 2))

        # 고차 통계
        skewness = stats.skew(axis_data)
        kurtosis_val = stats.kurtosis(axis_data)

        # Peak 분석
        peak_to_peak = np.ptp(axis_data)
        max_abs = np.max(np.abs(axis_data))

        # 형상 지표
        crest_factor = max_abs / rms_val if rms_val > 0 else 0
        shape_factor = rms_val / np.mean(np.abs(axis_data)) if np.mean(np.abs(axis_data)) > 0 else 0
        impulse_factor = max_abs / np.mean(np.abs(axis_data)) if np.mean(np.abs(axis_data)) > 0 else 0

        features[f'{axis_name}_mean'] = mean_val
        features[f'{axis_name}_std'] = std_val
        features[f'{axis_name}_rms'] = rms_val
        features[f'{axis_name}_skewness'] = skewness
        features[f'{axis_name}_kurtosis'] = kurtosis_val
        features[f'{axis_name}_peak_to_peak'] = peak_to_peak
        features[f'{axis_name}_crest_factor'] = crest_factor
        features[f'{axis_name}_shape_factor'] = shape_factor
        features[f'{axis_name}_impulse_factor'] = impulse_factor

    return features


def extract_frequency_domain_features(segment, sampling_rate):
    """
    주파수 도메인 상태 지표
    - 주요 주파수, 스펙트럼 에너지
    - 주파수 중심, 대역폭
    - 스펙트럼 엔트로피
    """
    features = {}

    for i, axis_name in enumerate(['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']):
        axis_data = segment[:, i]

        # FFT
        n = len(axis_data)
        fft_vals = fft(axis_data)
        fft_freqs = fftfreq(n, d=1 / sampling_rate)

        # 양의 주파수만
        positive_freqs = fft_freqs[:n // 2]
        positive_fft = np.abs(fft_vals[:n // 2])

        # DC 성분 제거
        positive_fft[0] = 0

        # 스펙트럼 에너지
        spectral_energy = np.sum(positive_fft ** 2)

        # 주요 주파수 (Peak Frequency)
        if len(positive_fft) > 0 and np.sum(positive_fft) > 0:
            peak_idx = np.argmax(positive_fft)
            peak_frequency = positive_freqs[peak_idx]
            peak_magnitude = positive_fft[peak_idx]

            # 주파수 중심 (Frequency Center)
            freq_center = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)

            # RMS 주파수
            rms_frequency = np.sqrt(np.sum((positive_freqs ** 2) * positive_fft) / np.sum(positive_fft))

            # 스펙트럼 엔트로피
            psd_normalized = positive_fft / np.sum(positive_fft)
            spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10))

            # 평균 주파수
            mean_frequency = np.mean(positive_fft)

            # 표준편차 주파수
            std_frequency = np.std(positive_fft)

        else:
            peak_frequency = 0
            peak_magnitude = 0
            freq_center = 0
            rms_frequency = 0
            spectral_entropy = 0
            mean_frequency = 0
            std_frequency = 0

        features[f'{axis_name}_peak_freq'] = peak_frequency
        features[f'{axis_name}_peak_magnitude'] = peak_magnitude
        features[f'{axis_name}_spectral_energy'] = spectral_energy
        features[f'{axis_name}_freq_center'] = freq_center
        features[f'{axis_name}_rms_freq'] = rms_frequency
        features[f'{axis_name}_spectral_entropy'] = spectral_entropy
        features[f'{axis_name}_mean_freq'] = mean_frequency
        features[f'{axis_name}_std_freq'] = std_frequency

    return features


def extract_envelope_spectrum_features(segment, sampling_rate):
    """
    포락선 스펙트럼 특징
    - Hilbert 변환을 통한 포락선 추출
    """
    features = {}

    for i, axis_name in enumerate(['Accel_X', 'Accel_Y', 'Accel_Z']):  # 가속도만
        axis_data = segment[:, i]

        # Hilbert 변환으로 포락선 추출
        analytic_signal = signal.hilbert(axis_data)
        envelope = np.abs(analytic_signal)

        # 포락선의 통계
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        envelope_max = np.max(envelope)

        features[f'{axis_name}_envelope_mean'] = envelope_mean
        features[f'{axis_name}_envelope_std'] = envelope_std
        features[f'{axis_name}_envelope_max'] = envelope_max

    return features


def extract_wavelet_features(segment):
    """
    웨이블릿 변환 특징
    - 다중 스케일 에너지
    """
    features = {}

    try:
        import pywt

        for i, axis_name in enumerate(['Accel_X', 'Accel_Y', 'Accel_Z']):
            axis_data = segment[:, i]

            # 웨이블릿 변환 (db4)
            coeffs = pywt.wavedec(axis_data, 'db4', level=3)

            # 각 레벨의 에너지
            for j, coeff in enumerate(coeffs):
                energy = np.sum(coeff ** 2)
                features[f'{axis_name}_wavelet_energy_level_{j}'] = energy

    except ImportError:
        # PyWavelets 없으면 스킵
        pass

    return features


def extract_all_health_indicators(segment, sampling_rate):
    """
    모든 상태 지표 추출
    """
    features = {}

    # 시간 도메인
    features.update(extract_time_domain_features(segment))

    # 주파수 도메인
    features.update(extract_frequency_domain_features(segment, sampling_rate))

    # 포락선 스펙트럼
    features.update(extract_envelope_spectrum_features(segment, sampling_rate))

    # 웨이블릿 (선택적)
    # features.update(extract_wavelet_features(segment))

    return features


print(f"✓ 상태 지표 함수 준비 완료")

# ==================== 3. 슬라이딩 윈도우로 상태 지표 추출 ====================
print("\n[3단계] 슬라이딩 윈도우 방식으로 상태 지표 추출...")
print(f"  - 윈도우 크기: {WINDOW_SIZE}개 샘플")
print(f"  - 오버랩: {OVERLAP}개 샘플 (50%)")
print("(이 과정은 시간이 걸릴 수 있습니다...)")

all_features = []
all_feature_labels = []

for file_idx, (data, label) in enumerate(zip(all_data, all_labels)):
    file_features = []

    # 슬라이딩 윈도우
    step = WINDOW_SIZE - OVERLAP
    for start_idx in range(0, len(data) - WINDOW_SIZE + 1, step):
        segment = data[start_idx:start_idx + WINDOW_SIZE]

        # 상태 지표 추출
        features = extract_all_health_indicators(segment, SAMPLING_RATE)
        file_features.append(features)
        all_feature_labels.append(label)

    all_features.extend(file_features)
    print(f"  ✓ 파일 {file_idx + 1}/{len(all_data)}: {len(file_features):,}개 윈도우")

# DataFrame으로 변환
features_df = pd.DataFrame(all_features)
labels_array = np.array(all_feature_labels)

print(f"\n✓ 상태 지표 추출 완료")
print(f"  - 총 샘플 수: {len(features_df):,}개")
print(f"  - 상태 지표 수: {len(features_df.columns)}개")

print(f"\n✓ 클래스별 샘플 수:")
for label_name, label_id in label_map.items():
    count = np.sum(labels_array == label_id)
    print(f"  - {label_name.capitalize()}: {count:,}개")

# ==================== 4. 특징 정리 및 결측치 처리 ====================
print("\n[4단계] 특징 정리 및 결측치 처리...")

# NaN이나 Inf 값 처리
features_df = features_df.replace([np.inf, -np.inf], np.nan)
features_df = features_df.fillna(0)

print(f"✓ 결측치 처리 완료")

# 특징 이름 출력
print(f"\n✓ 추출된 상태 지표 종류:")
print(f"  - 시간 도메인: {sum('mean' in col or 'std' in col or 'rms' in col for col in features_df.columns)}개")
print(f"  - 주파수 도메인: {sum('freq' in col or 'spectral' in col for col in features_df.columns)}개")
print(f"  - 포락선: {sum('envelope' in col for col in features_df.columns)}개")

# ==================== 5. 정규화 ====================
print("\n[5단계] 특징 정규화...")

X = features_df.values
y = labels_array

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ 정규화 완료 (평균 0, 표준편차 1)")

# ==================== 6. 데이터 분할 ====================
print("\n[6단계] 데이터 분할 (Train 70%, Val 15%, Test 15%)...")

# 분류용
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"✓ Train: {len(X_train):,}개")
print(f"✓ Validation: {len(X_val):,}개")
print(f"✓ Test: {len(X_test):,}개")

# ==================== 7. 데이터 저장 ====================
print("\n[7단계] 데이터 저장...")

# 역방향 라벨 맵 (숫자 -> 이름)
reverse_label_map = {v: k for k, v in label_map.items()}

data_dict = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'scaler': scaler,
    'feature_names': features_df.columns.tolist(),
    'feature_count': len(features_df.columns),
    'label_map': reverse_label_map
}

with open('processed_data_traditional_ml.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"✓ 저장: processed_data_traditional_ml.pkl")

# ==================== 8. 특징 중요도 분석 ====================
print("\n[8단계] 특징 분산 분석...")

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('results_traditional_ml', exist_ok=True)

# 특징별 분산
feature_variance = np.var(X_scaled, axis=0)
top_features_idx = np.argsort(feature_variance)[::-1][:20]
top_feature_names = [features_df.columns[i] for i in top_features_idx]

fig, ax = plt.subplots(figsize=(14, 6))
ax.barh(range(len(top_features_idx)), feature_variance[top_features_idx], color='steelblue')
ax.set_yticks(range(len(top_features_idx)))
ax.set_yticklabels(top_feature_names, fontsize=9)
ax.set_xlabel('분산', fontsize=12, fontweight='bold')
ax.set_title('상위 20개 상태 지표 (분산 기준)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results_traditional_ml/feature_variance.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_traditional_ml/feature_variance.png")
plt.close()

# ==================== 9. 요약 ====================
print("\n" + "=" * 70)
print("전통적 ML용 상태 지표 추출 완료!")
print("=" * 70)

print(f"\n 데이터셋 정보:")
print(f"  - 상태 지표 수: {len(features_df.columns)}개")
print(f"  - 총 샘플 수: {len(X_scaled):,}개")
print(f"  - 클래스 수: 5")
print(f"  - 클래스: {', '.join(label_map.keys())}")

print(f"\n 생성된 파일:")
print(f"  - processed_data_traditional_ml.pkl")
print(f"  - results_traditional_ml/feature_variance.png")
