import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy import stats
import os
from pathlib import Path

# ==================== 설정 ====================
DATA_FOLDER = '/Users/seohyeon/PycharmProjects/AT_data/data_v1'
SAMPLING_RATE = 100  # Hz
feature_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# 라벨 정의
LABEL_MAP = {
    'lidar': 0,
    'motor': 1,
    'driving': 2,
    'lidar_driving': 3,
    'motor_driving': 4
}

print("=" * 80)
print("센서 데이터 탐색적 분석 (EDA)")
print("=" * 80)

# ==================== 데이터 로드 ====================
print("\n[1단계] CSV 파일 로드 및 기본 정보")
print("-" * 80)

csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
print(f"✓ 발견된 CSV 파일 수: {len(csv_files)}개\n")

all_data_by_class = {label: [] for label in LABEL_MAP.keys()}
file_info = []

for filename in sorted(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)
    filename_lower = filename.lower()

    # 라벨 추출
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

    if not all(col in df.columns for col in feature_columns):
        print(f"⚠ {filename}: 필수 컬럼 누락")
        continue

    data = df[feature_columns].values
    duration = len(data) / SAMPLING_RATE

    all_data_by_class[label].append(data)
    file_info.append({
        'filename': filename,
        'label': label,
        'samples': len(data),
        'duration': duration
    })

    print(f"  📄 {filename}")
    print(f"     클래스: {label}")
    print(f"     샘플 수: {len(data):,}개")
    print(f"     길이: {duration:.2f}초")
    print()

# ==================== 클래스별 데이터 통합 ====================
print("\n[2단계] 클래스별 데이터 통합")
print("-" * 80)

class_data = {}
for label, data_list in all_data_by_class.items():
    if data_list:
        class_data[label] = np.vstack(data_list)
        print(f"✓ {label:20s}: {class_data[label].shape[0]:,}개 샘플 "
              f"({class_data[label].shape[0] / SAMPLING_RATE:.2f}초)")

# ==================== 시간 도메인 통계 ====================
print("\n[3단계] 시간 도메인 통계 분석")
print("=" * 80)

axis_names = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

for label, data in class_data.items():
    print(f"\n📊 클래스: {label.upper()}")
    print("-" * 80)

    for axis_idx, axis_name in enumerate(axis_names):
        axis_data = data[:, axis_idx]

        print(f"\n  [{axis_name}]")
        print(f"    평균 (Mean):           {np.mean(axis_data):>12.6f}")
        print(f"    표준편차 (Std):         {np.std(axis_data):>12.6f}")
        print(f"    분산 (Variance):       {np.var(axis_data):>12.6f}")
        print(f"    최대값 (Max):          {np.max(axis_data):>12.6f}")
        print(f"    최소값 (Min):          {np.min(axis_data):>12.6f}")
        print(f"    범위 (Range):          {np.ptp(axis_data):>12.6f}")
        print(f"    중앙값 (Median):       {np.median(axis_data):>12.6f}")
        print(f"    1사분위수 (Q1):        {np.percentile(axis_data, 25):>12.6f}")
        print(f"    3사분위수 (Q3):        {np.percentile(axis_data, 75):>12.6f}")
        print(f"    IQR (Q3-Q1):          {np.percentile(axis_data, 75) - np.percentile(axis_data, 25):>12.6f}")
        print(f"    왜도 (Skewness):       {stats.skew(axis_data):>12.6f}")
        print(f"    첨도 (Kurtosis):       {stats.kurtosis(axis_data):>12.6f}")
        print(f"    RMS (제곱평균제곱근):   {np.sqrt(np.mean(axis_data ** 2)):>12.6f}")
        print(f"    MAD (평균절대편차):     {np.mean(np.abs(axis_data - np.mean(axis_data))):>12.6f}")

# ==================== 주파수 도메인 분석 ====================
print("\n\n[4단계] 주파수 도메인 분석")
print("=" * 80)

for label, data in class_data.items():
    print(f"\n📊 클래스: {label.upper()}")
    print("-" * 80)

    # 전체 데이터에서 대표 샘플 추출 (처음 1000개)
    sample_size = min(1000, len(data))
    sample_data = data[:sample_size]

    for axis_idx, axis_name in enumerate(axis_names):
        axis_data = sample_data[:, axis_idx]

        # FFT 계산
        n = len(axis_data)
        fft_values = fft(axis_data)
        fft_freqs = fftfreq(n, d=1 / SAMPLING_RATE)

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
        spectral_power = spectral_energy / len(positive_fft)
        spectral_entropy = -np.sum((positive_fft / np.sum(positive_fft)) *
                                   np.log(positive_fft / np.sum(positive_fft) + 1e-10))

        # 상위 5개 주파수 찾기
        top5_indices = np.argsort(positive_fft)[-5:][::-1]
        top5_freqs = positive_freqs[top5_indices]
        top5_mags = positive_fft[top5_indices]

        print(f"\n  [{axis_name}]")
        print(f"    주요 주파수 (Dominant Freq):     {dominant_freq:>10.4f} Hz")
        print(f"    주요 주파수 크기:                {dominant_magnitude:>10.4f}")
        print(f"    스펙트럼 에너지:                 {spectral_energy:>10.4f}")
        print(f"    스펙트럼 파워:                   {spectral_power:>10.4f}")
        print(f"    스펙트럼 엔트로피:               {spectral_entropy:>10.4f}")
        print(f"    평균 스펙트럼:                   {np.mean(positive_fft):>10.4f}")
        print(f"    스펙트럼 표준편차:               {np.std(positive_fft):>10.4f}")
        print(f"    스펙트럼 중앙값:                 {np.median(positive_fft):>10.4f}")
        print(f"    상위 5개 주파수:")
        for i, (freq, mag) in enumerate(zip(top5_freqs, top5_mags), 1):
            print(f"      {i}. {freq:>8.4f} Hz (크기: {mag:>10.4f})")

# ==================== 축 간 상관관계 분석 ====================
print("\n\n[5단계] 축 간 상관관계 분석")
print("=" * 80)

for label, data in class_data.items():
    print(f"\n📊 클래스: {label.upper()}")
    print("-" * 80)

    # 상관 행렬 계산
    corr_matrix = np.corrcoef(data.T)

    print("\n  상관 행렬:")
    print("  " + " " * 12 + "".join([f"{name:>12s}" for name in axis_names]))
    for i, name in enumerate(axis_names):
        print(f"  {name:>10s}  ", end="")
        for j in range(len(axis_names)):
            print(f"{corr_matrix[i, j]:>12.4f}", end="")
        print()

    print("\n  강한 상관관계 (|r| > 0.7):")
    strong_corr = []
    for i in range(len(axis_names)):
        for j in range(i + 1, len(axis_names)):
            if abs(corr_matrix[i, j]) > 0.7:
                strong_corr.append((axis_names[i], axis_names[j], corr_matrix[i, j]))

    if strong_corr:
        for axis1, axis2, corr in strong_corr:
            print(f"    {axis1} ↔ {axis2}: {corr:>8.4f}")
    else:
        print("    없음")

# ==================== 신호 크기(Magnitude) 분석 ====================
print("\n\n[6단계] 신호 크기(Magnitude) 분석")
print("=" * 80)

for label, data in class_data.items():
    print(f"\n📊 클래스: {label.upper()}")
    print("-" * 80)

    # 전체 신호 크기
    magnitude = np.sqrt(np.sum(data ** 2, axis=1))

    print(f"\n  전체 신호 크기:")
    print(f"    평균:              {np.mean(magnitude):>12.6f}")
    print(f"    표준편차:          {np.std(magnitude):>12.6f}")
    print(f"    최대값:            {np.max(magnitude):>12.6f}")
    print(f"    최소값:            {np.min(magnitude):>12.6f}")
    print(f"    중앙값:            {np.median(magnitude):>12.6f}")

    # 가속도/자이로 별도 분석
    accel_magnitude = np.sqrt(np.sum(data[:, :3] ** 2, axis=1))
    gyro_magnitude = np.sqrt(np.sum(data[:, 3:] ** 2, axis=1))

    print(f"\n  가속도계 크기:")
    print(f"    평균:              {np.mean(accel_magnitude):>12.6f}")
    print(f"    표준편차:          {np.std(accel_magnitude):>12.6f}")

    print(f"\n  자이로스코프 크기:")
    print(f"    평균:              {np.mean(gyro_magnitude):>12.6f}")
    print(f"    표준편차:          {np.std(gyro_magnitude):>12.6f}")

# ==================== 클래스 간 비교 요약 ====================
print("\n\n[7단계] 클래스 간 비교 요약")
print("=" * 80)

print("\n평균 신호 크기 비교:")
print("-" * 40)
for label, data in sorted(class_data.items()):
    magnitude = np.sqrt(np.sum(data ** 2, axis=1))
    print(f"  {label:20s}: {np.mean(magnitude):>12.6f}")

print("\n평균 주요 주파수 비교 (Accel_X 기준):")
print("-" * 40)
for label, data in sorted(class_data.items()):
    sample_size = min(1000, len(data))
    sample_data = data[:sample_size, 0]  # Accel_X

    n = len(sample_data)
    fft_values = fft(sample_data)
    fft_freqs = fftfreq(n, d=1 / SAMPLING_RATE)
    positive_freqs = fft_freqs[:n // 2]
    positive_fft = np.abs(fft_values[:n // 2])
    positive_fft[0] = 0

    dominant_idx = np.argmax(positive_fft)
    dominant_freq = positive_freqs[dominant_idx]

    print(f"  {label:20s}: {dominant_freq:>10.4f} Hz")

print("\n" + "=" * 80)
print("분석 완료!")
print("=" * 80)