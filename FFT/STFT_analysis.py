import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft

# -------------------------------
# 1. 저역통과 필터 정의
# -------------------------------

# 버터워스 저역통과 필터 계수 계산
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyq  # 정규화 컷오프
    return butter(order, normal_cutoff, btype='low', analog=False)

# 필터 적용 함수
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)  # 양방향 필터로 위상 지연 방지

# -------------------------------
# 2. 이상치 탐지를 위한 Threshold 계산
# -------------------------------

def calculate_threshold(values, method="std", n_std=3, recon_error_value=0.3):
    if method == "std":
        # 평균 ± n * 표준편차
        mean = np.mean(values)
        std = np.std(values)
        threshold = mean + n_std * std
    elif method == "percentile":
        # 상위 97.5% 이상을 이상치로 간주
        threshold = np.percentile(values, 97.5)
    elif method == "recon_error":
        # 재구성 오류 기반 고정값 임계점 사용
        threshold = recon_error_value
    else:
        raise ValueError("지원하지 않는 threshold 방법입니다.")
    return threshold

# -------------------------------
# 3. STFT 분석 및 시각화
# -------------------------------

def stft_analysis(
    data,
    sample_rate=296,
    window='hann',
    nperseg=128,
    threshold_method="percentile",
    n_std=3,
    recon_error_value=0.3,
    plot=True,
    file_title="STFT Spectrogram"
):
    # DC 성분 제거 (중심화)
    data = data - np.mean(data)

    # STFT 실행: 시간별 주파수 분석
    f, t, Zxx = stft(data, fs=sample_rate, window=window, nperseg=nperseg)

    # 진폭 스펙트럼 계산
    magnitude = np.abs(Zxx)

    # 이상치 탐지를 위한 임계값 계산
    threshold = calculate_threshold(magnitude.flatten(), threshold_method, n_std, recon_error_value)

    # 임계값 초과 영역을 이상치로 판단
    anomaly_mask = magnitude >= threshold

    # 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        plt.pcolormesh(t, f, magnitude, shading='gouraud', cmap='viridis')  # 진폭 시각화
        plt.colorbar(label='Amplitude')  # 컬러바 추가
        plt.contour(t, f, anomaly_mask, levels=[0.5], colors='r', linewidths=0.7)  # 이상치 윤곽선
        plt.title(f"{file_title} - STFT Spectrogram")
        plt.xlabel("Time [sec]")
        plt.ylabel("Frequency [Hz]")
        plt.axhline(y=sample_rate / 2, color='gray', linestyle='--', linewidth=0.5)  # 나이퀴스트 기준선
        plt.tight_layout()
        plt.show()

    return f, t, magnitude, threshold, anomaly_mask

# -------------------------------
# 4. 다축 STFT 분석 루프
# -------------------------------

def analyze_stft_multiple_axes(
    file_path,
    axes,
    sample_rate=296,
    apply_filter=True,
    filter_order=5,
    window='hann',
    nperseg=128,
    threshold_method="percentile",  # 'std', 'percentile', 'recon_error'
    n_std=3,
    recon_error_value=0.3
):
    # CSV 로드
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    # 선택된 센서 축에 대해 반복 분석
    for axis in axes:
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다.")
            continue

        data = df[axis].dropna().values

        # 필터 적용 여부 설정
        if apply_filter:
            cutoff = sample_rate / 4  # Cutoff는 Nyquist의 절반 이하
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        print(f"\n[분석 축: {axis}]")

        # STFT 분석 실행
        f, t, mag, threshold, mask = stft_analysis(
            data,
            sample_rate=sample_rate,
            window=window,
            nperseg=nperseg,
            threshold_method=threshold_method,
            n_std=n_std,
            recon_error_value=recon_error_value,
            plot=True,
            file_title=f"{file_title} - {axis}"
        )

        print(f"→ 사용된 Threshold: {threshold:.4f} (기준: {threshold_method})")
        print(f"→ 이상치 검출된 시간-주파수 지점 수: {np.sum(mask)}")

# -------------------------------
# 5. 실행
# -------------------------------

analyze_stft_multiple_axes(
    r"/data_v0/mpu6050_vibration_data_set5.csv",
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    sample_rate=296,
    apply_filter=True,
    filter_order=5,
    window='hann',
    nperseg=128,
    threshold_method="percentile",       # 또는 'percentile', 'recon_error'
    n_std=2.75,                   # std 방식에 사용될 표준편차 배수
    recon_error_value=0.3         # 고정 에러값 방식
)
