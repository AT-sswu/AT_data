import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 저역통과 필터 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# Threshold 계산 함수
def calculate_threshold(amps, method="std", n_std=2.75, recon_error_value=0.3):
    if method == "std":
        mean = np.mean(amps)
        std = np.std(amps)
        threshold = mean + n_std * std
    elif method == "percentile":
        threshold = np.percentile(amps, 97.5)
    elif method == "recon_error":
        threshold = recon_error_value
    else:
        raise ValueError("지원하지 않는 threshold 방법입니다.")
    return threshold

# FFT 분석 함수 (플롯은 이 함수 외부에서 처리)
def fft_analysis(
    data,
    sample_rate=296,
    fft_size=None,
    threshold_method="std",
    n_std=2.75,
    recon_error_value=0.3
):
    data = data - np.mean(data)
    n = fft_size if fft_size else len(data)
    data = data[:n]

    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n
    resonance_freq = positive_freqs[np.argmax(positive_amps)]

    threshold = calculate_threshold(positive_amps, threshold_method, n_std, recon_error_value)

    # Threshold 이상 구간 탐색
    threshold_ranges = []
    above_threshold = positive_amps >= threshold
    in_range = False
    for i in range(len(positive_freqs)):
        if above_threshold[i] and not in_range:
            range_start = positive_freqs[i]
            in_range = True
        elif not above_threshold[i] and in_range:
            range_end = positive_freqs[i - 1]
            threshold_ranges.append((range_start, range_end))
            in_range = False
    if in_range:
        threshold_ranges.append((range_start, positive_freqs[-1]))

    return positive_freqs, positive_amps, resonance_freq, threshold_ranges, threshold

# 여러 축 분석 및 subplot 시각화
def analyze_multiple_axes(
    file_path,
    axes,
    sample_rate=296,
    fft_size=512,
    apply_filter=True,
    filter_order=5,
    threshold_method="std",
    n_std=2.75,
    recon_error_value=0.3
):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    num_axes = len(axes)
    num_rows = 2
    num_cols = 3

    plt.figure(figsize=(18, 8))  # 전체 subplot 사이즈

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다.")
            continue

        data = df[axis].dropna().values

        if apply_filter:
            cutoff = sample_rate / 4
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        print(f"\n[분석 축: {axis}]")
        freqs, amps, resonance_freq, threshold_ranges, threshold = fft_analysis(
            data,
            sample_rate=sample_rate,
            fft_size=fft_size,
            threshold_method=threshold_method,
            n_std=n_std,
            recon_error_value=recon_error_value
        )

        print(f"→ 공진 주파수: {resonance_freq:.2f} Hz")
        print(f"→ 사용된 Threshold 값: {threshold:.4f} (기준: {threshold_method})")

        if threshold_ranges:
            print("→ Threshold 이상 구간:")
            for r in threshold_ranges:
                print(f"   - {r[0]:.2f} Hz ~ {r[1]:.2f} Hz")
        else:
            print("→ Threshold 이상 구간 없음.")

        # subplot 그리기
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.plot(freqs, amps, label="Amplitude")
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.axhline(y=threshold, color='g', linestyle=':', label=f'Threshold: {threshold:.3f}')
        plt.title(f"{axis}\nResonance: {resonance_freq:.2f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.suptitle(f"FFT Frequency Spectrum - {file_title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 제목 공간 확보
    plt.show()

# 실행
analyze_multiple_axes(
    r"/Users/seohyeon/PycharmProjects/AT_data/datasets/mpu6050_vibration_data_set5.csv",
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    sample_rate=287,
    fft_size=512,
    apply_filter=True,
    filter_order=5,
    threshold_method="std",  # "std", "percentile", "recon_error"
    n_std=2.75,
    recon_error_value=0.3
)
