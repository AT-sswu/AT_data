import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 저역통과 필터 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# FFT 분석 함수
def fft_analysis(data, sample_rate=319, fft_size=None, plot=True, file_title="FFT Frequency Spectrum"):
    # DC 성분 제거
    data = data - np.mean(data)

    # FFT 사이즈 설정
    n = fft_size if fft_size else len(data)
    data = data[:n]

    # FFT 계산
    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)

    # 양의 주파수만 추출
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n

    # 공진 주파수
    resonance_freq = positive_freqs[np.argmax(positive_amps)]

    # 시각화
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(positive_freqs, positive_amps)
        plt.title(f"{file_title} - FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return positive_freqs, positive_amps, resonance_freq


def analyze_multiple_axes(file_path, axes, sample_rate=319, fft_size=1024, apply_filter=True, filter_order=20):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    for axis in axes:
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다.")
            continue

        data = df[axis].dropna().values

        if apply_filter:
            cutoff = sample_rate / 4  # Nyquist 기준으로 1/4 설정
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        print(f"\n 분석 축: {axis}")
        freqs, amps, resonance_freq = fft_analysis(
            data,
            sample_rate=sample_rate,
            fft_size=fft_size,
            plot=True,
            file_title=f"{file_title} - {axis}"
        )
        print(f"→ {axis} 축의 공진 주파수: {resonance_freq:.2f} Hz")

# 실행
analyze_multiple_axes(
    r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set1.csv",
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    sample_rate=319,
    fft_size=1024,
    apply_filter=True,
    filter_order=20
)
