import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 저역통과 필터 생성 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 주파수
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 필터 적용 함수
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# FFT 분석 함수
def fft_analysis(data, sample_rate=296, fft_size=None):
    data = data - np.mean(data)  # DC 성분 제거
    n = fft_size if fft_size else len(data)
    data = data[:n]
    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n
    resonance_freq = positive_freqs[np.argmax(positive_amps)]
    return positive_freqs, positive_amps, resonance_freq


def analyze_multiple_axes(file_path, axes, sample_rate=296, fft_size=512, apply_filter=True, filter_order=5):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    num_axes = len(axes)
    num_rows = 2
    num_cols = 3

    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다.")
            continue

        data = df[axis].dropna().values

        if apply_filter:
            cutoff = sample_rate / 4
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        freqs, amps, resonance_freq = fft_analysis(data, sample_rate=sample_rate, fft_size=fft_size)

        print(f"→ {axis} 축의 공진 주파수: {resonance_freq:.2f} Hz")

        # subplot 그리기
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.plot(freqs, amps)
        plt.title(f"{axis}\nResonance: {resonance_freq:.2f} Hz")  # 축 이름 + 공진 주파수 표시
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance')
        plt.legend()

    plt.suptitle(f"FFT Frequency Spectrum - {file_title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# 실행
analyze_multiple_axes(
    r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set1.csv",
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    sample_rate=296,
    fft_size=512,
    apply_filter=True,
    filter_order=5
)
