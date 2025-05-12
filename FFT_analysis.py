import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# FFT 분석 함수
def fft_analysis(data, sample_rate=319, fft_size=None, plot=True): #size 조정 필요
    # 평균 제거 (DC 성분 제거)
    data = data - data.mean()

    # 데이터 길이
    n_data = len(data)

    # FFT 사이즈 설정
    n = fft_size if fft_size is not None else n_data

    # Zero-padding
    if n > n_data:
        padded_data = np.pad(data, (0, n - n_data), 'constant')
    else:
        padded_data = data[:n]

    # FFT 계산
    y = fft(padded_data)
    x = fftfreq(n, 1 / sample_rate)

    # 양의 주파수 부분만 추출
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n  # 진폭 정규화

    # 공진 주파수 계산
    resonance_freq = positive_freqs[np.argmax(positive_amps)]

    # 시각화
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(positive_freqs, positive_amps)
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.show()

    return positive_freqs, positive_amps, resonance_freq


# CSV 파일 읽고 분석
def analyze_csv_fft(file_path, axis='Accel_X', sample_rate=319, fft_size=None):
    df = pd.read_csv(file_path)

    if axis not in df.columns:
        raise ValueError(f"{axis} 열이 CSV 파일에 없습니다. 가능한 열: {df.columns.tolist()}")

    # 필요한 열 데이터 추출
    data = df[axis].dropna().values

    # FFT 분석
    freqs, amps, resonance_freq = fft_analysis(data, sample_rate=sample_rate, fft_size=fft_size, plot=True)
    print(f"공진 주파수: {resonance_freq:.2f} Hz")

    return freqs, amps, resonance_freq

# 실행
analyze_csv_fft(
    r"C:\Users\USER\PycharmProjects\AT_data\mpu6050_vibration_data.csv",
    axis='Accel_X',
    sample_rate=319,
    fft_size=2048  # 원하는 FFT 사이즈 지정
)
