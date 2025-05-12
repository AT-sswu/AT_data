import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# FFT 분석 함수
def fft_analysis(data, sample_rate=319, plot=True):
    # 데이터의 DC 성분(평균값)을 제거하여 중심이 0이 되도록 함
    data = data - data.mean()

    # 샘플 수 계산
    n = len(data)

    # FFT 계산 (복소수 FFT로 변경)
    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)

    # 양의 주파수만
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n  # 진폭 정규화

    # 공진 주파수 (최대 진폭을 가지는 주파수)
    resonance_freq = positive_freqs[np.argmax(positive_amps)]

    if plot:
        # FFT 결과
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


# CSV 파일 읽기 + FFT 분석
def analyze_csv_fft(file_path, axis='Accel_X', sample_rate=319):
    df = pd.read_csv(file_path)

    if axis not in df.columns:
        raise ValueError(f"{axis} 열이 CSV 파일에 없습니다. 가능한 열: {df.columns.tolist()}")

    # 데이터 가져오기
    data = df[axis].dropna().values

    # FFT 분석 수행
    freqs, amps, resonance_freq = fft_analysis(data, sample_rate=sample_rate, plot=True)
    print(f"공진 주파수: {resonance_freq:.2f} Hz")

    return freqs, amps, resonance_freq

# 분석 함수
analyze_csv_fft(r"C:\Users\USER\PycharmProjects\AT_data\mpu6050_vibration_data.csv", axis='Accel_X')
