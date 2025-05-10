import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FFT 분석 함수
def fft_analysis(data, sample_rate=10000, plot=True):
    n = len(data)
    x = np.fft.fftfreq(n, 1 / sample_rate)
    y = np.fft.fft(data) / n
    mask = x >= 0
    freqs = x[mask]
    amps = np.abs(y[mask])

    # 공진 주파수 (최대 진폭을 가지는 주파수)
    resonance_freq = freqs[np.argmax(amps)]

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, amps)
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.show()

    return freqs, amps, resonance_freq

# CSV 파일 읽기 + FFT 분석
def analyze_csv_fft(file_path, axis='Gyro_Y'):
    df = pd.read_csv(file_path)

    if axis not in df.columns:
        raise ValueError(f"{axis} 열이 CSV 파일에 없습니다. 가능한 열: {df.columns.tolist()}")

    data = df[axis].dropna().values
    freqs, amps, resonance_freq = fft_analysis(data, sample_rate=10000, plot=True)

    print(f"공진 주파수: {resonance_freq:.2f} Hz")

    return freqs, amps, resonance_freq


analyze_csv_fft(r"C:\Users\USER\PycharmProjects\AT\AT_AFO_project\AT_data\mpu6050_data.csv", axis='Accel_X')