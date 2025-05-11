import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FFT 분석 함수
def fft_analysis(data, sample_rate=319, duration=60, plot=True): #sample_rate 조정 필요
    # 데이터의 DC 성분(평균값)을 제거하여 중심이 0이 되도록 함
    data = data - data.mean()
    # 샘플 수 계산
    n = sample_rate * duration
    # 샘플 주파수 계산
    x = np.fft.rfftfreq(n, 1 / sample_rate)
    # FFT 계산 (실수 FFT)
    y = np.fft.rfft(data[:n])/len(data)
    # 진폭 계산
    amps = np.abs(y)
    # 공진 주파수 (최대 진폭을 가지는 주파수)
    resonance_freq = x[np.argmax(amps)]

    if plot:
        # FFT 결과 시각화
        plt.figure(figsize=(10, 5))
        plt.plot(x, amps)
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.show()

    return x, amps, resonance_freq

# CSV 파일 읽기 + FFT 분석
def analyze_csv_fft(file_path, axis='Accel_X', sample_rate=319, duration=60):
    df = pd.read_csv(file_path)

    if axis not in df.columns:
        raise ValueError(f"{axis} 열이 CSV 파일에 없습니다. 가능한 열: {df.columns.tolist()}")

    # 데이터 가져오기
    data = df[axis].dropna().values
    # FFT 분석 수행
    freqs, amps, resonance_freq = fft_analysis(data, sample_rate=sample_rate, duration=duration, plot=True)
    print(f"공진 주파수: {resonance_freq:.2f} Hz")

    return freqs, amps, resonance_freq


# 분석 함수 실행
analyze_csv_fft(r"C:\Users\USER\PycharmProjects\AT_data\mpu6050_vibration_data.csv", axis='Accel_X')