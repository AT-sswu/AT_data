import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# FFT 분석 함수
def fft_analysis(data, sample_rate=319, duration=None, plot=True):
    # 데이터 평균 제거
    data = data - np.mean(data)

    # 샘플 수 계산
    if duration is not None:
        n = int(sample_rate * duration)
        if len(data) < n:
            raise ValueError(f"데이터 길이 {len(data)}보다 duration * sample_rate = {n}이 더 큽니다.")
        data = data[:n]
    else:
        n = len(data)

    # 주파수 벡터 (실수 FFT)
    freqs = np.fft.rfftfreq(n, d=1 / sample_rate)
    # FFT 계산
    fft_result = np.fft.rfft(data)
    # 진폭 계산
    amps = np.abs(fft_result) / n
    # 공진 주파수
    resonance_freq = freqs[np.argmax(amps)]

    if plot:
        # FFT 결과
        plt.figure(figsize=(10, 5))
        plt.plot(freqs, amps)
        plt.title("FFT Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return freqs, amps, resonance_freq

# CSV 파일 읽기 + FFT 분석
def analyze_csv_fft(file_path, axis='Accel_X', sample_rate=319, duration=None):
    df = pd.read_csv(file_path)

    if axis not in df.columns:
        raise ValueError(f"{axis} 열이 CSV 파일에 없습니다. 가능한 열: {df.columns.tolist()}")

    # NaN 제거 및 데이터 추출
    data = df[axis].dropna().values

    # FFT 분석 수행
    freqs, amps, resonance_freq = fft_analysis(data, sample_rate=sample_rate, duration=duration, plot=True)
    print(f"공진 주파수: {resonance_freq:.2f} Hz")

    return freqs, amps, resonance_freq

# 분석 함수 실행
analyze_csv_fft(
    r"C:\Users\USER\PycharmProjects\AT_data\mpu6050_static_data.csv",
    axis='Accel_X',
    sample_rate=319,
    #duration=64
)