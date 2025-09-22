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

    # CSV 저장을 위한 결과 리스트
    results = []

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

        # 결과를 리스트에 추가
        results.append({
            'Axis': axis,
            'Resonance_Frequency_Hz': round(resonance_freq, 2),
            'Sample_Rate': sample_rate,
            'FFT_Size': fft_size,
            'Filter_Applied': apply_filter,
            'Filter_Order': filter_order if apply_filter else 'N/A'
        })

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

    # 결과를 CSV로 저장
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(file_path), f"{file_title}_fft_analysis_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[결과 저장] FFT 분석 결과가 저장되었습니다: {output_path}")

    return results_df

# 실행
analyze_multiple_axes(
    r"/Users/seohyeon/PycharmProjects/AT_data/datasets/mpu6050_vibration_data_set5.csv",
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    sample_rate=296,
    fft_size=512,
    apply_filter=True,
    filter_order=5
)