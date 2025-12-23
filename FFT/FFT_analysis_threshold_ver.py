import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path


# ===============================
# 샘플링 레이트 계산
# ===============================
def calculate_sample_rate(df, time_column='Time_us'):
    if time_column not in df.columns:
        time_column = df.columns[0]

    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_sec = np.mean(time_diffs) / 1_000_000
    return 1 / avg_time_diff_sec


# ===============================
# Butterworth 저역통과 필터
# ===============================
def apply_butterworth_filter(data, sample_rate, cutoff=100.0, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


# ===============================
# Threshold 계산
# ===============================
def calculate_threshold(amps, method="std", n_std=2.75):
    if method == "std":
        return np.mean(amps) + n_std * np.std(amps)
    elif method == "percentile":
        return np.percentile(amps, 97.5)
    else:
        raise ValueError("지원하지 않는 threshold 방식")


# ===============================
# FFT 분석
# ===============================
def fft_analysis(
        data,
        sample_rate,
        window_size=4096,
        fft_size=16384,
        threshold_method="std",
        n_std=2.75
):
    # Window 적용
    data = data[:window_size]
    data = data - np.mean(data)

    # Zero-padding
    if fft_size > window_size:
        data = np.pad(data, (0, fft_size - window_size))

    y = fft(data)
    x = fftfreq(fft_size, 1 / sample_rate)

    freqs = x[:fft_size // 2]
    amps = np.abs(y[:fft_size // 2]) * 2 / window_size

    resonance_freq = freqs[np.argmax(amps)]
    threshold = calculate_threshold(amps, threshold_method, n_std)

    return freqs, amps, resonance_freq, threshold


# ===============================
# 단일 파일 분석
# ===============================
def analyze_single_file(
        file_path,
        axes,
        window_size=4096,
        fft_size=16384,
        cutoff_freq=100.0,
        time_column='Time_us'
):
    df = pd.read_csv(file_path)
    sample_rate = calculate_sample_rate(df, time_column)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    print(f"  Sample rate: {sample_rate:.2f} Hz")

    results = []
    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            continue

        data = df[axis].dropna().values

        # Butterworth 필터
        data = apply_butterworth_filter(
            data,
            sample_rate=sample_rate,
            cutoff=cutoff_freq,
            order=4
        )

        freqs, amps, resonance_freq, threshold = fft_analysis(
            data,
            sample_rate,
            window_size,
            fft_size
        )

        results.append({
            "File": file_title,
            "Axis": axis,
            "Resonance_Frequency_Hz": round(resonance_freq, 2),
            "Sample_Rate": round(sample_rate, 2),
            "Window_Size": window_size,
            "FFT_Size": fft_size,
            "Filter": "Butterworth",
            "Cutoff_Hz": cutoff_freq
        })

        plt.subplot(2, 3, idx + 1)
        plt.plot(freqs, amps)
        plt.axvline(resonance_freq, color='r', linestyle='--')
        plt.axhline(threshold, color='g', linestyle=':')
        plt.title(f"{axis} | {resonance_freq:.2f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.suptitle(f"FFT Spectrum - {file_title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(os.path.dirname(file_path), f"{file_title}_fft.png"), dpi=150)
    plt.close()

    return results


# ===============================
# 전체 파일 분석
# ===============================
def analyze_all_files(
        data_dir,
        axes,
        window_size=4096,
        fft_size=16384,
        cutoff_freq=100.0,
        time_column='Time_us'
):
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("mpu_raw_optimized_*.csv"))

    all_results = []

    for idx, file_path in enumerate(csv_files, 1):
        print(f"\n[{idx}/{len(csv_files)}] {file_path.name}")
        file_results = analyze_single_file(
            file_path,
            axes,
            window_size,
            fft_size,
            cutoff_freq,
            time_column
        )
        all_results.extend(file_results)

    df_results = pd.DataFrame(all_results)
    output_path = data_path / "fft_results_butterworth.csv"
    df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n결과 저장: {output_path}")
    return df_results


# ===============================
# 실행
# ===============================
if __name__ == "__main__":
    DATA_DIR = "/Users/seohyeon/PycharmProjects/AT_data/data_v1"

    analyze_all_files(
        data_dir=DATA_DIR,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        window_size=4096,
        fft_size=16384,
        cutoff_freq=100.0,
        time_column="Time_us"
    )
