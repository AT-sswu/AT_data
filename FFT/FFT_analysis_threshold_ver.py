import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path


# ===============================
# Sampling rate 계산
# ===============================
def calculate_sample_rate(df, time_column="Time_us"):
    if time_column not in df.columns:
        time_column = df.columns[0]

    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_dt_sec = np.mean(time_diffs) / 1_000_000
    return 1.0 / avg_dt_sec


# ===============================
# Butterworth LPF
# ===============================
def apply_butterworth_filter(data, sample_rate, cutoff=100.0, order=4):
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, data)


# ===============================
# Band-limited CV
# ===============================
def calculate_band_cv(freqs, amps, f_min=1.0, f_max=100.0):
    band = (freqs >= f_min) & (freqs <= f_max)
    band_amps = amps[band]

    mean_val = np.mean(band_amps)
    if mean_val < 1e-12:
        return np.nan

    return np.std(band_amps) / mean_val * 100.0


# ===============================
# FFT 분석
# ===============================
def fft_analysis(data, sample_rate, window_size=4096, fft_size=16384):
    data = data[:window_size]
    data = data - np.mean(data)

    if fft_size > window_size:
        data = np.pad(data, (0, fft_size - window_size))

    y = fft(data)
    x = fftfreq(fft_size, 1 / sample_rate)

    freqs = x[:fft_size // 2]
    amps = np.abs(y[:fft_size // 2]) * 2 / window_size

    resonance_freq = freqs[np.argmax(amps)]
    return freqs, amps, resonance_freq


# ===============================
# Class 추출 함수 (★ 추가된 부분)
# ===============================
def extract_class_from_filename(filename):
    parts = filename.split("_")
    for p in parts:
        if p.lower().startswith("class"):
            return p
    return "Unknown"


# ===============================
# 단일 파일 분석
# ===============================
def analyze_single_file(
    file_path,
    axes,
    window_size=4096,
    fft_size=16384,
    cutoff_freq=100.0,
    time_column="Time_us"
):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]
    class_name = extract_class_from_filename(file_title)

    sample_rate = calculate_sample_rate(df, time_column)
    print(f"  Sample rate: {sample_rate:.2f} Hz")

    results = []
    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            continue

        data = df[axis].dropna().values

        data = apply_butterworth_filter(
            data,
            sample_rate=sample_rate,
            cutoff=cutoff_freq,
            order=4
        )

        freqs, amps, resonance_freq = fft_analysis(
            data,
            sample_rate,
            window_size,
            fft_size
        )

        band_cv = calculate_band_cv(freqs, amps, 1.0, 100.0)

        results.append({
            "Class": class_name,                 # ★ 추가
            "File": file_title,
            "Axis": axis,
            "Resonance_Frequency_Hz": round(resonance_freq, 2),
            "Band_CV_1_100Hz": round(band_cv, 2),
            "Sample_Rate": round(sample_rate, 2),
            "Window_Size": window_size,
            "FFT_Size": fft_size,
            "Filter": "Butterworth",
            "Cutoff_Hz": cutoff_freq
        })

        plt.subplot(2, 3, idx + 1)
        plt.plot(freqs, amps)
        plt.axvline(resonance_freq, color="r", linestyle="--")
        plt.title(f"{axis} | {resonance_freq:.2f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)

    plt.suptitle(f"FFT Spectrum - {file_title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(
        os.path.join(os.path.dirname(file_path), f"{file_title}_fft.png"),
        dpi=150
    )
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
    time_column="Time_us"
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

    results_df = pd.DataFrame(all_results)
    output_path = data_path / "fft_results_butterworth_bandCV.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n결과 저장 완료: {output_path}")
    return results_df


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
