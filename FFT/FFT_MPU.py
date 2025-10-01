import smbus2
import time
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from datetime import datetime

# ==================== MPU6050 설정 ====================
MPU_ADDR = 0x68
bus = smbus2.SMBus(1)
interval = 0.001


def write_reg(addr, value):
    bus.write_byte_data(MPU_ADDR, addr, value)


def read_regs(addr, length):
    return bus.read_i2c_block_data(MPU_ADDR, addr, length)


def read_word(high, low):
    value = (high << 8) | low
    if value >= 0x8000:
        value -= 0x10000
    return value


def init_mpu():
    write_reg(0x6B, 0x00)
    write_reg(0x19, 0x07)
    write_reg(0x1A, 0x00)


# ==================== FFT 분석 함수 ====================
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


def calculate_threshold(amps, method="std", n_std=2.75, recon_error_value=0.3):
    if method == "std":
        mean = np.mean(amps)
        std = np.std(amps)
        threshold = mean + n_std * std
    elif method == "percentile":
        threshold = np.percentile(amps, 97.5)
    elif method == "recon_error":
        threshold = recon_error_value
    else:
        raise ValueError("지원하지 않는 threshold 방법입니다.")
    return threshold


def fft_analysis(data, sample_rate, fft_size=None,
                 threshold_method="std", n_std=2.75, recon_error_value=0.3):
    data = data - np.mean(data)
    n = fft_size if fft_size else len(data)
    data = data[:n]

    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n
    resonance_freq = positive_freqs[np.argmax(positive_amps)]

    threshold = calculate_threshold(positive_amps, threshold_method, n_std, recon_error_value)

    threshold_ranges = []
    above_threshold = positive_amps >= threshold
    in_range = False

    for i in range(len(positive_freqs)):
        if above_threshold[i] and not in_range:
            range_start = positive_freqs[i]
            in_range = True
        elif not above_threshold[i] and in_range:
            range_end = positive_freqs[i - 1]
            threshold_ranges.append((range_start, range_end))
            in_range = False

    if in_range:
        threshold_ranges.append((range_start, positive_freqs[-1]))

    return positive_freqs, positive_amps, resonance_freq, threshold_ranges, threshold


def analyze_multiple_axes(df, sample_rate, fft_size, file_title, save_dir,
                          apply_filter=True, filter_order=5,
                          threshold_method="std", n_std=2.75, recon_error_value=0.3):
    axes = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
    results = []

    num_rows = 2
    num_cols = 3
    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 데이터에 없습니다.")
            continue

        data = df[axis].dropna().values

        if apply_filter:
            cutoff = sample_rate / 4
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        print(f"\n[분석 축: {axis}]")
        freqs, amps, resonance_freq, threshold_ranges, threshold = fft_analysis(
            data,
            sample_rate=sample_rate,
            fft_size=fft_size,
            threshold_method=threshold_method,
            n_std=n_std,
            recon_error_value=recon_error_value
        )

        print(f"→ 공진 주파수: {resonance_freq:.2f} Hz")
        print(f"→ 사용된 Threshold 값: {threshold:.4f} (기준: {threshold_method})")

        threshold_ranges_str = ""
        if threshold_ranges:
            print("→ Threshold 이상 구간:")
            ranges_list = []
            for r in threshold_ranges:
                range_str = f"{r[0]:.2f}-{r[1]:.2f}Hz"
                ranges_list.append(range_str)
                print(f"   - {r[0]:.2f} Hz ~ {r[1]:.2f} Hz")
            threshold_ranges_str = "; ".join(ranges_list)
        else:
            print("→ Threshold 이상 구간 없음.")
            threshold_ranges_str = "없음"

        results.append({
            'Axis': axis,
            'Resonance_Frequency_Hz': round(resonance_freq, 2),
            'Threshold_Value': round(threshold, 4),
            'Threshold_Method': threshold_method,
            'Threshold_Ranges': threshold_ranges_str,
            'Sample_Rate': sample_rate,
            'FFT_Size': fft_size,
            'Filter_Applied': apply_filter,
            'Filter_Order': filter_order if apply_filter else 'N/A'
        })

        plt.subplot(num_rows, num_cols, idx + 1)
        plt.plot(freqs, amps, label="Amplitude")
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.axhline(y=threshold, color='g', linestyle=':', label=f'Threshold: {threshold:.3f}')
        plt.title(f"{axis}\nResonance: {resonance_freq:.2f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.suptitle(f"FFT Frequency Spectrum - {file_title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    results_df = pd.DataFrame(results)
    output_path = os.path.join(save_dir, f"{file_title}_fft_analysis_results.csv")
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[결과 저장] FFT 분석 결과가 저장되었습니다: {output_path}")

    return results_df


# ==================== 메인 실행 ====================
DURATION = 10
SAVE_DIR = "/home/pi/mpu6050_data"

os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"mpu6050_data_{timestamp}.csv"
csv_filepath = os.path.join(SAVE_DIR, csv_filename)

print("=" * 60)
print("MPU6050 실시간 데이터 수집 및 FFT 분석 시작")
print("=" * 60)
print(f"수집 시간: {DURATION}초")
print(f"저장 경로: {csv_filepath}")
print("=" * 60)

init_mpu()

data_list = []
time_list = []

try:
    start_time = time.time()
    last_time = time.perf_counter()

    print("\n[데이터 수집 중...]")

    while (time.time() - start_time) < DURATION:
        now = time.perf_counter()

        if (now - last_time) >= interval:
            last_time += interval

            acc_bytes = read_regs(0x3B, 6)
            acc = [read_word(acc_bytes[i], acc_bytes[i + 1]) for i in range(0, 6, 2)]

            gyro_bytes = read_regs(0x43, 6)
            gyro = [read_word(gyro_bytes[i], gyro_bytes[i + 1]) for i in range(0, 6, 2)]

            t_us = int(time.time() * 1_000_000)

            time_list.append(t_us)
            data_list.append({
                'Time_us': t_us,
                'Accel_X': acc[0],
                'Accel_Y': acc[1],
                'Accel_Z': acc[2],
                'Gyro_X': gyro[0],
                'Gyro_Y': gyro[1],
                'Gyro_Z': gyro[2]
            })

    print(f"[데이터 수집 완료] 총 {len(data_list)}개 샘플 수집")

    df = pd.DataFrame(data_list)
    df.to_csv(csv_filepath, index=False, encoding='utf-8-sig')
    print(f"[저장 완료] {csv_filepath}")

    time_diffs = np.diff(df['Time_us'].values)
    avg_dt = np.mean(time_diffs) / 1_000_000
    calculated_sample_rate = round(1 / avg_dt)
    calculated_fft_size = calculated_sample_rate * 2

    print("\n" + "=" * 60)
    print("샘플링 파라미터 계산 결과")
    print("=" * 60)
    print(f"평균 샘플링 간격: {avg_dt * 1000:.3f} ms")
    print(f"계산된 샘플링 레이트: {calculated_sample_rate} Hz")
    print(f"설정된 FFT 크기: {calculated_fft_size}")
    print("=" * 60)

    print("\n[FFT 분석 시작...]")

    results_df = analyze_multiple_axes(
        df=df,
        sample_rate=calculated_sample_rate,
        fft_size=calculated_fft_size,
        file_title=os.path.splitext(csv_filename)[0],
        save_dir=SAVE_DIR,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3
    )

    print("\n" + "=" * 60)
    print("모든 작업이 완료되었습니다")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\n[중단됨] 사용자에 의해 프로그램이 중단되었습니다.")

except Exception as e:
    print(f"\n[오류 발생] {str(e)}")

finally:
    bus.close()
    print("[정리 완료] I2C 버스가 닫혔습니다.")
