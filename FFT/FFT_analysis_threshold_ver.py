import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path


# ===============================
# Butterworth LPF
# ===============================
def apply_butterworth_filter(data, sample_rate, cutoff=100.0, order=4):
    """
    Butterworth 저역통과 필터 적용
    """
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low")
    return filtfilt(b, a, data)


# 샘플링 레이트 계산 함수
def calculate_sample_rate(df, time_column='Time_us'):
    """
    CSV 파일의 시간 열(마이크로초)에서 샘플링 레이트를 계산합니다.
    """
    if time_column not in df.columns:
        time_column = df.columns[0]

    time_data = df[time_column].dropna().values
    if len(time_data) < 2:
        raise ValueError("시간 데이터가 충분하지 않습니다.")

    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    avg_time_diff_sec = avg_time_diff_us / 1_000_000
    sample_rate = 1 / avg_time_diff_sec
    return sample_rate


# ===============================
# Band-limited CV
# ===============================
def calculate_band_cv(freqs, amps, f_min=1.0, f_max=100.0):
    """
    특정 주파수 대역의 Coefficient of Variation 계산
    """
    band = (freqs >= f_min) & (freqs <= f_max)
    band_amps = amps[band]

    mean_val = np.mean(band_amps)
    if mean_val < 1e-12:
        return np.nan

    return np.std(band_amps) / mean_val * 100.0


# Threshold 계산 함수
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


# FFT 분석 함수
def fft_analysis(
    data,
    sample_rate,
    window_size=4096,
    fft_size=16384,
    threshold_method="std",
    n_std=2.75,
    recon_error_value=0.3
):
    """
    FFT 분석 수행
    window_size 만큼의 데이터를 사용하고, fft_size로 zero-padding
    """
    # window_size 만큼만 사용
    data = data[:window_size]
    data = data - np.mean(data)

    # zero-padding
    if fft_size > window_size:
        data = np.pad(data, (0, fft_size - window_size))

    y = fft(data)
    x = fftfreq(fft_size, 1 / sample_rate)

    positive_freqs = x[:fft_size // 2]
    positive_amps = np.abs(y[:fft_size // 2]) * 2 / window_size

    resonance_freq = positive_freqs[np.argmax(positive_amps)]
    threshold = calculate_threshold(positive_amps, threshold_method, n_std, recon_error_value)

    # Threshold 이상 구간 탐색
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


# 단일 파일 분석 함수
def analyze_single_file(
    file_path,
    axes,
    window_size=4096,
    fft_size=16384,
    cutoff_freq=100.0,
    threshold_method="std",
    n_std=2.75,
    recon_error_value=0.3,
    time_column='Time_us',
    band_cv_f_min=0.3,
    band_cv_f_max=60.0
):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    # 샘플링 레이트 자동 계산
    sample_rate = calculate_sample_rate(df, time_column)
    print(f"  계산된 샘플링 레이트: {sample_rate:.2f} Hz")

    results = []
    num_rows = 2
    num_cols = 3
    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다: {file_path}")
            continue

        data = df[axis].dropna().values

        # Butterworth 필터 적용
        data = apply_butterworth_filter(
            data,
            sample_rate=sample_rate,
            cutoff=cutoff_freq,
            order=4
        )

        freqs, amps, resonance_freq, threshold_ranges, threshold = fft_analysis(
            data,
            sample_rate=sample_rate,
            window_size=window_size,
            fft_size=fft_size,
            threshold_method=threshold_method,
            n_std=n_std,
            recon_error_value=recon_error_value
        )

        # Band CV 계산
        band_cv = calculate_band_cv(freqs, amps, f_min=band_cv_f_min, f_max=band_cv_f_max)

        # Threshold 범위를 문자열로 변환
        threshold_ranges_str = ""
        if threshold_ranges:
            ranges_list = [f"{r[0]:.2f}-{r[1]:.2f}Hz" for r in threshold_ranges]
            threshold_ranges_str = "; ".join(ranges_list)
        else:
            threshold_ranges_str = "없음"

        results.append({
            'File': file_title,
            'Axis': axis,
            'Resonance_Frequency_Hz': round(resonance_freq, 2),
            'Band_CV': round(band_cv, 2) if not np.isnan(band_cv) else 'N/A',
            'Band_CV_Freq_Range': f"{band_cv_f_min}-{band_cv_f_max}Hz",
            'Threshold_Value': round(threshold, 4),
            'Threshold_Method': threshold_method,
            'Threshold_Ranges': threshold_ranges_str,
            'Sample_Rate': round(sample_rate, 2),
            'Window_Size': window_size,
            'FFT_Size': fft_size,
            'Filter_Type': 'Butterworth',
            'Cutoff_Hz': cutoff_freq
        })

        # subplot 그리기
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.plot(freqs, amps, label="Amplitude")
        plt.axvline(resonance_freq, color='r', linestyle='--', label=f'Resonance: {resonance_freq:.2f} Hz')
        plt.axhline(y=threshold, color='g', linestyle=':', label=f'Threshold: {threshold:.3f}')
        plt.title(f"{axis}\nResonance: {resonance_freq:.2f} Hz | CV: {band_cv:.2f}%")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()

    plt.suptitle(f"FFT Frequency Spectrum - {file_title} (Filter: Butterworth)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(os.path.dirname(file_path), f"{file_title}_fft_plot_butterworth.png"), dpi=150)
    plt.close()

    return results


# 배치 분석 함수 (모든 파일 처리)
def analyze_all_files(
    data_dir,
    axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
    window_size=4096,
    fft_size=16384,
    cutoff_freq=100.0,
    threshold_method="std",
    n_std=2.75,
    recon_error_value=0.3,
    time_column='Time_us',
    band_cv_f_min=0.3,
    band_cv_f_max=60.0
):
    """
    지정된 디렉토리의 모든 CSV 파일에 대해 FFT 분석을 수행합니다.
    """
    data_path = Path(data_dir)

    # CSV 파일 목록 가져오기
    csv_files = sorted(list(data_path.glob("mpu_raw_optimized_*.csv")))

    if not csv_files:
        print(f"[오류] {data_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return None

    print(f"총 {len(csv_files)}개의 파일을 찾았습니다.\n")
    print(f"사용 필터: Butterworth (Cutoff: {cutoff_freq} Hz)")
    print(f"Window Size: {window_size}, FFT Size: {fft_size}")

    # 클래스별로 파일 분류
    class_files = {
        'driving': [],
        'lidar': [],
        'motor': [],
        'lidar_driving': [],
        'motor_driving': []
    }

    for file in csv_files:
        filename = file.name
        if 'motor_driving' in filename:
            class_files['motor_driving'].append(file)
        elif 'lidar_driving' in filename:
            class_files['lidar_driving'].append(file)
        elif 'driving' in filename:
            class_files['driving'].append(file)
        elif 'lidar' in filename:
            class_files['lidar'].append(file)
        elif 'motor' in filename:
            class_files['motor'].append(file)

    # 전체 결과를 저장할 리스트
    all_results = []

    # 각 클래스별로 처리
    for class_name, files in class_files.items():
        if not files:
            continue

        print(f"\n{'=' * 60}")
        print(f"클래스: {class_name.upper()} ({len(files)}개 파일)")
        print(f"{'=' * 60}")

        for idx, file_path in enumerate(files, 1):
            print(f"\n[{idx}/{len(files)}] 분석 중: {file_path.name}")
            try:
                file_results = analyze_single_file(
                    file_path=str(file_path),
                    axes=axes,
                    window_size=window_size,
                    fft_size=fft_size,
                    cutoff_freq=cutoff_freq,
                    threshold_method=threshold_method,
                    n_std=n_std,
                    recon_error_value=recon_error_value,
                    time_column=time_column,
                    band_cv_f_min=band_cv_f_min,
                    band_cv_f_max=band_cv_f_max
                )

                # 클래스 정보 추가
                for result in file_results:
                    result['Class'] = class_name

                all_results.extend(file_results)
                print(f"  ✓ 완료")
            except Exception as e:
                print(f"  ✗ 오류 발생: {e}")
                continue

    # 전체 결과를 하나의 CSV로 저장
    if all_results:
        results_df = pd.DataFrame(all_results)

        # 컬럼 순서 재정렬
        column_order = ['Class', 'File', 'Axis', 'Resonance_Frequency_Hz', 'Band_CV',
                       'Band_CV_Freq_Range', 'Threshold_Value', 'Threshold_Method',
                       'Threshold_Ranges', 'Sample_Rate', 'Window_Size', 'FFT_Size',
                       'Filter_Type', 'Cutoff_Hz']
        results_df = results_df[column_order]

        output_path = data_path / f"all_files_fft_analysis_butterworth_bandCV.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 60}")
        print(f"[완료] 전체 분석 결과 저장됨: {output_path}")
        print(f"총 {len(all_results)}개의 분석 결과 (파일 {len(csv_files)}개 × 축 {len(axes)}개)")
        print(f"{'=' * 60}")

        # 클래스별 요약 통계
        print("\n[클래스별 요약]")
        for class_name in class_files.keys():
            class_data = results_df[results_df['Class'] == class_name]
            if len(class_data) > 0:
                print(f"  {class_name}: {len(class_data) // len(axes)}개 파일")

        return results_df
    else:
        print("[오류] 분석된 결과가 없습니다.")
        return None


# 실행
if __name__ == "__main__":
    # 데이터 디렉토리 경로 설정
    DATA_DIR = "/Users/seohyeon/PycharmProjects/AT_data/data_v1"

    # 배치 분석 실행
    results = analyze_all_files(
        data_dir=DATA_DIR,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        window_size=4096,
        fft_size=16384,
        cutoff_freq=100.0,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us',
        band_cv_f_min=0.3,
        band_cv_f_max=60.0
    )

    if results is not None:
        print("\n[결과 미리보기]")
        print(results.head(10))