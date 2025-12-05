import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path


# 저역통과 필터 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


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
        sample_rate=296,
        fft_size=None,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3
):
    data = data - np.mean(data)
    n = fft_size if fft_size else len(data)
    data = data[:n]

    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_freqs = x[:n // 2]
    positive_amps = np.abs(y[:n // 2]) * 2 / n
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
        sample_rate=287,
        fft_size=512,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3
):
    df = pd.read_csv(file_path)
    file_title = os.path.splitext(os.path.basename(file_path))[0]

    results = []

    num_rows = 2
    num_cols = 3

    plt.figure(figsize=(18, 8))

    for idx, axis in enumerate(axes):
        if axis not in df.columns:
            print(f"[경고] {axis} 열이 CSV 파일에 없습니다: {file_path}")
            continue

        data = df[axis].dropna().values

        if apply_filter:
            cutoff = sample_rate / 4
            data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)

        freqs, amps, resonance_freq, threshold_ranges, threshold = fft_analysis(
            data,
            sample_rate=sample_rate,
            fft_size=fft_size,
            threshold_method=threshold_method,
            n_std=n_std,
            recon_error_value=recon_error_value
        )

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
            'Threshold_Value': round(threshold, 4),
            'Threshold_Method': threshold_method,
            'Threshold_Ranges': threshold_ranges_str,
            'Sample_Rate': sample_rate,
            'FFT_Size': fft_size,
            'Filter_Applied': apply_filter,
            'Filter_Order': filter_order if apply_filter else 'N/A'
        })

        # subplot 그리기
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
    plt.savefig(os.path.join(os.path.dirname(file_path), f"{file_title}_fft_plot.png"), dpi=150)
    plt.close()

    return results


# 배치 분석 함수 (모든 파일 처리)
def analyze_all_files(
        data_dir,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        sample_rate=287,
        fft_size=512,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3
):
    """
    지정된 디렉토리의 모든 CSV 파일에 대해 FFT 분석을 수행합니다.

    Parameters:
    -----------
    data_dir : str
        데이터 파일이 있는 디렉토리 경로
    """
    data_path = Path(data_dir)

    # CSV 파일 목록 가져오기
    csv_files = sorted(list(data_path.glob("mpu_raw_optimized_*.csv")))

    if not csv_files:
        print(f"[오류] {data_dir}에서 CSV 파일을 찾을 수 없습니다.")
        return None

    print(f"총 {len(csv_files)}개의 파일을 찾았습니다.\n")

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
                    sample_rate=sample_rate,
                    fft_size=fft_size,
                    apply_filter=apply_filter,
                    filter_order=filter_order,
                    threshold_method=threshold_method,
                    n_std=n_std,
                    recon_error_value=recon_error_value
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
        column_order = ['Class', 'File', 'Axis', 'Resonance_Frequency_Hz',
                        'Threshold_Value', 'Threshold_Method', 'Threshold_Ranges',
                        'Sample_Rate', 'FFT_Size', 'Filter_Applied', 'Filter_Order']
        results_df = results_df[column_order]

        output_path = data_path / "all_files_fft_analysis_results.csv"
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
        sample_rate=287,
        fft_size=512,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",  # "std", "percentile", "recon_error"
        n_std=2.75,
        recon_error_value=0.3
    )

    if results is not None:
        print("\n[결과 미리보기]")
        print(results.head(10))