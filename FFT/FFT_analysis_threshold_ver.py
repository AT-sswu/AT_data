import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from pathlib import Path


# 칼만 필터 클래스
class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, estimated_measurement_variance=1e-2):
        """
        1D 칼만 필터 초기화

        Parameters:
        -----------
        process_variance : float
            프로세스 노이즈 분산 (Q) - 시스템 모델의 불확실성
        measurement_variance : float
            측정 노이즈 분산 (R) - 센서 측정의 불확실성
        estimated_measurement_variance : float
            초기 추정 오차 공분산 (P)
        """
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R
        self.estimated_measurement_variance = estimated_measurement_variance  # P
        self.posteri_estimate = 0.0  # 사후 추정값
        self.posteri_error_estimate = 1.0  # 사후 오차 추정

    def update(self, measurement):
        """
        칼만 필터 업데이트

        Parameters:
        -----------
        measurement : float
            새로운 측정값

        Returns:
        --------
        float : 필터링된 값
        """
        # 예측 단계 (Prediction)
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # 업데이트 단계 (Update)
        kalman_gain = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + kalman_gain * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * priori_error_estimate

        return self.posteri_estimate


def apply_kalman_filter(data, process_variance=1e-5, measurement_variance=1e-2):
    """
    데이터에 칼만 필터 적용

    Parameters:
    -----------
    data : numpy.array
        필터링할 데이터
    process_variance : float
        프로세스 노이즈 분산
    measurement_variance : float
        측정 노이즈 분산

    Returns:
    --------
    numpy.array : 필터링된 데이터
    """
    kf = KalmanFilter(
        process_variance=process_variance,
        measurement_variance=measurement_variance
    )

    filtered_data = np.zeros(len(data))
    for i, measurement in enumerate(data):
        filtered_data[i] = kf.update(measurement)

    return filtered_data


# 저역통과 필터 함수
# 저역통과 필터 함수 제거됨 (칼만 필터만 사용)


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
        fft_size=16384,
        filter_type='kalman',  # 'none', 'kalman'
        kalman_process_var=1e-5,
        kalman_measurement_var=1e-2,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us'
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
        filter_applied_str = filter_type

        # 칼만 필터 적용
        if filter_type == 'kalman':
            data = apply_kalman_filter(data, kalman_process_var, kalman_measurement_var)

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
            'Sample_Rate': round(sample_rate, 2),
            'FFT_Size': fft_size,
            'Filter_Type': filter_applied_str,
            'Kalman_Process_Var': kalman_process_var if filter_type == 'kalman' else 'N/A',
            'Kalman_Measurement_Var': kalman_measurement_var if filter_type == 'kalman' else 'N/A'
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

    plt.suptitle(f"FFT Frequency Spectrum - {file_title} (Filter: {filter_type})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(os.path.dirname(file_path), f"{file_title}_fft_plot_{filter_type}.png"), dpi=150)
    plt.close()

    return results


# 배치 분석 함수 (모든 파일 처리)
def analyze_all_files(
        data_dir,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        fft_size=16384,
        filter_type='kalman',  # 'none', 'kalman'
        kalman_process_var=1e-5,
        kalman_measurement_var=1e-2,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us'
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
    print(f"사용 필터: {filter_type}")

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
                    fft_size=fft_size,
                    filter_type=filter_type,
                    kalman_process_var=kalman_process_var,
                    kalman_measurement_var=kalman_measurement_var,
                    threshold_method=threshold_method,
                    n_std=n_std,
                    recon_error_value=recon_error_value,
                    time_column=time_column
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
                        'Sample_Rate', 'FFT_Size', 'Filter_Type',
                        'Kalman_Process_Var', 'Kalman_Measurement_Var']
        results_df = results_df[column_order]

        output_path = data_path / f"all_files_fft_analysis_results_{filter_type}.csv"
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
    # filter_type 옵션: 'none', 'kalman'
    results = analyze_all_files(
        data_dir=DATA_DIR,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        fft_size=16384,
        filter_type='kalman',  # 칼만 필터 사용
        kalman_process_var=1e-5,  # 칼만 필터 프로세스 노이즈
        kalman_measurement_var=1e-2,  # 칼만 필터 측정 노이즈
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us'
    )

    if results is not None:
        print("\n[결과 미리보기]")
        print(results.head(10))