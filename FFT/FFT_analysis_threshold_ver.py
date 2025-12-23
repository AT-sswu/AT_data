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


# 샘플링 레이트 계산 함수
def calculate_sample_rate(df, time_column='Time_us'):
    """
    CSV 파일의 시간 열(마이크로초)에서 샘플링 레이트를 계산합니다.
    """
    if time_column not in df.columns:
        # 시간 열이 없으면 첫 번째 열을 시간으로 간주
        time_column = df.columns[0]

    time_data = df[time_column].dropna().values

    if len(time_data) < 2:
        raise ValueError("시간 데이터가 충분하지 않습니다.")

    # 시간 간격 계산 (마이크로초 단위)
    time_diffs = np.diff(time_data)

    # 평균 시간 간격 (마이크로초)
    avg_time_diff_us = np.mean(time_diffs)

    # 평균 시간 간격을 초 단위로 변환
    avg_time_diff_sec = avg_time_diff_us / 1_000_000

    # 샘플링 레이트 = 1 / 평균 시간 간격
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
        fft_size=8192,
        apply_filter=True,
        filter_order=5,
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
            'Sample_Rate': round(sample_rate, 2),
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
        fft_size=8192,
        apply_filter=True,
        filter_order=5,
        threshold_method="std",
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us'
):
    """
    지정된 디렉토리의 모든 CSV 파일에 대해 FFT 분석을 수행합니다.
    """
    data_path = Path(data_dir)

    # CSV 파일 목록 가져오기 - mpu_raw_base_ 패턴으로 변경
    csv_files = sorted(list(data_path.glob("mpu_raw_base_*.csv")))

    if not csv_files:
        print(f"[오류] {data_dir}에서 mpu_raw_base_*.csv 파일을 찾을 수 없습니다.")
        return None

    print(f"총 {len(csv_files)}개의 파일을 찾았습니다.\n")
    print(f"[파일 목록]")
    for f in csv_files:
        print(f"  - {f.name}")

    # 전체 결과를 저장할 리스트
    all_results = []

    print(f"\n{'=' * 60}")
    print(f"BASE 파일 분석 시작 ({len(csv_files)}개 파일)")
    print(f"{'=' * 60}")

    # 모든 파일 처리
    for idx, file_path in enumerate(csv_files, 1):
        print(f"\n[{idx}/{len(csv_files)}] 분석 중: {file_path.name}")

        try:
            file_results = analyze_single_file(
                file_path=str(file_path),
                axes=axes,
                fft_size=fft_size,
                apply_filter=apply_filter,
                filter_order=filter_order,
                threshold_method=threshold_method,
                n_std=n_std,
                recon_error_value=recon_error_value,
                time_column=time_column
            )

            # 클래스 정보를 'base'로 추가
            for result in file_results:
                result['Class'] = 'base'

            all_results.extend(file_results)
            print(f"  ✓ 완료 ({len(file_results)}개 축 분석)")

        except Exception as e:
            import traceback
            print(f"  ✗ 오류 발생: {e}")
            print(f"  ✗ 상세 오류:")
            traceback.print_exc()
            continue

    # 전체 결과를 하나의 CSV로 저장
    if all_results:
        results_df = pd.DataFrame(all_results)

        # 컬럼 순서 재정렬
        column_order = ['Class', 'File', 'Axis', 'Resonance_Frequency_Hz',
                        'Threshold_Value', 'Threshold_Method', 'Threshold_Ranges',
                        'Sample_Rate', 'FFT_Size', 'Filter_Applied', 'Filter_Order']
        results_df = results_df[column_order]

        output_path = data_path / "fft_analysis_base_results.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"\n{'=' * 60}")
        print(f"[완료] 전체 분석 결과 저장됨: {output_path}")
        print(f"총 {len(all_results)}개의 분석 결과 (파일 {len(csv_files)}개 × 축 {len(axes)}개)")
        print(f"{'=' * 60}")

        # 요약 통계
        print("\n[분석 요약]")
        print(f"  분석된 파일 수: {len(csv_files)}개")
        print(f"  분석된 축: {', '.join(axes)}")
        print(f"  총 결과 개수: {len(all_results)}개")

        return results_df
    else:
        print("\n[오류] 분석된 결과가 없습니다.")
        print("[가능한 원인]")
        print("  1. 모든 파일에서 예외가 발생했습니다")
        print("  2. CSV 파일의 열 이름이 예상과 다릅니다")
        print("  3. 데이터가 충분하지 않습니다")
        return None


# 실행
if __name__ == "__main__":
    # 데이터 디렉토리 경로 설정
    DATA_DIR = "/Users/seohyeon/PycharmProjects/AT_data/data_v2"

    # 배치 분석 실행
    results = analyze_all_files(
        data_dir=DATA_DIR,
        axes=["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"],
        fft_size=8192,  # FFT 크기 8192로 설정
        apply_filter=True,
        filter_order=5,
        threshold_method="std",  # "std", "percentile", "recon_error"
        n_std=2.75,
        recon_error_value=0.3,
        time_column='Time_us'  # 시간 열 이름 (마이크로초 단위)
    )

    if results is not None:
        print("\n[결과 미리보기]")
        print(results.head(10))