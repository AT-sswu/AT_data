import os
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

# 저역통과 필터 구성 함수
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 저역통과 필터 적용 함수
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

# threshold 계산 함수
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

# FFT 분석 함수 및 에너지 계산
def fft_energy(data, sample_rate=296, fft_size=None):
    data = data - np.mean(data)
    n = fft_size if fft_size else len(data)
    data = data[:n]

    y = fft(data)
    x = fftfreq(n, 1 / sample_rate)
    positive_amps = np.abs(y[:n // 2]) * 2 / n
    energy = np.sum(positive_amps**2)
    return energy

# 클래스별 평균 에너지 계산
def compute_class_energy(file_label_map, axes, sample_rate=296, fft_size=512, apply_filter=True, filter_order=5):
    energy_by_class = {}

    for file_path, class_label in file_label_map.items():
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        if class_label not in energy_by_class:
            energy_by_class[class_label] = []

        for axis in axes:
            if axis in df.columns:
                data = df[axis].dropna().values
                if apply_filter:
                    cutoff = sample_rate / 4
                    data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)
                energy = fft_energy(data, sample_rate=sample_rate, fft_size=fft_size)
                energy_by_class[class_label].append(energy)

    mean_energy_by_class = {
        label: np.mean(energies) for label, energies in energy_by_class.items() if energies
    }
    return mean_energy_by_class

# 새 데이터 분류 함수
def classify_with_threshold(file_path, mean_energy_by_class, axes,
                            sample_rate=296, fft_size=512, apply_filter=True, filter_order=5,
                            threshold_method="std", n_std=2.75, recon_error_value=0.3):
    df = pd.read_csv(file_path)
    sample_energies = []

    for axis in axes:
        if axis in df.columns:
            data = df[axis].dropna().values
            if apply_filter:
                cutoff = sample_rate / 4
                data = butter_lowpass_filter(data, cutoff=cutoff, fs=sample_rate, order=filter_order)
            energy = fft_energy(data, sample_rate=sample_rate, fft_size=fft_size)
            sample_energies.append(energy)

    mean_sample_energy = np.mean(sample_energies)

    # 각 클래스와의 에너지 차이 측정
    diffs = {label: abs(mean_sample_energy - class_energy)
             for label, class_energy in mean_energy_by_class.items()}

    closest_class = min(diffs, key=diffs.get)
    diff_value = diffs[closest_class]

    # threshold 기반 필터링
    threshold = calculate_threshold(list(mean_energy_by_class.values()),
                                    method=threshold_method,
                                    n_std=n_std,
                                    recon_error_value=recon_error_value)

    if diff_value > threshold:
        return "etc"

    return closest_class

# 클래스 → 매핑 코드
def get_class_code():
    return {
        "stationary": "00",
        "windy": "01",
        "vibration": "10",
        "windy_vibration": "11",
        "etc": "00"
    }

# 실행 예시
if __name__ == "__main__":
    file_label_map = {
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set1.csv": "stationary",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set2.csv": "stationary",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set3.csv": "stationary",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set4.csv": "stationary",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_stationary_data_set5.csv": "stationary",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set1.csv": "windy",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set2.csv": "windy",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set3.csv": "windy",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set4.csv": "windy",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_windy_data_set5.csv": "windy",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_vibration_data_set1.csv": "vibration",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_vibration_data_set2.csv": "vibration",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_vibration_data_set3.csv": "vibration",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_vibration_data_set4.csv": "vibration",
        r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_vibration_data_set5.csv": "vibration"
    }

    axes = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]

    mean_energy_by_class = compute_class_energy(file_label_map, axes)

    new_file = r"C:\Users\USER\PycharmProjects\AT_data\datasets\mpu6050_static_data.csv"
    predicted_class = classify_with_threshold(
        new_file,
        mean_energy_by_class,
        axes,
        threshold_method="std",          # std 또는 "percentile", "recon_error"
        n_std=2.75,
        recon_error_value=0.3
    )

    label_to_code = get_class_code()

    # 바람 + 진동 복합 처리
    if "windy" in predicted_class and "vibration" in predicted_class:
        predicted_class = "windy_vibration"
    elif "windy" not in predicted_class and "vibration" not in predicted_class:
        predicted_class = "stationary"

    code = label_to_code.get(predicted_class, "00")
    print(f"예측된 클래스: {predicted_class}, 코드: {code}")
