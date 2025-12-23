import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일 불러오기
file_path = '../data_v0/mpu6050_vibration_data_set5.csv'
file_title = os.path.splitext(os.path.basename(file_path))[0]
df = pd.read_csv(file_path)

# 시간 데이터 (μs → s)
t = df['Time'].values / 1_000_000

# 1. 시간 간격 Dt 계산 (샘플 간 시간 차이)
dt_array = np.diff(t)  # t[i+1] - t[i]
Dt = np.mean(dt_array)
# 2. 샘플링 주파수 fs (1초당 몇 번 측정했는지)
fs = 1 / Dt
# 3. 나이퀴스트 주파수 fn (최대 해석 가능한 주파수)
fn = fs / 2
# 4. 샘플 개수 N (데이터 길이)
N = len(t)
# 5. 주파수 해상도 df (FFT로 얻을 수 있는 주파수 간격)
df_ = fs / N
# 6. 신호 총 길이 duration
duration = t[-1] - t[0]

# 결과
print(f"샘플 간 간격 Dt: {Dt:.3e} 초")
print(f"샘플링 주파수 fs: {fs:.2f} Hz")
print(f"나이퀴스트 주파수 fn: {fn:.2f} Hz")
print(f"샘플 수 N: {N}")
print(f"주파수 해상도 df: {df_:.4f} Hz")
print(f"신호의 총 길이 duration: {duration:.2f} 초")

# 센서 축
sensor_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# 전체 subplot
plt.figure(figsize=(18, 10))

for i, col in enumerate(sensor_columns):
    data = df[col].values
    plt.subplot(2, 3, i + 1)  # 2행 3열 subplot
    plt.plot(t, data)
    plt.title(f"{col}")
    plt.xlabel("Time (s)")
    plt.ylabel(col)
    plt.grid(True)

plt.suptitle(f"{file_title} - Sensor Signals (Time Domain)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 전체 제목 공간 확보
plt.show()
