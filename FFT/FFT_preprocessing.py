import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../mpu6050_vibration_data.csv')

t = df['Time'].values / 1_000_000  # 시간 배열 변환
data = df['Accel_X'].values

# 1. 시간 간격 Dt 계산 (샘플 간 시간 차이)
dt_array = np.diff(t)  # t[i+1] - t[i] -> 전체 간격 계산
Dt = np.mean(dt_array)  # 평균으로 변환
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

print(f"샘플 간 간격 Dt: {Dt:.3e} 초")
print(f"샘플링 주파수 fs: {fs:.2f} Hz")
print(f"나이퀴스트 주파수 fn: {fn:.2f} Hz")
print(f"샘플 수 N: {N}")
print(f"주파수 해상도 df: {df_:.4f} Hz")
print(f"신호의 총 길이 duration: {duration:.2f} 초")

plt.figure(figsize=(10, 4))
plt.plot(t, data)
plt.title("Original Signal vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Signal (Accel_X)")
plt.grid(True)
plt.show()

