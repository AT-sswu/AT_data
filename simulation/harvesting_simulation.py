"""
에너지 하베스팅 시뮬레이션 (5개 클래스)
- 기존 방법(고정 공진 주파수) vs 제안 방법(AI 기반 적응형)
- 가상 하베스터 물리 모델 기반 에너지 수확량 계산
- 배터리 수명 비교
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import pickle
import tensorflow as tf
from tensorflow import keras

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 물리 파라미터 설정 ====================

# 하베스터 물리 특성 (일반적인 압전 하베스터 기준)
FIXED_RESONANCE_FREQ = 60.0  # 기존 방법: 고정 공진 주파수 (Hz)
MAX_POWER_OUTPUT = 10.0  # 최대 출력 전력 (mW) - 공진 시
QUALITY_FACTOR = 50  # 품질 계수 Q (높을수록 대역폭 좁음)
BANDWIDTH = FIXED_RESONANCE_FREQ / QUALITY_FACTOR  # 유효 대역폭 (~1.2Hz)

# 환경 진동 특성 (실제 환경 시뮬레이션) - 5개 클래스
CLASS_FREQUENCY_RANGES = {
    'lidar': (0, 5),  # 거의 진동 없음
    'motor': (15, 35),  # 바람에 의한 저주파 진동
    'driving': (50, 80),  # 기계 진동 (고주파)
    'lidar_driving': (25, 45),  # 복합 환경 1
    'motor_driving': (40, 65)  # 복합 환경 2
}

# 라벨 매핑 (5개 클래스)
LABEL_MAP = {
    'lidar': 0,
    'motor': 1,
    'driving': 2,
    'lidar_driving': 3,
    'motor_driving': 4
}

# 역매핑
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 배터리 및 소비 전력 설정
BATTERY_CAPACITY = 100.0  # 배터리 용량 (mAh)
POWER_CONSUMPTION = 1.0  # 시간당 소비 전력 (mAh/hour)
SIMULATION_HOURS = 24 * 100  # 시뮬레이션 기간 (100일)

# 클래스 이름
CLASS_NAMES = ['lidar', 'motor', 'driving', 'lidar_driving', 'motor_driving']

print("=" * 60)
print("에너지 하베스팅 시뮬레이션 (5개 클래스)")
print("=" * 60)

print(f"\n📊 시뮬레이션 설정:")
print(f"  - 고정 공진 주파수: {FIXED_RESONANCE_FREQ} Hz")
print(f"  - 최대 출력: {MAX_POWER_OUTPUT} mW")
print(f"  - 품질 계수 Q: {QUALITY_FACTOR}")
print(f"  - 유효 대역폭: {BANDWIDTH:.2f} Hz")
print(f"  - 배터리 용량: {BATTERY_CAPACITY} mAh")
print(f"  - 시간당 소비: {POWER_CONSUMPTION} mAh/h")
print(f"  - 시뮬레이션 기간: {SIMULATION_HOURS // 24}일")
print(f"  - 클래스 수: {len(CLASS_NAMES)}개")


# ==================== 1. 물리 모델: 에너지 수확 효율 계산 ====================

def lorentzian_efficiency(f_input, f_resonance, Q=QUALITY_FACTOR):
    """
    로렌츠 함수 기반 에너지 수확 효율
    공진 주파수에서 최대, 멀어질수록 급격히 감소

    Args:
        f_input: 입력 진동 주파수 (Hz)
        f_resonance: 하베스터 공진 주파수 (Hz)
        Q: 품질 계수 (높을수록 대역폭 좁음)

    Returns:
        효율 (0~1)
    """
    gamma = f_resonance / (2 * Q)  # 반치전폭
    efficiency = 1 / (1 + ((f_input - f_resonance) / gamma) ** 2)
    return efficiency


def calculate_harvested_power(f_input, f_resonance):
    """
    수확된 전력 계산 (mW)
    """
    efficiency = lorentzian_efficiency(f_input, f_resonance)
    power = MAX_POWER_OUTPUT * efficiency
    return power


# ==================== 2. LSTM 모델 로드 ====================
print("\n[1단계] LSTM 모델 로드...")

try:
    # 최신 모델 로드 (과적합 방지 버전이 있으면 우선)
    try:
        model = keras.models.load_model('results_v2/best_model_v2.keras')
        data_file = 'processed_data_v2.pkl'
        print("✓ 개선 모델 로드: results_v2/best_model_v2.keras")
    except:
        model = keras.models.load_model('/Users/seohyeon/PycharmProjects/AT_data/results/best_model.keras')
        data_file = '/Users/seohyeon/PycharmProjects/AT_data/processed_data.pkl'
        print("✓ 기본 모델 로드: results/best_model_improved.keras")

    # 데이터 로드 (스케일러 등 필요)
    with open(data_file, 'rb') as f:
        data_dict = pickle.load(f)

    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    label_map = data_dict.get('label_map', LABEL_MAP)

    print(f"✓ Test 데이터: {X_test.shape[0]:,}개 시퀀스")
    print(f"✓ 클래스 수: {len(label_map)}개")

except Exception as e:
    print(f"⚠️  모델 로드 실패: {e}")
    print("먼저 lstm_train.py를 실행하세요!")
    exit(1)

# ==================== 3. AI 예측 수행 ====================
print("\n[2단계] AI 모델로 환경 분류 및 주파수 예측...")

# 예측
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred == y_true)
print(f"✓ 모델 정확도: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# 클래스별 정확도
print(f"\n✓ 클래스별 정확도:")
for class_id in range(len(CLASS_NAMES)):
    mask = y_true == class_id
    if mask.sum() > 0:
        class_acc = np.mean(y_pred[mask] == y_true[mask])
        print(f"  - {CLASS_NAMES[class_id]}: {class_acc:.4f} ({class_acc * 100:.2f}%)")


# 각 예측에 대해 최적 주파수 할당
def get_optimal_frequency(class_id):
    """클래스별 최적 공진 주파수"""
    class_name = REVERSE_LABEL_MAP[class_id]
    freq_range = CLASS_FREQUENCY_RANGES[class_name]
    # 범위의 중앙값
    return (freq_range[0] + freq_range[1]) / 2


# 실제 입력 주파수 생성 (실제 환경 시뮬레이션)
np.random.seed(42)
actual_frequencies = []
for true_class in y_true:
    class_name = REVERSE_LABEL_MAP[true_class]
    freq_range = CLASS_FREQUENCY_RANGES[class_name]
    # 범위 내에서 랜덤 주파수 생성
    freq = np.random.uniform(freq_range[0], freq_range[1])
    actual_frequencies.append(freq)

actual_frequencies = np.array(actual_frequencies)

# AI 예측 기반 공진 주파수
predicted_frequencies = np.array([get_optimal_frequency(pred) for pred in y_pred])

print(f"\n✓ 실제 진동 주파수 범위: {actual_frequencies.min():.1f} ~ {actual_frequencies.max():.1f} Hz")
print(f"✓ AI 예측 주파수 범위: {predicted_frequencies.min():.1f} ~ {predicted_frequencies.max():.1f} Hz")

# ==================== 4. 에너지 수확량 계산 ====================
print("\n[3단계] 에너지 수확량 계산...")

# 기존 방법: 고정 공진 주파수
fixed_powers = np.array([
    calculate_harvested_power(f, FIXED_RESONANCE_FREQ)
    for f in actual_frequencies
])

# 제안 방법: AI 기반 적응형 공진 주파수
adaptive_powers = np.array([
    calculate_harvested_power(actual_frequencies[i], predicted_frequencies[i])
    for i in range(len(actual_frequencies))
])

print(f"\n✓ 기존 방법 (고정 60Hz):")
print(f"  - 평균 수확 전력: {fixed_powers.mean():.4f} mW")
print(f"  - 최대 수확 전력: {fixed_powers.max():.4f} mW")
print(f"  - 최소 수확 전력: {fixed_powers.min():.4f} mW")

print(f"\n✓ 제안 방법 (AI 적응형):")
print(f"  - 평균 수확 전력: {adaptive_powers.mean():.4f} mW")
print(f"  - 최대 수확 전력: {adaptive_powers.max():.4f} mW")
print(f"  - 최소 수확 전력: {adaptive_powers.min():.4f} mW")

improvement = (adaptive_powers.mean() / fixed_powers.mean() - 1) * 100
print(f"\n📊 성능 개선: +{improvement:.2f}%")

# ==================== 5. 배터리 수명 시뮬레이션 ====================
print("\n[4단계] 배터리 수명 시뮬레이션...")


def simulate_battery_life(harvested_power_per_hour, consumption_per_hour,
                          initial_battery, max_battery, hours):
    """
    배터리 충방전 시뮬레이션
    """
    battery_levels = [initial_battery]

    for hour in range(1, hours):
        # 이전 배터리 잔량
        current_battery = battery_levels[-1]

        # 충전 (mW → mAh 변환: 1mW = 0.001mAh 가정)
        charged = harvested_power_per_hour * 0.001

        # 방전
        discharged = consumption_per_hour

        # 새 배터리 잔량
        new_battery = current_battery + charged - discharged

        # 배터리 용량 제한
        new_battery = max(0, min(new_battery, max_battery))

        battery_levels.append(new_battery)

        # 방전 시 시뮬레이션 종료
        if new_battery <= 0:
            break

    return np.array(battery_levels)


# 시뮬레이션 실행
hours = np.arange(SIMULATION_HOURS)

# 기존 방법
fixed_battery = simulate_battery_life(
    fixed_powers.mean(),
    POWER_CONSUMPTION,
    BATTERY_CAPACITY,
    BATTERY_CAPACITY,
    SIMULATION_HOURS
)

# 제안 방법
adaptive_battery = simulate_battery_life(
    adaptive_powers.mean(),
    POWER_CONSUMPTION,
    BATTERY_CAPACITY,
    BATTERY_CAPACITY,
    SIMULATION_HOURS
)

# 수명 계산
fixed_lifetime_hours = len(fixed_battery) - 1
adaptive_lifetime_hours = len(adaptive_battery) - 1

fixed_lifetime_days = fixed_lifetime_hours / 24
adaptive_lifetime_days = adaptive_lifetime_hours / 24

print(f"\n✓ 기존 방법 배터리 수명: {fixed_lifetime_days:.1f}일")
print(f"✓ 제안 방법 배터리 수명: {adaptive_lifetime_days:.1f}일")
print(f"✓ 수명 연장: {adaptive_lifetime_days - fixed_lifetime_days:.1f}일 (+{(adaptive_lifetime_days / fixed_lifetime_days - 1) * 100:.1f}%)")

# ==================== 6. 시각화 ====================
print("\n[5단계] 결과 시각화...")

import os

os.makedirs('results_simulation', exist_ok=True)

# 6-1. 주파수 응답 곡선 (물리 모델 시각화)
fig, ax = plt.subplots(figsize=(14, 8))

freq_range = np.linspace(0, 100, 1000)
efficiency_fixed = [lorentzian_efficiency(f, FIXED_RESONANCE_FREQ) for f in freq_range]

ax.plot(freq_range, efficiency_fixed, 'r-', linewidth=3, label='기존 방법 (고정 60Hz)')
ax.axvline(FIXED_RESONANCE_FREQ, color='r', linestyle='--', alpha=0.5)

# 각 클래스별 최적 주파수
colors = ['blue', 'green', 'orange', 'purple', 'cyan']
for i, (class_name, freq_range_class) in enumerate(CLASS_FREQUENCY_RANGES.items()):
    optimal_freq = (freq_range_class[0] + freq_range_class[1]) / 2
    efficiency_adaptive = [lorentzian_efficiency(f, optimal_freq) for f in freq_range]
    ax.plot(freq_range, efficiency_adaptive, color=colors[i], linewidth=2,
            linestyle='--', label=f'제안 방법 ({class_name}: {optimal_freq:.0f}Hz)', alpha=0.7)
    ax.axvline(optimal_freq, color=colors[i], linestyle=':', alpha=0.5)

ax.fill_between([FIXED_RESONANCE_FREQ - BANDWIDTH, FIXED_RESONANCE_FREQ + BANDWIDTH],
                0, 1, alpha=0.2, color='red', label=f'유효 대역폭 (±{BANDWIDTH:.1f}Hz)')

ax.set_xlabel('입력 진동 주파수 (Hz)', fontsize=13, fontweight='bold')
ax.set_ylabel('에너지 수확 효율', fontsize=13, fontweight='bold')
ax.set_title('하베스터 주파수 응답 특성 (로렌츠 모델) - 5개 클래스', fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('results_simulation/frequency_response.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_simulation/frequency_response.png")
plt.close()

# 6-2. 클래스별 평균 수확 전력 비교
fig, ax = plt.subplots(figsize=(14, 8))

class_fixed_powers = []
class_adaptive_powers = []

for class_id in range(len(CLASS_NAMES)):
    mask = y_true == class_id
    if mask.sum() > 0:
        class_fixed_powers.append(fixed_powers[mask].mean())
        class_adaptive_powers.append(adaptive_powers[mask].mean())
    else:
        class_fixed_powers.append(0)
        class_adaptive_powers.append(0)

x = np.arange(len(CLASS_NAMES))
width = 0.35

bars1 = ax.bar(x - width / 2, class_fixed_powers, width, label='기존 방법 (고정 60Hz)',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width / 2, class_adaptive_powers, width, label='제안 방법 (AI 적응형)',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}mW',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('환경 클래스', fontsize=13, fontweight='bold')
ax.set_ylabel('평균 수확 전력 (mW)', fontsize=13, fontweight='bold')
ax.set_title('클래스별 평균 에너지 수확량 비교 (5개 클래스)', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=11, rotation=15, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results_simulation/power_comparison_by_class.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_simulation/power_comparison_by_class.png")
plt.close()

# 6-3. 배터리 수명 시뮬레이션 (핵심 그래프)
fig, ax = plt.subplots(figsize=(14, 8))

hours_fixed = np.arange(len(fixed_battery))
hours_adaptive = np.arange(len(adaptive_battery))

ax.plot(hours_fixed / 24, fixed_battery, 'r-', linewidth=3,
        label=f'기존 방법 (수명: {fixed_lifetime_days:.1f}일)', alpha=0.8)
ax.plot(hours_adaptive / 24, adaptive_battery, 'g-', linewidth=3,
        label=f'제안 방법 (수명: {adaptive_lifetime_days:.1f}일)', alpha=0.8)

# 방전 시점 표시
ax.axvline(fixed_lifetime_days, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax.axvline(adaptive_lifetime_days, color='green', linestyle='--', alpha=0.5, linewidth=2)

# 연장된 수명 영역 표시
ax.fill_betweenx([0, BATTERY_CAPACITY], fixed_lifetime_days, adaptive_lifetime_days,
                 alpha=0.2, color='green', label=f'수명 연장: {adaptive_lifetime_days - fixed_lifetime_days:.1f}일')

ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(BATTERY_CAPACITY, color='black', linestyle='--', alpha=0.3)

ax.set_xlabel('시간 (일)', fontsize=13, fontweight='bold')
ax.set_ylabel('배터리 잔량 (mAh)', fontsize=13, fontweight='bold')
ax.set_title('배터리 수명 시뮬레이션 (100mAh, 1mAh/h 소비)', fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, min(SIMULATION_HOURS / 24, adaptive_lifetime_days * 1.1)])
ax.set_ylim([-5, BATTERY_CAPACITY + 10])

plt.tight_layout()
plt.savefig('results_simulation/battery_lifetime.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_simulation/battery_lifetime.png")
plt.close()

# 6-4. 전체 성능 요약 대시보드
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# (1) 평균 전력 비교
ax1 = fig.add_subplot(gs[0, 0])
methods = ['기존 방법\n(고정 60Hz)', '제안 방법\n(AI 적응형)']
powers = [fixed_powers.mean(), adaptive_powers.mean()]
bars = ax1.bar(methods, powers, color=['#FF6B6B', '#4ECDC4'], alpha=0.8,
               edgecolor='black', linewidth=2)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.4f}mW',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('평균 수확 전력 (mW)', fontsize=12, fontweight='bold')
ax1.set_title('평균 에너지 수확량', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# (2) 배터리 수명 비교
ax2 = fig.add_subplot(gs[0, 1])
lifetimes = [fixed_lifetime_days, adaptive_lifetime_days]
bars = ax2.bar(methods, lifetimes, color=['#FF6B6B', '#4ECDC4'], alpha=0.8,
               edgecolor='black', linewidth=2)
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height,
             f'{height:.1f}일',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('배터리 수명 (일)', fontsize=12, fontweight='bold')
ax2.set_title('센서 노드 수명', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# (3) 효율 분포
ax3 = fig.add_subplot(gs[1, :])
fixed_efficiency = fixed_powers / MAX_POWER_OUTPUT * 100
adaptive_efficiency = adaptive_powers / MAX_POWER_OUTPUT * 100

bins = np.linspace(0, 100, 50)
ax3.hist(fixed_efficiency, bins=bins, alpha=0.6, color='red',
         label=f'기존 (평균: {fixed_efficiency.mean():.1f}%)', edgecolor='black')
ax3.hist(adaptive_efficiency, bins=bins, alpha=0.6, color='green',
         label=f'제안 (평균: {adaptive_efficiency.mean():.1f}%)', edgecolor='black')
ax3.axvline(fixed_efficiency.mean(), color='red', linestyle='--', linewidth=2)
ax3.axvline(adaptive_efficiency.mean(), color='green', linestyle='--', linewidth=2)
ax3.set_xlabel('에너지 수확 효율 (%)', fontsize=12, fontweight='bold')
ax3.set_ylabel('샘플 수', fontsize=12, fontweight='bold')
ax3.set_title('에너지 수확 효율 분포', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)

plt.suptitle('에너지 하베스팅 시스템 성능 요약 (5개 클래스)', fontsize=16, fontweight='bold', y=0.98)
plt.savefig('results_simulation/performance_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_simulation/performance_dashboard.png")
plt.close()

# ==================== 7. 결과 요약 테이블 ====================
print("\n[6단계] 결과 요약...")

summary_df = pd.DataFrame({
    '항목': [
        '평균 수확 전력 (mW)',
        '최대 수확 전력 (mW)',
        '평균 효율 (%)',
        '배터리 수명 (일)',
        '센서 교체 주기 연장 (일)'
    ],
    '기존 방법 (고정 60Hz)': [
        f"{fixed_powers.mean():.4f}",
        f"{fixed_powers.max():.4f}",
        f"{(fixed_powers.mean() / MAX_POWER_OUTPUT * 100):.2f}",
        f"{fixed_lifetime_days:.1f}",
        "-"
    ],
    '제안 방법 (AI 적응형)': [
        f"{adaptive_powers.mean():.4f}",
        f"{adaptive_powers.max():.4f}",
        f"{(adaptive_powers.mean() / MAX_POWER_OUTPUT * 100):.2f}",
        f"{adaptive_lifetime_days:.1f}",
        f"+{adaptive_lifetime_days - fixed_lifetime_days:.1f}"
    ],
    '개선율 (%)': [
        f"+{improvement:.2f}",
        f"+{(adaptive_powers.max() / fixed_powers.max() - 1) * 100:.2f}",
        f"+{((adaptive_powers.mean() / MAX_POWER_OUTPUT) / (fixed_powers.mean() / MAX_POWER_OUTPUT) - 1) * 100:.2f}",
        f"+{(adaptive_lifetime_days / fixed_lifetime_days - 1) * 100:.1f}",
        "-"
    ]
})

print("\n" + "=" * 100)
print("📊 최종 성능 비교 (5개 클래스)")
print("=" * 100)
print(summary_df.to_string(index=False))

summary_df.to_csv('results_simulation/performance_summary.csv', index=False, encoding='utf-8-sig')
print("\n✓ 저장: results_simulation/performance_summary.csv")

# 클래스별 상세 결과
class_summary = []
for class_id in range(len(CLASS_NAMES)):
    mask = y_true == class_id
    if mask.sum() > 0:
        class_summary.append({
            '클래스': CLASS_NAMES[class_id],
            '샘플 수': mask.sum(),
            '기존 방법 (mW)': f"{fixed_powers[mask].mean():.4f}",
            '제안 방법 (mW)': f"{adaptive_powers[mask].mean():.4f}",
            '개선율 (%)': f"+{(adaptive_powers[mask].mean() / fixed_powers[mask].mean() - 1) * 100:.2f}"
        })

class_summary_df = pd.DataFrame(class_summary)
print("\n" + "=" * 100)
print("📊 클래스별 상세 성능")
print("=" * 100)
print(class_summary_df.to_string(index=False))

class_summary_df.to_csv('results_simulation/class_performance_summary.csv', index=False, encoding='utf-8-sig')
print("\n✓ 저장: results_simulation/class_performance_summary.csv")

# ==================== 8. 최종 요약 ====================
print("\n" + "=" * 100)
print("✅ 시뮬레이션 완료!")
print("=" * 100)

print(f"\n🎯 핵심 결과:")
print(f"  1. 에너지 수확 효율 {improvement:.2f}% 향상")
print(f"  2. 배터리 수명 {adaptive_lifetime_days - fixed_lifetime_days:.1f}일 연장 (+{(adaptive_lifetime_days / fixed_lifetime_days - 1) * 100:.1f}%)")
print(f"  3. 센서 교체 주기 대폭 증가 → 유지보수 비용 절감")
print(f"  4. {len(CLASS_NAMES)}개 클래스 환경 대응")

print(f"\n💾 생성된 파일:")
print(f"  - results_simulation/frequency_response.png (주파수 응답 특성)")
print(f"  - results_simulation/power_comparison_by_class.png (클래스별 전력 비교)")
print(f"  - results_simulation/battery_lifetime.png (배터리 수명 시뮬레이션)")
print(f"  - results_simulation/performance_dashboard.png (성능 요약 대시보드)")
print(f"  - results_simulation/performance_summary.csv (수치 데이터)")
print(f"  - results_simulation/class_performance_summary.csv (클래스별 상세)")

print(f"\n📝 보고서 작성 팁:")
print(f"  1. 물리 모델: 로렌츠 함수 기반 공진 현상 구현")
print(f"  2. 실험 설정: 일반적인 압전 하베스터 특성 반영 (Q={QUALITY_FACTOR})")
print(f"  3. 실용성: 배터리 수명 연장으로 IoT 센서 유지보수 비용 {(adaptive_lifetime_days / fixed_lifetime_days - 1) * 100:.1f}% 절감")
print(f"  4. 확장성: 실제 하드웨어 구현 시에도 동일한 원리 적용 가능")
print(f"  5. 다양성: {len(CLASS_NAMES)}개 복합 환경 대응 (lidar, motor, driving, lidar_driving, motor_driving)")