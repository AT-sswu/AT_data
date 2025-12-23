import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 한글 폰트 설정 (Mac)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def analyze_frequency_distribution(results_csv_path, output_dir=None):
    """
    FFT 분석 결과에서 주파수 오차 분포를 분석합니다.
    (모든 파일을 'base' 클래스로 통일)
    """
    # 1. 파일 읽기 및 경로 설정
    if not Path(results_csv_path).exists():
        print(f"Error: 파일을 찾을 수 없습니다 -> {results_csv_path}")
        return None

    df = pd.read_csv(results_csv_path)

    # [변경 사항] 모든 데이터를 'base' 클래스로 통일
    df['Class'] = 'base'

    if output_dir is None:
        output_dir = Path(results_csv_path).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    axes = df['Axis'].unique()

    print("=" * 70)
    print(f"주파수 오차 및 통계 분석 (Base 모드)")
    print("=" * 70)

    # 2. 축별 통계 계산
    stats_list = []
    for axis in axes:
        data = df[df['Axis'] == axis]['Resonance_Frequency_Hz'].values
        if len(data) > 0:
            stats = {
                'Class': 'base',
                'Axis': axis,
                'Count': len(data),
                'Mean': np.mean(data),
                'Std': np.std(data),
                'Min': np.min(data),
                'Max': np.max(data),
                'Range': np.max(data) - np.min(data),
                'CV(%)': (np.std(data) / np.mean(data)) * 100
            }
            stats_list.append(stats)

    stats_df = pd.DataFrame(stats_list)
    stats_output = output_dir / "frequency_statistics_base.csv"
    stats_df.to_csv(stats_output, index=False, encoding='utf-8-sig')
    print(f"✓ 통계 CSV 저장 완료: {stats_output}")

    # 3. 히트맵 시각화 (Mean, Std, CV%)
    print("\n[2] 히트맵 시각화 생성 중...")

    # 히트맵을 위한 피벗 (Index: Axis, Columns: Class('base'))
    pivot_mean = stats_df.pivot(index='Axis', columns='Class', values='Mean')
    pivot_std = stats_df.pivot(index='Axis', columns='Class', values='Std')
    pivot_cv = stats_df.pivot(index='Axis', columns='Class', values='CV(%)')

    fig, axes_heat = plt.subplots(1, 3, figsize=(18, 6))

    # (1) 평균 주파수 히트맵
    sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes_heat[0])
    axes_heat[0].set_title('평균 주파수 (Hz)')

    # (2) 표준편차(절대 오차) 히트맵
    sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='Blues', ax=axes_heat[1])
    axes_heat[1].set_title('표준편차 (오차 크기)')

    # (3) 변동계수(상대 오차) 히트맵
    sns.heatmap(pivot_cv, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes_heat[2])
    axes_heat[2].set_title('변동계수 (%)')

    plt.suptitle('축별 공진 주파수 통계 히트맵 (Base)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    heatmap_output = output_dir / "stats_heatmaps_base.png"
    plt.savefig(heatmap_output, dpi=150)
    plt.close()
    print(f"  ✓ 히트맵 저장: {heatmap_output}")

    # 4. 히스토그램 (축별 분포)
    print("\n[3] 주파수 분포 히스토그램 생성 중...")
    fig, axes_plot = plt.subplots(2, 3, figsize=(18, 10))
    for idx, axis in enumerate(axes):
        row, col = idx // 3, idx % 3
        ax = axes_plot[row, col]
        axis_data = df[df['Axis'] == axis]['Resonance_Frequency_Hz'].values
        if len(axis_data) > 0:
            ax.hist(axis_data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
            mean_val = np.mean(axis_data)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.set_title(f'{axis}')
            ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "frequency_distribution_base.png", dpi=150)
    plt.close()

    # 5. 막대 그래프 (평균 및 범위 비교)
    print("\n[4] 통계 막대 그래프 생성 중...")
    plt.figure(figsize=(12, 6))
    plt.bar(stats_df['Axis'], stats_df['Mean'], yerr=stats_df['Std'], capsize=5, color='lightgray', edgecolor='black')
    plt.title('축별 평균 공진 주파수 및 표준편차 (Base)')
    plt.savefig(output_dir / "stats_bar_comparison_base.png", dpi=150)
    plt.close()

    print("\n" + "=" * 70)
    print("모든 분석 및 시각화 완료!")
    print("=" * 70)

    return stats_df


# --- 실행부 ---
if __name__ == "__main__":
    # 요청하신 data_v2 경로 및 파일명
    RESULTS_CSV = "/Users/seohyeon/PycharmProjects/AT_data/data_v2/fft_analysis_base_results.csv"

    stats = analyze_frequency_distribution(RESULTS_CSV)
    if stats is not None:
        print(stats.to_string(index=False))