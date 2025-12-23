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

    Parameters:
    -----------
    results_csv_path : str
        FFT 분석 결과 CSV 파일 경로
    output_dir : str
        결과 저장 디렉토리 (None이면 CSV와 같은 위치)
    """
    # 결과 파일 읽기
    df = pd.read_csv(results_csv_path)

    if output_dir is None:
        output_dir = Path(results_csv_path).parent
    else:
        output_dir = Path(output_dir)

    # 클래스와 축 목록
    classes = df['Class'].unique()
    axes = df['Axis'].unique()

    print("=" * 70)
    print("주파수 오차 분포 분석")
    print("=" * 70)

    # 1. 클래스별, 축별 통계
    print("\n[1] 클래스별, 축별 공진 주파수 통계")
    print("-" * 70)

    stats_list = []
    for class_name in classes:
        for axis in axes:
            data = df[(df['Class'] == class_name) & (df['Axis'] == axis)]

            if len(data) > 0:
                freqs = data['Resonance_Frequency_Hz'].values
                stats = {
                    'Class': class_name,
                    'Axis': axis,
                    'Count': len(freqs),
                    'Mean': np.mean(freqs),
                    'Std': np.std(freqs),
                    'Min': np.min(freqs),
                    'Max': np.max(freqs),
                    'Range': np.max(freqs) - np.min(freqs),
                    'CV(%)': (np.std(freqs) / np.mean(freqs)) * 100  # 변동계수
                }
                stats_list.append(stats)

                print(f"\n{class_name.upper()} - {axis}")
                print(f"  샘플 수: {stats['Count']}")
                print(f"  평균: {stats['Mean']:.2f} Hz")
                print(f"  표준편차: {stats['Std']:.2f} Hz")
                print(f"  범위: {stats['Min']:.2f} ~ {stats['Max']:.2f} Hz")
                print(f"  변동계수: {stats['CV(%)']:.2f}%")

    stats_df = pd.DataFrame(stats_list)
    stats_output = output_dir / "frequency_statistics.csv"
    stats_df.to_csv(stats_output, index=False, encoding='utf-8-sig')
    print(f"\n✓ 통계 저장: {stats_output}")

    # 2. 클래스별 히스토그램 (축별로 서브플롯)
    print("\n[2] 클래스별 주파수 분포 히스토그램 생성 중...")

    for class_name in classes:
        class_data = df[df['Class'] == class_name]

        if len(class_data) == 0:
            continue

        fig, axes_plot = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'공진 주파수 분포 - {class_name.upper()}', fontsize=16, fontweight='bold')

        for idx, axis in enumerate(axes):
            row = idx // 3
            col = idx % 3
            ax = axes_plot[row, col]

            axis_data = class_data[class_data['Axis'] == axis]['Resonance_Frequency_Hz'].values

            if len(axis_data) > 0:
                # 히스토그램
                ax.hist(axis_data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')

                # 통계 정보 표시
                mean_val = np.mean(axis_data)
                std_val = np.std(axis_data)

                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'평균: {mean_val:.2f} Hz')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                           label=f'±1σ: {std_val:.2f} Hz')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)

                ax.set_title(f'{axis}\n(n={len(axis_data)})', fontweight='bold')
                ax.set_xlabel('주파수 (Hz)')
                ax.set_ylabel('빈도')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_title(axis)

        plt.tight_layout()
        hist_output = output_dir / f"frequency_distribution_{class_name}.png"
        plt.savefig(hist_output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {class_name}: {hist_output}")

    # 3. 축별 박스플롯 (클래스 비교) - tick_labels로 수정
    print("\n[3] 축별 클래스 비교 박스플롯 생성 중...")

    fig, axes_plot = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('축별 공진 주파수 비교 (클래스별)', fontsize=16, fontweight='bold')

    for idx, axis in enumerate(axes):
        row = idx // 3
        col = idx % 3
        ax = axes_plot[row, col]

        # 클래스별 데이터 준비
        data_for_box = []
        labels_for_box = []

        for class_name in classes:
            axis_data = df[(df['Class'] == class_name) & (df['Axis'] == axis)]['Resonance_Frequency_Hz'].values
            if len(axis_data) > 0:
                data_for_box.append(axis_data)
                labels_for_box.append(class_name)

        if len(data_for_box) > 0:
            bp = ax.boxplot(data_for_box, tick_labels=labels_for_box, patch_artist=True)

            # 박스 색상 설정
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_title(f'{axis}', fontweight='bold', fontsize=12)
            ax.set_ylabel('주파수 (Hz)')
            ax.set_xlabel('클래스')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(axis)

    plt.tight_layout()
    boxplot_output = output_dir / "frequency_comparison_boxplot.png"
    plt.savefig(boxplot_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {boxplot_output}")

    # 4. 전체 바이올린 플롯
    print("\n[4] 전체 바이올린 플롯 생성 중...")

    fig, ax = plt.subplots(figsize=(16, 8))

    # 데이터 재구성
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({
            'Class_Axis': f"{row['Class']}\n{row['Axis']}",
            'Frequency': row['Resonance_Frequency_Hz'],
            'Class': row['Class'],
            'Axis': row['Axis']
        })

    plot_df = pd.DataFrame(plot_data)

    # 바이올린 플롯
    sns.violinplot(data=plot_df, x='Axis', y='Frequency', hue='Class', ax=ax, split=False)

    ax.set_title('축별 주파수 분포 비교 (모든 클래스)', fontsize=16, fontweight='bold')
    ax.set_xlabel('축', fontsize=12)
    ax.set_ylabel('주파수 (Hz)', fontsize=12)
    ax.legend(title='클래스', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    violin_output = output_dir / "frequency_violin_plot.png"
    plt.savefig(violin_output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {violin_output}")

    # 5. 오차 범위 요약
    print("\n[5] 오차 범위 요약")
    print("-" * 70)

    for axis in axes:
        print(f"\n{axis}:")
        axis_data = df[df['Axis'] == axis]

        for class_name in classes:
            class_axis_data = axis_data[axis_data['Class'] == class_name]['Resonance_Frequency_Hz'].values

            if len(class_axis_data) > 0:
                mean_val = np.mean(class_axis_data)
                std_val = np.std(class_axis_data)

                # 68% 신뢰구간 (±1σ)
                ci_68_lower = mean_val - std_val
                ci_68_upper = mean_val + std_val

                # 95% 신뢰구간 (±2σ)
                ci_95_lower = mean_val - 2 * std_val
                ci_95_upper = mean_val + 2 * std_val

                print(f"  {class_name}:")
                print(f"    평균: {mean_val:.2f} Hz")
                print(f"    68% 신뢰구간: [{ci_68_lower:.2f}, {ci_68_upper:.2f}] Hz")
                print(f"    95% 신뢰구간: [{ci_95_lower:.2f}, {ci_95_upper:.2f}] Hz")

    print("\n" + "=" * 70)
    print("기본 분석 완료! 이제 통계 CSV 시각화를 시작합니다...")
    print("=" * 70)

    # 6. 통계 CSV 시각화 추가
    visualize_statistics_csv(stats_df, output_dir)

    return stats_df


def visualize_statistics_csv(stats_df, output_dir):
    """
    frequency_statistics DataFrame을 다양한 플롯으로 시각화합니다.

    Parameters:
    -----------
    stats_df : DataFrame
        통계 데이터프레임
    output_dir : Path
        결과 저장 디렉토리
    """
    print("\n" + "=" * 70)
    print("통계 CSV 시각화")
    print("=" * 70)

    classes = stats_df['Class'].unique()
    axes = stats_df['Axis'].unique()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    # 6-1. 평균 주파수 히트맵
    print("\n[6-1] 평균 주파수 히트맵 생성 중...")

    pivot_mean = stats_df.pivot(index='Axis', columns='Class', values='Mean')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=0.5, ax=ax, cbar_kws={'label': '평균 주파수 (Hz)'})
    ax.set_title('클래스별-축별 평균 공진 주파수 히트맵', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('클래스', fontsize=12, fontweight='bold')
    ax.set_ylabel('축', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "stats_heatmap_mean.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-2. 표준편차 히트맵
    print("\n[6-2] 표준편차 히트맵 생성 중...")

    pivot_std = stats_df.pivot(index='Axis', columns='Class', values='Std')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='Blues',
                linewidths=0.5, ax=ax, cbar_kws={'label': '표준편차 (Hz)'})
    ax.set_title('클래스별-축별 표준편차 히트맵 (오차 크기)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('클래스', fontsize=12, fontweight='bold')
    ax.set_ylabel('축', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "stats_heatmap_std.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-3. 변동계수(CV) 히트맵
    print("\n[6-3] 변동계수(CV) 히트맵 생성 중...")

    pivot_cv = stats_df.pivot(index='Axis', columns='Class', values='CV(%)')

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_cv, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': '변동계수 (%)'})
    ax.set_title('클래스별-축별 변동계수 히트맵 (상대적 오차)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('클래스', fontsize=12, fontweight='bold')
    ax.set_ylabel('축', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / "stats_heatmap_cv.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-4. 축별 평균 주파수 비교 막대그래프
    print("\n[6-4] 축별 평균 주파수 막대그래프 생성 중...")

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(axes))
    width = 0.15

    for idx, class_name in enumerate(classes):
        class_data = stats_df[stats_df['Class'] == class_name]
        means = [class_data[class_data['Axis'] == axis]['Mean'].values[0]
                 if len(class_data[class_data['Axis'] == axis]) > 0 else 0
                 for axis in axes]
        stds = [class_data[class_data['Axis'] == axis]['Std'].values[0]
                if len(class_data[class_data['Axis'] == axis]) > 0 else 0
                for axis in axes]

        offset = width * (idx - len(classes) / 2 + 0.5)
        ax.bar(x + offset, means, width, label=class_name,
               yerr=stds, capsize=5, alpha=0.8, color=colors[idx % len(colors)])

    ax.set_xlabel('축', fontsize=12, fontweight='bold')
    ax.set_ylabel('평균 주파수 (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('클래스별 축 평균 공진 주파수 비교 (±표준편차)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.legend(title='클래스', loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "stats_bar_mean_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-5. 클래스별 표준편차 비교 막대그래프
    print("\n[6-5] 클래스별 표준편차 비교 막대그래프 생성 중...")

    fig, ax = plt.subplots(figsize=(14, 8))

    for idx, class_name in enumerate(classes):
        class_data = stats_df[stats_df['Class'] == class_name]
        stds = [class_data[class_data['Axis'] == axis]['Std'].values[0]
                if len(class_data[class_data['Axis'] == axis]) > 0 else 0
                for axis in axes]

        offset = width * (idx - len(classes) / 2 + 0.5)
        ax.bar(x + offset, stds, width, label=class_name,
               alpha=0.8, color=colors[idx % len(colors)])

    ax.set_xlabel('축', fontsize=12, fontweight='bold')
    ax.set_ylabel('표준편차 (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('클래스별 축 표준편차 비교 (오차 크기)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.legend(title='클래스', loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "stats_bar_std_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-6. Range(범위) 비교
    print("\n[6-6] 주파수 범위(Range) 비교 생성 중...")

    fig, ax = plt.subplots(figsize=(14, 8))

    for idx, class_name in enumerate(classes):
        class_data = stats_df[stats_df['Class'] == class_name]
        ranges = [class_data[class_data['Axis'] == axis]['Range'].values[0]
                  if len(class_data[class_data['Axis'] == axis]) > 0 else 0
                  for axis in axes]

        offset = width * (idx - len(classes) / 2 + 0.5)
        ax.bar(x + offset, ranges, width, label=class_name,
               alpha=0.8, color=colors[idx % len(colors)])

    ax.set_xlabel('축', fontsize=12, fontweight='bold')
    ax.set_ylabel('주파수 범위 (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('클래스별 축 주파수 범위 (Max - Min)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(axes)
    ax.legend(title='클래스', loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "stats_bar_range_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-7. 평균 vs 표준편차 산점도
    print("\n[6-7] 평균 vs 표준편차 산점도 생성 중...")

    fig, ax = plt.subplots(figsize=(12, 8))

    for idx, class_name in enumerate(classes):
        class_data = stats_df[stats_df['Class'] == class_name]
        ax.scatter(class_data['Mean'], class_data['Std'],
                   s=150, alpha=0.6, label=class_name,
                   color=colors[idx % len(colors)])

        # 축 이름 표시
        for _, row in class_data.iterrows():
            ax.annotate(row['Axis'], (row['Mean'], row['Std']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)

    ax.set_xlabel('평균 주파수 (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('표준편차 (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('평균 주파수 vs 표준편차 관계', fontsize=16, fontweight='bold')
    ax.legend(title='클래스')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "stats_scatter_mean_vs_std.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    # 6-8. 축별 요약 플롯 (평균, 표준편차, CV 한번에)
    print("\n[6-8] 축별 종합 요약 플롯 생성 중...")

    fig, axes_plot = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('축별 통계 종합 요약', fontsize=16, fontweight='bold')

    for idx, axis in enumerate(axes):
        row = idx // 3
        col = idx % 3
        ax = axes_plot[row, col]

        axis_data = stats_df[stats_df['Axis'] == axis]

        x_pos = np.arange(len(classes))

        # 평균 막대
        means = axis_data['Mean'].values
        stds = axis_data['Std'].values

        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                      alpha=0.7, color=colors[:len(classes)])

        # CV 값을 막대 위에 표시
        for i, (bar, cv) in enumerate(zip(bars, axis_data['CV(%)'].values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + stds[i],
                    f'CV: {cv:.1f}%', ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{axis}', fontweight='bold', fontsize=12)
        ax.set_ylabel('평균 주파수 (Hz)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(axis_data['Class'].values, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / "stats_summary_by_axis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {output_path}")

    print("\n" + "=" * 70)
    print("모든 시각화 완료!")
    print("=" * 70)


# 실행
if __name__ == "__main__":
    # FFT 분석 결과 CSV 파일 경로
    RESULTS_CSV = "/Users/seohyeon/PycharmProjects/AT_data/data_v1/all_files_fft_analysis_results.csv"

    # 분석 실행 (통계 CSV 시각화 포함)
    stats = analyze_frequency_distribution(RESULTS_CSV)

    print("\n[통계 요약]")
    print(stats.to_string(index=False))