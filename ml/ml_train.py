"""
전통적 머신러닝 분류기 비교
- Random Forest
- SVM
- XGBoost
- KNN
- Decision Tree
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import pandas as pd
import os
import time

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 5개 클래스
CLASS_NAMES = ['lidar', 'motor', 'driving', 'lidar_driving', 'motor_driving']

print("=" * 70)
print("전통적 머신러닝 분류기 성능 비교")
print("=" * 70)

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] 데이터 로드...")

try:
    with open('processed_data_traditional_ml.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    feature_names = data_dict['feature_names']

    # Train + Val 합치기 (전통적 ML은 Early Stopping 불필요)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])

    print(f"✓ 데이터 로드 완료")
    print(f"  - Train: {len(X_train_full):,}개")
    print(f"  - Test: {len(X_test):,}개")
    print(f"  - 특징 수: {len(feature_names)}개")

except FileNotFoundError:
    print("️  processed_data_traditional_ml.pkl을 찾을 수 없습니다.")
    print("먼저 traditional_ml_data_prep.py를 실행하세요!")
    exit(1)

os.makedirs('results_traditional_ml', exist_ok=True)

# ==================== 2. 분류기 정의 ====================
print("\n[2단계] 분류기 정의...")

classifiers = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        random_state=42
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=5,
        n_jobs=-1
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=20,
        random_state=42
    )
}

print(f"✓ {len(classifiers)}개 분류기 준비 완료")

# ==================== 3. 모델 학습 및 평가 ====================
print("\n[3단계] 모델 학습 및 평가...")

results = {}

for name, clf in classifiers.items():
    print(f"\n{'=' * 60}")
    print(f"[{name}] 학습 중...")

    # 학습 시작
    start_time = time.time()
    clf.fit(X_train_full, y_train_full)
    train_time = time.time() - start_time

    # 예측
    start_time = time.time()
    y_pred = clf.predict(X_test)
    predict_time = time.time() - start_time

    # 성능 지표
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    print(f"✓ 학습 완료!")
    print(f"  - 학습 시간: {train_time:.2f}초")
    print(f"  - 예측 시간: {predict_time:.4f}초")
    print(f"  - Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}")

    # 결과 저장
    results[name] = {
        'model': clf,
        'y_pred': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'predict_time': predict_time
    }

# ==================== 4. 성능 비교 시각화 ====================
print("\n[4단계] 성능 비교 시각화...")

# 4-1. 성능 지표 비교
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]

    model_names = list(results.keys())
    scores = [results[name][metric] for name in model_names]

    bars = ax.barh(model_names, scores, color='steelblue', alpha=0.8, edgecolor='black')

    # 값 표시
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{score:.4f}',
                va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} 비교', fontsize=13, fontweight='bold')
    ax.set_xlim([0, 1.1])
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results_traditional_ml/performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_traditional_ml/performance_comparison.png")
plt.close()

# 4-2. 학습/예측 시간 비교
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(results.keys())
train_times = [results[name]['train_time'] for name in model_names]
predict_times = [results[name]['predict_time'] * 1000 for name in model_names]  # ms

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width / 2, train_times, width, label='학습 시간 (초)',
               color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width / 2, predict_times, width, label='예측 시간 (ms)',
               color='#4ECDC4', alpha=0.8, edgecolor='black')

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('모델', fontsize=12, fontweight='bold')
ax.set_ylabel('시간', fontsize=12, fontweight='bold')
ax.set_title('모델별 학습/예측 시간 비교', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results_traditional_ml/time_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_traditional_ml/time_comparison.png")
plt.close()

# 4-3. 최고 성능 모델의 혼동 행렬
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model_pred = results[best_model_name]['y_pred']

cm = confusion_matrix(y_test, best_model_pred)

# 5x5 혼동 행렬
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': '샘플 수'},
            annot_kws={'size': 11})
plt.xlabel('예측 클래스', fontsize=13, fontweight='bold')
plt.ylabel('실제 클래스', fontsize=13, fontweight='bold')
plt.title(f'{best_model_name} 혼동 행렬\nAccuracy: {results[best_model_name]["accuracy"]:.4f}',
          fontsize=15, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results_traditional_ml/confusion_matrix_best.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_traditional_ml/confusion_matrix_best.png")
plt.close()

# ==================== 5. 특징 중요도 분석 (Random Forest) ====================
print("\n[5단계] 특징 중요도 분석 (Random Forest)...")

if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_

    # 상위 20개 특징
    top_indices = np.argsort(feature_importance)[::-1][:20]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(top_features)), top_importance, color='forestgreen', alpha=0.8, edgecolor='black')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel('중요도', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest 특징 중요도 (상위 20개)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_traditional_ml/feature_importance_rf.png', dpi=300, bbox_inches='tight')
    print("✓ 저장: results_traditional_ml/feature_importance_rf.png")
    plt.close()

# ==================== 6. 클래스별 성능 분석 ====================
print("\n[6단계] 클래스별 성능 분석...")

print("\n" + "=" * 70)
print(f"[{best_model_name}] 클래스별 세부 성능")
print("=" * 70)

print(classification_report(y_test, best_model_pred, target_names=CLASS_NAMES, digits=4))

# 클래스별 정확도 시각화
class_accuracies = []
for class_id, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_test == class_id)[0]
    if len(class_indices) > 0:
        class_accuracy = np.mean(best_model_pred[class_indices] == class_id)
        class_accuracies.append(class_accuracy)
    else:
        class_accuracies.append(0)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(CLASS_NAMES, class_accuracies, color='teal', alpha=0.8, edgecolor='black')

# 값 표시
for bar, acc in zip(bars, class_accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{acc:.4f}\n({acc * 100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('클래스', fontsize=12, fontweight='bold')
ax.set_ylabel('정확도', fontsize=12, fontweight='bold')
ax.set_title(f'{best_model_name} - 클래스별 정확도', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('results_traditional_ml/class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_traditional_ml/class_accuracy.png")
plt.close()

# ==================== 7. 결과 테이블 생성 ====================
print("\n[7단계] 결과 요약 테이블...")

comparison_df = pd.DataFrame({
    '모델': list(results.keys()),
    'Accuracy': [f"{results[name]['accuracy']:.4f}" for name in results.keys()],
    'Precision': [f"{results[name]['precision']:.4f}" for name in results.keys()],
    'Recall': [f"{results[name]['recall']:.4f}" for name in results.keys()],
    'F1-Score': [f"{results[name]['f1']:.4f}" for name in results.keys()],
    '학습 시간(초)': [f"{results[name]['train_time']:.2f}" for name in results.keys()],
    '예측 시간(ms)': [f"{results[name]['predict_time'] * 1000:.2f}" for name in results.keys()]
})

print("\n" + "=" * 100)
print("📊 전통적 머신러닝 모델 성능 비교")
print("=" * 100)
print(comparison_df.to_string(index=False))

comparison_df.to_csv('results_traditional_ml/model_comparison.csv', index=False, encoding='utf-8-sig')
print("\n✓ 저장: results_traditional_ml/model_comparison.csv")

# ==================== 8. 최고 모델 저장 ====================
print("\n[8단계] 최고 성능 모델 저장...")

best_model = results[best_model_name]['model']

with open('results_traditional_ml/best_traditional_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'model_name': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'feature_names': feature_names,
        'class_names': CLASS_NAMES
    }, f)

print(f"✓ 저장: results_traditional_ml/best_traditional_model.pkl")
print(f"  - 최고 모델: {best_model_name}")
print(f"  - Accuracy: {results[best_model_name]['accuracy']:.4f}")

# ==================== 9. 최종 요약 ====================
print("\n" + "=" * 70)
print("전통적 머신러닝 분류기 비교 완료!")
print("=" * 70)

print(f"\n 최고 성능 모델: {best_model_name}")
print(f"  - Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"  - F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"  - 학습 시간: {results[best_model_name]['train_time']:.2f}초")
print(f"  - 예측 시간: {results[best_model_name]['predict_time'] * 1000:.2f}ms")

print(f"\n 생성된 파일:")
print(f"  - results_traditional_ml/performance_comparison.png")
print(f"  - results_traditional_ml/time_comparison.png")
print(f"  - results_traditional_ml/confusion_matrix_best.png")
print(f"  - results_traditional_ml/feature_importance_rf.png")
print(f"  - results_traditional_ml/class_accuracy.png")
print(f"  - results_traditional_ml/model_comparison.csv")
print(f"  - results_traditional_ml/best_traditional_model.pkl")

print(f"\n 클래스 정보:")
print(f"  - 총 {len(CLASS_NAMES)}개 클래스")
print(f"  - {', '.join(CLASS_NAMES)}")
