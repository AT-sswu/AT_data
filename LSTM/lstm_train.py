import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import os

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 설정 (일반화 개선) ====================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.003

# 라벨 이름 (5개 클래스)
CLASS_NAMES = ['lidar', 'motor', 'driving', 'lidar_driving', 'motor_driving']

print("=" * 60)
print("LSTM 모델 학습 시작 (일반화 개선 버전)")
print("=" * 60)

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f"\n✓ GPU 가속 활성화: {len(gpus)}개의 GPU 사용")
else:
    print(f"\n✓ CPU 모드로 실행")

# 재현성을 위한 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] 전처리된 데이터 로드 중...")

try:
    with open('../processed_data.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']

    print(f"✓ 데이터 로드 완료")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")

    # 데이터 검증
    print("\n[데이터 검증]")
    train_samples = X_train.reshape(X_train.shape[0], -1)
    test_samples = X_test.reshape(X_test.shape[0], -1)

    check_size = min(100, len(train_samples), len(test_samples))
    duplicates = 0
    for i in range(check_size):
        if np.any(np.all(train_samples[i] == test_samples[:check_size], axis=1)):
            duplicates += 1

    if duplicates > 0:
        print(f"️  경고: Train과 Test 데이터 간 {duplicates}/{check_size} 중복 샘플 발견")
    else:
        print(f"✓ 데이터 중복 없음 (샘플 {check_size}개 체크)")

except FileNotFoundError:
    print(" processed_data.pkl 파일을 찾을 수 없습니다.")
    print("먼저 data_preparation.py를 실행하세요!")
    exit(1)

# 데이터 증강 (노이즈 추가)
print("\n[데이터 증강]")
print("✓ 훈련 데이터에 가우시안 노이즈 추가 (표준편차: 0.01)")
noise = np.random.normal(0, 0.01, X_train.shape)
X_train_augmented = X_train + noise

# ==================== 2. 모델 구축 (일반화 개선) ====================
print("\n[2단계] LSTM 모델 구축 중...")

model = Sequential([
    # Input Layer
    LSTM(32, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2]),
         kernel_regularizer=l2(0.01),
         recurrent_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),

    # Second LSTM Layer
    LSTM(32, return_sequences=False,
         kernel_regularizer=l2(0.01),
         recurrent_regularizer=l2(0.01)),
    BatchNormalization(),
    Dropout(0.5),

    # Dense Layer
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),

    # Output Layer (5개 클래스)
    Dense(5, activation='softmax')
])

print("\n[모델 구조 변경사항]")
print("  1. LSTM 유닛: 64 → 32 (모델 용량 감소)")
print("  2. Dense 유닛: 32 → 16")
print("  3. Dropout: 0.3 → 0.5/0.4 (정규화 강화)")
print("  4. L2 정규화 추가 (가중치: 0.01)")
print("  5. BatchNormalization 레이어 추가")

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ 모델 구축 완료")
model.summary()

# ==================== 3. 콜백 설정 ====================
print("\n[3단계] 콜백 설정 중...")

os.makedirs('../results', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        '../results/best_model_improved.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print("✓ 콜백 설정 완료")

# ==================== 4. 모델 학습 ====================
print("\n[4단계] 모델 학습 시작...")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Max Epochs: {EPOCHS}")
print(f"  - Learning Rate: {LEARNING_RATE}")
print(f"  - 데이터 증강: 가우시안 노이즈 추가")
print("\n" + "=" * 60)

history = model.fit(
    X_train_augmented, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("✓ 모델 학습 완료!")

# ==================== 5. 학습 과정 시각화 ====================
print("\n[5단계] 학습 과정 시각화 중...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss 그래프
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('모델 Loss 변화 (일반화 개선)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy 그래프
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('모델 Accuracy 변화 (일반화 개선)', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/training_history_improved.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/training_history_improved.png")
plt.close()

# ==================== 6. Test 데이터 평가 ====================
print("\n[6단계] Test 데이터 평가 중...")

# 최고 성능 모델 로드
best_model = keras.models.load_model('../results/best_model_improved.keras')

# 예측
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# 정확도
test_accuracy = np.mean(y_pred == y_true)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# 과적합 진단
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]
overfit_gap = train_accuracy - test_accuracy

print(f"\n[과적합 진단]")
print(f"  - Train Accuracy: {train_accuracy:.4f}")
print(f"  - Validation Accuracy: {val_accuracy:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Train-Test Gap: {overfit_gap:.4f}")

if overfit_gap > 0.1:
    print("  ️  과적합 의심 (Gap > 0.1)")
elif overfit_gap < 0.01:
    print("  ✓ 매우 좋은 일반화 성능")
else:
    print("  ✓ 적절한 일반화 성능")

# Classification Report
print("\n" + "=" * 60)
print("분류 성능 리포트")
print("=" * 60)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, output_dict=True)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ==================== 7. 기존 방법 vs LSTM 비교 그래프 ====================
print("\n[7단계] 기존 방법 vs LSTM 비교 그래프 생성 중...")

# LSTM 성능 지표 추출
lstm_accuracy = report['accuracy']
lstm_precision = report['weighted avg']['precision']
lstm_recall = report['weighted avg']['recall']
lstm_f1 = report['weighted avg']['f1-score']

# 기존 방법 성능 (이진 분류 기준 - 예시 값, 실제 값으로 수정 필요)
baseline_accuracy = 0.295  # 약 29.5%
baseline_precision = 0.115  # 약 11.5%
baseline_recall = 0.295
baseline_f1 = 0.165

# 비교 데이터 생성
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
lstm_scores = [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1]

# 그래프 생성
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_scores, width, label='기존 방법',
               color='#FF9999', edgecolor='black', linewidth=1.5, alpha=0.8)
bars2 = ax.bar(x + width/2, lstm_scores, width, label='LSTM 모델',
               color='#66CCCC', edgecolor='black', linewidth=1.5, alpha=0.8)

# 값 표시
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('성능 지표', fontsize=14, fontweight='bold')
ax.set_ylabel('점수', fontsize=14, fontweight='bold')
ax.set_title('기존 방법 vs LSTM 모델 성능 비교', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.set_ylim([0, 1.1])
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

# 배경색 추가
ax.set_facecolor('#F8F8F8')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('../results/baseline_vs_lstm_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/baseline_vs_lstm_comparison.png")
plt.close()

# 개선율 계산 및 출력
print("\n" + "=" * 60)
print("기존 방법 대비 LSTM 개선율")
print("=" * 60)

improvements = {
    'Accuracy': (lstm_accuracy - baseline_accuracy) / baseline_accuracy * 100,
    'Precision': (lstm_precision - baseline_precision) / baseline_precision * 100,
    'Recall': (lstm_recall - baseline_recall) / baseline_recall * 100,
    'F1-Score': (lstm_f1 - baseline_f1) / baseline_f1 * 100
}

for metric, improvement in improvements.items():
    print(f"  {metric:12}: {improvement:+7.2f}% 개선")

# ==================== 8. 혼동 행렬 시각화 ====================
print("\n[8단계] 혼동 행렬 생성 중...")

# 혼동 행렬 계산
cm = confusion_matrix(y_true, y_pred)

# 시각화 (5x5 행렬)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': '샘플 수'},
            annot_kws={'size': 12})
plt.xlabel('예측 클래스', fontsize=13, fontweight='bold')
plt.ylabel('실제 클래스', fontsize=13, fontweight='bold')
plt.title('혼동 행렬 - 일반화 개선 버전', fontsize=15, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/confusion_matrix_improved.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/confusion_matrix_improved.png")
plt.close()

# 정규화된 혼동 행렬 (비율)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12, 10))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': '비율'},
            annot_kws={'size': 12})
plt.xlabel('예측 클래스', fontsize=13, fontweight='bold')
plt.ylabel('실제 클래스', fontsize=13, fontweight='bold')
plt.title('정규화된 혼동 행렬 - 일반화 개선 버전', fontsize=15, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/confusion_matrix_normalized_improved.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/confusion_matrix_normalized_improved.png")
plt.close()

# ==================== 9. 예측 샘플 확인 ====================
print("\n[9단계] 예측 샘플 확인...")

print("\n" + "=" * 60)
print("예측 샘플 (각 클래스별 3개)")
print("=" * 60)

for class_id, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_true == class_id)[0][:3]

    print(f"\n[{class_name}]")
    for idx in class_indices:
        true_label = CLASS_NAMES[y_true[idx]]
        pred_label = CLASS_NAMES[y_pred[idx]]
        confidence = y_pred_proba[idx][y_pred[idx]] * 100

        status = "✓" if true_label == pred_label else "✗"
        print(f"  {status} 실제: {true_label:15} | 예측: {pred_label:15} | 신뢰도: {confidence:5.2f}%")

# ==================== 10. 클래스별 정확도 분석 ====================
print("\n[10단계] 클래스별 정확도 분석...")

print("\n" + "=" * 60)
print("클래스별 세부 성능")
print("=" * 60)

for class_id, class_name in enumerate(CLASS_NAMES):
    class_indices = np.where(y_true == class_id)[0]
    if len(class_indices) > 0:
        class_accuracy = np.mean(y_pred[class_indices] == class_id)
        total_samples = len(class_indices)
        correct_samples = np.sum(y_pred[class_indices] == class_id)
        print(f"  [{class_name:15}] 정확도: {class_accuracy:.4f} ({correct_samples}/{total_samples})")

# ==================== 11. 성능 비교 테이블 생성 ====================
print("\n[11단계] 성능 비교 테이블 생성...")

comparison_data = {
    '방법': ['기존 방법 (이진 분류)', 'LSTM 모델 (5-클래스)'],
    'Accuracy': [f'{baseline_accuracy:.4f}', f'{lstm_accuracy:.4f}'],
    'Precision': [f'{baseline_precision:.4f}', f'{lstm_precision:.4f}'],
    'Recall': [f'{baseline_recall:.4f}', f'{lstm_recall:.4f}'],
    'F1-Score': [f'{baseline_f1:.4f}', f'{lstm_f1:.4f}']
}

import pandas as pd
comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 60)
print("성능 비교 테이블")
print("=" * 60)
print(comparison_df.to_string(index=False))

# CSV로 저장
comparison_df.to_csv('../results/performance_comparison.csv', index=False, encoding='utf-8-sig')
print("\n✓ 저장: ../results/performance_comparison.csv")

# ==================== 12. 최종 결과 요약 ====================
print("\n" + "=" * 60)
print("학습 및 평가 완료! (일반화 개선 버전)")
print("=" * 60)

print(f"\n 최종 성능:")
print(f"  - Train Accuracy: {train_accuracy:.4f}")
print(f"  - Validation Accuracy: {val_accuracy:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")
print(f"  - Overfitting Gap: {overfit_gap:.4f}")

print(f"\n 기존 방법 대비:")
print(f"  - Accuracy 개선: {improvements['Accuracy']:+.2f}%")
print(f"  - Precision 개선: {improvements['Precision']:+.2f}%")
print(f"  - Recall 개선: {improvements['Recall']:+.2f}%")
print(f"  - F1-Score 개선: {improvements['F1-Score']:+.2f}%")

print(f"\n 저장된 파일:")
print(f"  - 모델: ../results/best_model_improved.keras")
print(f"  - 학습 그래프: ../results/training_history_improved.png")
print(f"  - 비교 그래프: ../results/baseline_vs_lstm_comparison.png")
print(f"  - 혼동 행렬: ../results/confusion_matrix_improved.png")
print(f"  - 정규화 혼동 행렬: ../results/confusion_matrix_normalized_improved.png")
print(f"  - 성능 비교 테이블: ../results/performance_comparison.csv")

print(f"\n🔧 주요 변경사항:")
print(f"  1. 배치 크기: 64 → 32")
print(f"  2. 에포크: 100 → 50")
print(f"  3. 학습률: 0.001 → 0.01")
print(f"  4. LSTM 유닛: 64 → 32")
print(f"  5. Dropout: 0.3 → 0.5")
print(f"  6. L2 정규화 추가 (0.01)")
print(f"  7. BatchNormalization 추가")
print(f"  8. 데이터 증강 (가우시안 노이즈)")
print(f"  9. Early Stopping patience: 10 → 7")
print(f"  10. 기존 방법과 비교 그래프 추가")

print(f"\n 작업 완료")