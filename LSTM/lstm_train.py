import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False



# ==================== 설정 ====================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001

# 라벨 이름 (5개 클래스)
CLASS_NAMES = ['lidar', 'motor', 'driving', 'lidar_driving', 'motor_driving']

print("=" * 60)
print("LSTM 모델 학습 시작")
print("=" * 60)

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f"\n✓ GPU 가속 활성화: {len(gpus)}개의 GPU 사용")
else:
    print(f"\n✓ CPU 모드로 실행")

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

except FileNotFoundError:
    print(" processed_data.pkl 파일을 찾을 수 없습니다.")
    print("먼저 data_preparation.py를 실행하세요!")
    exit(1)

# ==================== 2. 모델 구축 ====================
print("\n[2단계] LSTM 모델 구축 중...")

model = Sequential([
    # Input Layer
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),

    # Second LSTM Layer
    LSTM(64, return_sequences=False),
    Dropout(0.3),

    # Dense Layer
    Dense(32, activation='relu'),
    Dropout(0.3),

    # Output Layer (5개 클래스)
    Dense(5, activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ 모델 구축 완료")
model.summary()

# ==================== 3. 콜백 설정 ====================
print("\n[3단계] 콜백 설정 중...")

# 결과 저장 폴더
os.makedirs('../results', exist_ok=True)

callbacks = [
    # Early Stopping: Validation loss가 10 에포크 동안 개선되지 않으면 중단
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    # Model Checkpoint: 최고 성능 모델 저장
    ModelCheckpoint(
        '../results/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),

    # Learning Rate Reduction: 성능 개선이 없으면 학습률 감소
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

print("✓ 콜백 설정 완료")

# ==================== 4. 모델 학습 ====================
print("\n[4단계] 모델 학습 시작...")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Max Epochs: {EPOCHS}")
print(f"  - Learning Rate: {LEARNING_RATE}")
print("\n" + "=" * 60)

history = model.fit(
    X_train, y_train,
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
axes[0].set_title('모델 Loss 변화', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy 그래프
axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('모델 Accuracy 변화', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/training_history.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/training_history.png")
plt.close()

# ==================== 6. Test 데이터 평가 ====================
print("\n[6단계] Test 데이터 평가 중...")

# 최고 성능 모델 로드
best_model = keras.models.load_model('../results/best_model.keras')

# 예측
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = np.argmax(y_test, axis=1)

# 정확도
test_accuracy = np.mean(y_pred == y_true)
print(f"\n✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

# Classification Report
print("\n" + "=" * 60)
print("분류 성능 리포트")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# ==================== 7. 혼동 행렬 시각화 ====================
print("\n[7단계] 혼동 행렬 생성 중...")

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
plt.title('혼동 행렬 (Confusion Matrix)', fontsize=15, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/confusion_matrix.png")
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
plt.title('정규화된 혼동 행렬 (비율)', fontsize=15, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../results/confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
print("✓ 저장: ../results/confusion_matrix_normalized.png")
plt.close()

# ==================== 8. 예측 샘플 확인 ====================
print("\n[8단계] 예측 샘플 확인...")

# 각 클래스별로 3개씩 샘플 출력
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

# ==================== 9. 클래스별 정확도 분석 ====================
print("\n[9단계] 클래스별 정확도 분석...")

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

# ==================== 10. 최종 결과 요약 ====================
print("\n" + "=" * 60)
print("학습 및 평가 완료!")
print("=" * 60)

print(f"\n 최종 성능:")
print(f"  - Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  - Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  - Test Accuracy: {test_accuracy:.4f}")

print(f"\n 저장된 파일:")
print(f"  - 모델: ../results/best_model.keras")
print(f"  - 학습 그래프: ../results/training_history.png")
print(f"  - 혼동 행렬: ../results/confusion_matrix.png")
print(f"  - 정규화 혼동 행렬: ../results/confusion_matrix_normalized.png")

print(f"\n모든 작업 완료!")
