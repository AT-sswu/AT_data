"""
DNN-MLP 회귀 모델 학습 (5개 클래스)
특징 벡터 → 최적 공진 주파수 직접 예측
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# ==================== 설정 ====================
BATCH_SIZE = 128
EPOCHS = 300  # 충분한 학습 시간 확보
LEARNING_RATE = 0.001

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

print("=" * 60)
print("DNN-MLP 회귀 모델 학습 (5개 클래스)")
print("=" * 60)

# GPU 확인
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f"\n✓ GPU 가속: {len(gpus)}개 GPU")
else:
    print(f"\n✓ CPU 모드")

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] 데이터 로드...")

try:
    with open('processed_data_dnn_regression.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    X_train = data_dict['X_train']
    X_val = data_dict['X_val']
    X_test = data_dict['X_test']
    y_train = data_dict['y_train']
    y_val = data_dict['y_val']
    y_test = data_dict['y_test']
    feature_count = data_dict['feature_count']
    label_map = data_dict.get('label_map', LABEL_MAP)

    print(f"✓ 데이터 로드 완료")
    print(f"  - Train: {X_train.shape}")
    print(f"  - Val: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")
    print(f"  - 특징 수: {feature_count}개")
    print(f"  - 클래스 수: {len(label_map)}개")
    print(f"  - 주파수 범위: {y_train.min():.2f} ~ {y_train.max():.2f} Hz")

except FileNotFoundError:
    print("️  processed_data_dnn_regression.pkl을 찾을 수 없습니다.")
    print("먼저 dnn_mlp_data_prep.py를 실행하세요!")
    exit(1)

# ==================== 2. DNN-MLP 회귀 모델 구축 ====================
print("\n[2단계] DNN-MLP 회귀 모델 구축...")

model = Sequential([
    # Input Layer
    Dense(256, activation='relu', input_shape=(feature_count,)),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden Layer 1
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    # Hidden Layer 2
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    # Hidden Layer 3
    Dense(32, activation='relu'),
    Dropout(0.3),

    # Hidden Layer 4
    Dense(16, activation='relu'),

    # Output Layer (회귀)
    Dense(1, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mean_squared_error',
    metrics=['mae']
)

print("✓ 모델 구축 완료 (DNN-MLP 회귀)")
model.summary()

# ==================== 3. 콜백 설정 ====================
print("\n[3단계] 콜백 설정...")

os.makedirs('results_dnn', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=100,  # 인내심을 2배로 늘림 (15 → 30)
        restore_best_weights=True,
        verbose=1
    ),

    ModelCheckpoint(
        'results_dnn/best_dnn_regression_model.keras',
        monitor='val_mae',
        save_best_only=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,  # 학습률 감소도 더 늦게 (7 → 12)
        min_lr=1e-8,  # 최소 학습률도 더 작게
        verbose=1
    )
]

print("✓ 콜백 설정 완료")

# ==================== 4. 모델 학습 ====================
print("\n[4단계] 모델 학습...")
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
print("✓ 학습 완료!")

# ==================== 5. 학습 과정 시각화 ====================
print("\n[5단계] 학습 과정 시각화...")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss (MSE)
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('MSE Loss', fontsize=12)
axes[0].set_title('DNN-MLP Loss 변화 (MSE)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# MAE
axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('MAE (Hz)', fontsize=12)
axes[1].set_title('DNN-MLP 평균 절대 오차', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_dnn/dnn_regression_training_history.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_dnn/dnn_regression_training_history.png")
plt.close()

# ==================== 6. Test 평가 ====================
print("\n[6단계] Test 데이터 평가...")

best_model = keras.models.load_model('results_dnn/best_dnn_regression_model.keras')

y_pred = best_model.predict(X_test, verbose=0).flatten()

# 성능 지표
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n✓ Test 성능:")
print(f"  - MAE: {mae:.4f} Hz")
print(f"  - RMSE: {rmse:.4f} Hz")
print(f"  - R² Score: {r2:.4f}")

# 오차 분석
errors = np.abs(y_test - y_pred)
print(f"\n✓ 오차 통계:")
print(f"  - 최소 오차: {errors.min():.4f} Hz")
print(f"  - 최대 오차: {errors.max():.4f} Hz")
print(f"  - 중앙값 오차: {np.median(errors):.4f} Hz")
print(f"  - 오차 < 1Hz: {np.sum(errors < 1.0) / len(errors) * 100:.2f}%")
print(f"  - 오차 < 2Hz: {np.sum(errors < 2.0) / len(errors) * 100:.2f}%")
print(f"  - 오차 < 5Hz: {np.sum(errors < 5.0) / len(errors) * 100:.2f}%")

# ==================== 7. 예측 결과 시각화 ====================
print("\n[7단계] 예측 결과 시각화...")

# 실제 vs 예측
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 산점도
axes[0].scatter(y_test, y_pred, alpha=0.5, s=20)
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='완벽한 예측')
axes[0].set_xlabel('실제 주파수 (Hz)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('예측 주파수 (Hz)', fontsize=12, fontweight='bold')
axes[0].set_title(f'DNN-MLP 실제 vs 예측\n(MAE: {mae:.2f} Hz, R²: {r2:.3f})',
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# 오차 히스토그램
axes[1].hist(errors, bins=50, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(mae, color='red', linestyle='--', linewidth=2,
                label=f'MAE: {mae:.2f} Hz')
axes[1].set_xlabel('예측 오차 (Hz)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('샘플 수', fontsize=12, fontweight='bold')
axes[1].set_title('DNN-MLP 예측 오차 분포', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_dnn/dnn_regression_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 저장: results_dnn/dnn_regression_prediction_analysis.png")
plt.close()

# ==================== 8. 결과 저장 ====================
print("\n[8단계] 결과 저장...")

results = {
    'y_test': y_test,
    'y_pred': y_pred,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'label_map': label_map,
    'num_classes': len(label_map),
    'model_path': 'results_dnn/best_dnn_regression_model.keras'
}

with open('results_dnn/dnn_regression_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✓ 저장: results_dnn/dnn_regression_results.pkl")

# ==================== 9. 최종 요약 ====================
print("\n" + "=" * 60)
print("DNN-MLP 회귀 모델 학습 완료!")
print("=" * 60)

print(f"\n📊 최종 성능:")
print(f"  - MAE: {mae:.4f} Hz")
print(f"  - RMSE: {rmse:.4f} Hz")
print(f"  - R² Score: {r2:.4f}")

print(f"\n 학습 클래스:")
for label_name, label_id in sorted(label_map.items(), key=lambda x: x[1]):
    print(f"  - {label_id}: {label_name}")

print(f"\n 생성된 파일:")
print(f"  - 모델: results_dnn/best_dnn_regression_model.keras")
print(f"  - 학습 그래프: results_dnn/dnn_regression_training_history.png")
print(f"  - 예측 분석: results_dnn/dnn_regression_prediction_analysis.png")
print(f"  - 결과 데이터: results_dnn/dnn_regression_results.pkl")

print(f"\n 모든 DNN-MLP 모델 학습 완료")