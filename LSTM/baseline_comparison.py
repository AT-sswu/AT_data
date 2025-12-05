import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import pandas as pd

plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ['Stationary', 'Vibration', 'Windy']

with open('../processed_data.pkl', 'rb') as f:
    data_dict = pickle.load(f)

X_test = data_dict['X_test']
y_test = data_dict['y_test']
y_true = np.argmax(y_test, axis=1)


def baseline_2bit_classifier(X):
    predictions = []
    for sequence in X:
        accel_data = sequence[:, :3]
        std = np.std(accel_data)
        if std < 0.1:
            pred = 0
        elif std < 0.3:
            pred = 2
        else:
            pred = 1
        predictions.append(pred)
    return np.array(predictions)


y_pred_baseline = baseline_2bit_classifier(X_test)

baseline_accuracy = accuracy_score(y_true, y_pred_baseline)
baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_baseline, average='weighted', zero_division=0
)

lstm_model = keras.models.load_model('../results/best_model.keras')
y_pred_lstm_proba = lstm_model.predict(X_test, verbose=0)
y_pred_lstm = np.argmax(y_pred_lstm_proba, axis=1)

lstm_accuracy = accuracy_score(y_true, y_pred_lstm)
lstm_precision, lstm_recall, lstm_f1, _ = precision_recall_fscore_support(
    y_true, y_pred_lstm, average='weighted', zero_division=0
)

improvement_accuracy = ((lstm_accuracy - baseline_accuracy) / baseline_accuracy) * 100
improvement_precision = ((lstm_precision - baseline_precision) / baseline_precision) * 100
improvement_recall = ((lstm_recall - baseline_recall) / baseline_recall) * 100
improvement_f1 = ((lstm_f1 - baseline_f1) / baseline_f1) * 100


# 성능 지표 비교 시각화
fig, ax = plt.subplots(figsize=(12, 7))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_scores = [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1]
lstm_scores = [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_scores, width, label='기존 방법',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, lstm_scores, width, label='LSTM 모델',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('성능 지표', fontsize=13, fontweight='bold')
ax.set_ylabel('점수', fontsize=13, fontweight='bold')
ax.set_title('기존 방법 vs LSTM 모델 성능 비교', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# 혼동 행렬 비교
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

cm_baseline = confusion_matrix(y_true, y_pred_baseline)
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])

axes[0].set_xlabel('예측 클래스')
axes[0].set_ylabel('실제 클래스')
axes[0].set_title(f'기존 방법\nAcc: {baseline_accuracy:.4f}')

cm_lstm = confusion_matrix(y_true, y_pred_lstm)
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])

axes[1].set_xlabel('예측 클래스')
axes[1].set_ylabel('실제 클래스')
axes[1].set_title(f'LSTM 모델\nAcc: {lstm_accuracy:.4f}')

plt.tight_layout()
plt.savefig('results/confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
plt.close()


# 클래스별 F1-score 비교
baseline_f1_per_class = precision_recall_fscore_support(
    y_true, y_pred_baseline, average=None, zero_division=0)[2]
lstm_f1_per_class = precision_recall_fscore_support(
    y_true, y_pred_lstm, average=None, zero_division=0)[2]

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(CLASS_NAMES))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_f1_per_class, width, label='기존 방법',
               color='#FF6B6B', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, lstm_f1_per_class, width, label='LSTM 모델',
               color='#4ECDC4', alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('클래스')
ax.set_ylabel('F1-Score')
ax.set_title('클래스별 F1-Score 비교', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES)
ax.legend(fontsize=12)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('results/class_wise_f1_comparison.png', dpi=300, bbox_inches='tight')
plt.close()



# CSV 저장
comparison_df = pd.DataFrame({
    '성능 지표': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    '기존 방법': [baseline_accuracy, baseline_precision, baseline_recall, baseline_f1],
    'LSTM 모델': [lstm_accuracy, lstm_precision, lstm_recall, lstm_f1],
    '개선율 (%)': [improvement_accuracy, improvement_precision, improvement_recall, improvement_f1]
})

comparison_df.to_csv('results/performance_comparison.csv', index=False, encoding='utf-8-sig')

class_comparison_df = pd.DataFrame({
    '클래스': CLASS_NAMES,
    '기존 방법 F1': baseline_f1_per_class,
    'LSTM F1': lstm_f1_per_class,
})

class_comparison_df.to_csv('results/class_wise_comparison.csv', index=False, encoding='utf-8-sig')
