import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# 설정
DATA_FOLDER = '/Users/seohyeon/PycharmProjects/AT_data/datasets'
SEQUENCE_LENGTH = 100  # 시퀀스 길이 (100개 연속 데이터포인트)
STEP_SIZE = 10  # 슬라이딩 윈도우 스텝 (10개씩 건너뛰기)
RANDOM_STATE = 42

# 라벨 매핑
LABEL_MAP = {
    'stationary': 0,
    'vibration': 1,
    'windy': 2
}

print("=" * 60)
print("데이터 로드 및 전처리 시작")
print("=" * 60)



# ==================== 1. 데이터 로드 ====================
print("\n[1단계] CSV 파일 로드 중...")

all_data = []
all_labels = []

# 데이터 폴더 확인
if not os.path.exists(DATA_FOLDER):
    print(f" 폴더를 찾을 수 없습니다: {DATA_FOLDER}")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print("\n해결 방법:")
    print("1. 스크립트 상단의 DATA_FOLDER 변수를 수정하세요")
    print("2. 또는 CSV 파일을 './data' 폴더에 넣으세요")
    exit(1)

# CSV 파일 목록
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

if len(csv_files) == 0:
    print(f"  CSV 파일을 찾을 수 없습니다: {DATA_FOLDER}")
    exit(1)

print(f" 찾은 CSV 파일 개수: {len(csv_files)}개")

# 각 파일 로드
for filename in sorted(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)

    # 파일명에서 라벨 추출
    if 'stationary' in filename.lower():
        label = LABEL_MAP['stationary']
        label_name = 'Stationary'
    elif 'vibration' in filename.lower():
        label = LABEL_MAP['vibration']
        label_name = 'Vibration'
    elif 'windy' in filename.lower():
        label = LABEL_MAP['windy']
        label_name = 'Windy'
    else:
        print(f"  라벨을 인식할 수 없는 파일: {filename}")
        continue

    # CSV 로드
    df = pd.read_csv(filepath)

    # 필요한 컬럼만 선택 (Time 제외)
    feature_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    # 컬럼 존재 확인
    if not all(col in df.columns for col in feature_columns):
        print(f" 필요한 컬럼이 없는 파일: {filename}")
        print(f" 파일 컬럼: {df.columns.tolist()}")
        continue

    data = df[feature_columns].values

    print(f"  ✓ {filename}: {len(data):,}개 샘플 ({label_name})")

    all_data.append(data)
    all_labels.extend([label] * len(data))

# 데이터 통합
X_raw = np.vstack(all_data)
y_raw = np.array(all_labels)

print(f"\n✓ 전체 데이터 형태: {X_raw.shape}")
print(f"✓ 전체 라벨 형태: {y_raw.shape}")
print(f"✓ 클래스별 샘플 수:")
for label_name, label_id in LABEL_MAP.items():
    count = np.sum(y_raw == label_id)
    print(f"  - {label_name.capitalize()}: {count:,}개")



# ==================== 2. 데이터 정규화 ====================
print("\n[2단계] 데이터 정규화 중...")

scaler = MinMaxScaler(feature_range=(0, 1))
X_normalized = scaler.fit_transform(X_raw)

print(f"✓ 정규화 완료: 범위 [0, 1]")
print(f"  - 최소값: {X_normalized.min():.4f}")
print(f"  - 최대값: {X_normalized.max():.4f}")
print(f"  - 평균: {X_normalized.mean():.4f}")



# ==================== 3. 슬라이딩 윈도우로 시퀀스 생성 ====================
print(f"\n[3단계] 시퀀스 생성 중 (길이={SEQUENCE_LENGTH}, 스텝={STEP_SIZE})...")


def create_sequences(data, labels, seq_length, step_size):
    # 슬라이딩 윈도우로 시퀀스 생성
    sequences = []
    sequence_labels = []

    # 각 클래스별로 처리
    for label_id in np.unique(labels):
        # 해당 클래스의 데이터만 추출
        class_indices = np.where(labels == label_id)[0]
        class_data = data[class_indices]

        # 슬라이딩 윈도우
        for i in range(0, len(class_data) - seq_length + 1, step_size):
            seq = class_data[i:i + seq_length]
            sequences.append(seq)
            sequence_labels.append(label_id)

    return np.array(sequences), np.array(sequence_labels)


X_seq, y_seq = create_sequences(X_normalized, y_raw, SEQUENCE_LENGTH, STEP_SIZE)

print(f"✓ 생성된 시퀀스 개수: {len(X_seq):,}개")
print(f"✓ 시퀀스 형태: {X_seq.shape}")
print(f"  - 샘플 수: {X_seq.shape[0]:,}")
print(f"  - 시퀀스 길이: {X_seq.shape[1]}")
print(f"  - 특성 수: {X_seq.shape[2]}")

print(f"\n✓ 클래스별 시퀀스 수:")
for label_name, label_id in LABEL_MAP.items():
    count = np.sum(y_seq == label_id)
    print(f"  - {label_name.capitalize()}: {count:,}개")



# ==================== 4. 데이터 분할 ====================
print(f"\n[4단계] 데이터 분할 중 (Train 70%, Val 15%, Test 15%)...")

# One-hot 인코딩
y_seq_onehot = np.eye(3)[y_seq]

# Train + Temp 분할 (70% / 30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq_onehot,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_seq
)

# Validation + Test 분할 (15% / 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=np.argmax(y_temp, axis=1)
)

print(f"✓ Train 세트: {X_train.shape[0]:,}개")
print(f"✓ Validation 세트: {X_val.shape[0]:,}개")
print(f"✓ Test 세트: {X_test.shape[0]:,}개")



# ==================== 5. 데이터 저장 ====================
print(f"\n[5단계] 전처리된 데이터 저장 중...")

# 저장할 데이터
data_dict = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'scaler': scaler,
    'label_map': LABEL_MAP
}

# 피클로 저장
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"✓ 저장 완료: processed_data.pkl")


# ==================== 6. 데이터 요약 ====================
print("\n" + "=" * 60)
print("데이터 전처리 완료")
print("=" * 60)

print(f"\n 최종 데이터셋 정보:")
print(f"  - Train: {X_train.shape}")
print(f"  - Validation: {X_val.shape}")
print(f"  - Test: {X_test.shape}")
print(f"  - 시퀀스 길이: {SEQUENCE_LENGTH}")
print(f"  - 특성 수: 6 (Accel_X/Y/Z, Gyro_X/Y/Z)")
print(f"  - 클래스 수: 3 (Stationary, Vibration, Windy)")

