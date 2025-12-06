import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import pickle

# 설정
DATA_FOLDER = '/Users/seohyeon/PycharmProjects/AT_data/data_v1'
SEQUENCE_LENGTH = 100  # 시퀀스 길이 (100개 연속 데이터포인트)
STEP_SIZE = 10  # 슬라이딩 윈도우 스텝 (10개씩 건너뛰기)
RANDOM_STATE = 42

# 🔧 변경1: 시드 고정 (재현성)
np.random.seed(RANDOM_STATE)

# 라벨 매핑 (5개 클래스)
LABEL_MAP = {
    'lidar': 0,
    'motor': 1,
    'driving': 2,
    'lidar_driving': 3,
    'motor_driving': 4
}

print("=" * 60)
print("데이터 로드 및 전처리 시작 (데이터 누수 해결 버전)")
print("=" * 60)

# ==================== 1. 데이터 로드 ====================
print("\n[1단계] CSV 파일 로드 중...")

all_data = []
all_labels = []

# 데이터 폴더 확인
if not os.path.exists(DATA_FOLDER):
    print(f"❌ 폴더를 찾을 수 없습니다: {DATA_FOLDER}")
    print(f"현재 작업 디렉토리: {os.getcwd()}")
    print("\n해결 방법:")
    print("1. 스크립트 상단의 DATA_FOLDER 변수를 수정하세요")
    print("2. 또는 CSV 파일을 './data' 폴더에 넣으세요")
    exit(1)

# CSV 파일 목록
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

if len(csv_files) == 0:
    print(f"❌ CSV 파일을 찾을 수 없습니다: {DATA_FOLDER}")
    exit(1)

print(f"✓ 찾은 CSV 파일 개수: {len(csv_files)}개")

# 각 파일 로드
for filename in sorted(csv_files):
    filepath = os.path.join(DATA_FOLDER, filename)

    # 파일명에서 라벨 추출 (순서 중요: 복합 라벨을 먼저 체크)
    if 'lidar_driving' in filename.lower():
        label = LABEL_MAP['lidar_driving']
        label_name = 'lidar_driving'
    elif 'motor_driving' in filename.lower():
        label = LABEL_MAP['motor_driving']
        label_name = 'motor_driving'
    elif 'lidar' in filename.lower():
        label = LABEL_MAP['lidar']
        label_name = 'lidar'
    elif 'motor' in filename.lower():
        label = LABEL_MAP['motor']
        label_name = 'motor'
    elif 'driving' in filename.lower():
        label = LABEL_MAP['driving']
        label_name = 'driving'
    else:
        print(f"⚠️ 라벨을 인식할 수 없는 파일: {filename}")
        continue

    # CSV 로드
    df = pd.read_csv(filepath)

    # 필요한 컬럼만 선택 (Time 제외)
    feature_columns = ['Accel_X', 'Accel_Y', 'Accel_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

    # 컬럼 존재 확인
    if not all(col in df.columns for col in feature_columns):
        print(f"⚠️ 필요한 컬럼이 없는 파일: {filename}")
        print(f"   파일 컬럼: {df.columns.tolist()}")
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

# ==================== 2. 슬라이딩 윈도우로 시퀀스 생성 (정규화 전!) ====================
print(f"\n[2단계] 시퀀스 생성 중 (길이={SEQUENCE_LENGTH}, 스텝={STEP_SIZE})...")

# 🔧 변경2: 정규화 전에 시퀀스 생성
def create_sequences(data, labels, seq_length, step_size):
    """슬라이딩 윈도우로 시퀀스 생성"""
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

X_seq, y_seq = create_sequences(X_raw, y_raw, SEQUENCE_LENGTH, STEP_SIZE)

print(f"✓ 생성된 시퀀스 개수: {len(X_seq):,}개")
print(f"✓ 시퀀스 형태: {X_seq.shape}")
print(f"  - 샘플 수: {X_seq.shape[0]:,}")
print(f"  - 시퀀스 길이: {X_seq.shape[1]}")
print(f"  - 특성 수: {X_seq.shape[2]}")

print(f"\n✓ 클래스별 시퀀스 수:")
for label_name, label_id in LABEL_MAP.items():
    count = np.sum(y_seq == label_id)
    print(f"  - {label_name.capitalize()}: {count:,}개")

# ==================== 3. 데이터 분할 (정규화 전!) ====================
print(f"\n[3단계] 데이터 분할 중 (Train 70%, Val 15%, Test 15%)...")

# 🔧 변경3: 정규화 전에 먼저 분할!
print("\n⚠️  [중요] 데이터 누수 방지:")
print("  → Train/Val/Test 분할 후 Train 데이터로만 정규화")

# One-hot 인코딩 (5개 클래스)
y_seq_onehot = np.eye(5)[y_seq]

# Train + Temp 분할 (70% / 30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_seq, y_seq_onehot,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y_seq,
    shuffle=True  # 명시적으로 섞기
)

# Validation + Test 분할 (15% / 15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=np.argmax(y_temp, axis=1),
    shuffle=True
)

print(f"✓ Train 세트: {X_train.shape[0]:,}개")
print(f"✓ Validation 세트: {X_val.shape[0]:,}개")
print(f"✓ Test 세트: {X_test.shape[0]:,}개")

# ==================== 4. 데이터 정규화 (분할 후!) ====================
print(f"\n[4단계] 데이터 정규화 중...")

# 🔧 변경4: Train 데이터로만 Scaler 학습!
print("\n✓ Train 데이터로만 Scaler 학습 (데이터 누수 방지)")

# 시퀀스를 2D로 변환 (정규화를 위해)
n_samples_train = X_train.shape[0]
n_samples_val = X_val.shape[0]
n_samples_test = X_test.shape[0]

X_train_2d = X_train.reshape(-1, X_train.shape[2])
X_val_2d = X_val.reshape(-1, X_val.shape[2])
X_test_2d = X_test.reshape(-1, X_test.shape[2])

# Scaler 생성 및 Train으로만 학습
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_normalized = scaler.fit_transform(X_train_2d)  # fit_transform (Train)
X_val_normalized = scaler.transform(X_val_2d)          # transform만 (Val)
X_test_normalized = scaler.transform(X_test_2d)        # transform만 (Test)

# 다시 3D로 변환
X_train = X_train_normalized.reshape(n_samples_train, SEQUENCE_LENGTH, -1)
X_val = X_val_normalized.reshape(n_samples_val, SEQUENCE_LENGTH, -1)
X_test = X_test_normalized.reshape(n_samples_test, SEQUENCE_LENGTH, -1)

print(f"✓ 정규화 완료: 범위 [0, 1]")
print(f"\n[Train 데이터]")
print(f"  - 최소값: {X_train.min():.4f}")
print(f"  - 최대값: {X_train.max():.4f}")
print(f"  - 평균: {X_train.mean():.4f}")
print(f"\n[Validation 데이터]")
print(f"  - 최소값: {X_val.min():.4f}")
print(f"  - 최대값: {X_val.max():.4f}")
print(f"  - 평균: {X_val.mean():.4f}")
print(f"\n[Test 데이터]")
print(f"  - 최소값: {X_test.min():.4f}")
print(f"  - 최대값: {X_test.max():.4f}")
print(f"  - 평균: {X_test.mean():.4f}")

# 🔧 변경5: 데이터 중복 체크
print(f"\n[5단계] 데이터 품질 검증...")

# Train-Test 간 중복 체크
print("\n✓ Train-Test 중복 샘플 검사 중...")
train_flat = X_train.reshape(X_train.shape[0], -1)
test_flat = X_test.reshape(X_test.shape[0], -1)

# 샘플링해서 체크 (전체 체크는 시간이 오래 걸림)
check_size = min(100, len(train_flat), len(test_flat))
duplicates = 0

for i in range(check_size):
    # 각 test 샘플이 train에 있는지 확인
    if np.any(np.all(np.abs(train_flat - test_flat[i]) < 1e-6, axis=1)):
        duplicates += 1

if duplicates > 0:
    print(f"⚠️  경고: {duplicates}/{check_size} 중복 샘플 발견")
    print("  → 슬라이딩 윈도우의 STEP_SIZE를 크게 조정하거나")
    print("  → 시계열 분할(TimeSeriesSplit) 사용 고려")
else:
    print(f"✓ 중복 없음 (샘플 {check_size}개 체크)")

# 클래스 분포 확인
print("\n✓ 분할 후 클래스 분포:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    print(f"\n  [{split_name}]")
    y_labels = np.argmax(y_split, axis=1)
    for label_name, label_id in LABEL_MAP.items():
        count = np.sum(y_labels == label_id)
        percentage = (count / len(y_labels)) * 100
        print(f"    - {label_name:15}: {count:5,}개 ({percentage:5.2f}%)")

# ==================== 6. 데이터 저장 ====================
print(f"\n[6단계] 전처리된 데이터 저장 중...")

# 저장할 데이터
data_dict = {
    'X_train': X_train,
    'X_val': X_val,
    'X_test': X_test,
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,
    'scaler': scaler,
    'label_map': LABEL_MAP,
    'config': {
        'sequence_length': SEQUENCE_LENGTH,
        'step_size': STEP_SIZE,
        'random_state': RANDOM_STATE,
        'normalization': 'MinMaxScaler',
        'normalization_range': (0, 1),
        'data_leakage_prevented': True  # 데이터 누수 방지 확인
    }
}

# 피클로 저장
with open('../processed_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)

print(f"✓ 저장 완료: processed_data.pkl")

# ==================== 7. 데이터 요약 ====================
print("\n" + "=" * 60)
print("데이터 전처리 완료 (데이터 누수 방지 버전)")
print("=" * 60)

print(f"\n📊 최종 데이터셋 정보:")
print(f"  - Train: {X_train.shape}")
print(f"  - Validation: {X_val.shape}")
print(f"  - Test: {X_test.shape}")
print(f"  - 시퀀스 길이: {SEQUENCE_LENGTH}")
print(f"  - 특성 수: 6 (Accel_X/Y/Z, Gyro_X/Y/Z)")
print(f"  - 클래스 수: 5 (lidar, motor, driving, lidar_driving, motor_driving)")
