import tensorflow as tf
import sys

print("=" * 60)
print("TensorFlow 환경 확인")
print("=" * 60)

# TensorFlow 버전
print(f"\n✓ TensorFlow 버전: {tf.__version__}")
print(f"✓ Python 버전: {sys.version}")

# GPU 장치 확인
print("\n" + "=" * 60)
print("사용 가능한 디바이스 목록")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
cpus = tf.config.list_physical_devices('CPU')

print(f"\n✓ GPU 개수: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

print(f"\n✓ CPU 개수: {len(cpus)}")
for cpu in cpus:
    print(f"  - {cpu}")

# Metal GPU 확인
if len(gpus) > 0:
    print("\n" + "=" * 60)
    print(" GPU 가속이 활성화되었습니다.")
    print("=" * 60)
    print("\nTensorFlow Metal이 정상적으로 작동하고 있습니다.")
    print("LSTM 학습 시 GPU 가속을 사용할 수 있습니다.")
else:
    print("\n" + "=" * 60)
    print("⚠  GPU를 찾을 수 없습니다")
    print("=" * 60)
    print("\nCPU 모드로 작동합니다.")
    print("\n해결 방법:")
    print("1. tensorflow-macos가 설치되어 있는지 확인하세요:")
    print("   pip install tensorflow-macos")
    print("\n2. tensorflow-metal이 설치되어 있는지 확인하세요:")
    print("   pip install tensorflow-metal")
    print("\n3. M1/M2/M3 맥북이 아닌 경우 GPU 가속을 사용할 수 없습니다.")

# 간단한 연산 테스트
print("\n" + "=" * 60)
print("간단한 GPU 연산 테스트")
print("=" * 60)

try:
    with tf.device('/GPU:0' if len(gpus) > 0 else '/CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)

    print("\n✓ 행렬 곱셈 테스트 성공")
    print(f"디바이스: {'GPU' if len(gpus) > 0 else 'CPU'}")
    print(f"결과:\n{c.numpy()}")
except Exception as e:
    print(f"\n✗ 테스트 실패: {e}")

print("\n" + "=" * 60)
print("확인 완료")
print("=" * 60)