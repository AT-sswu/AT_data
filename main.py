import serial
import csv
import time

PORT = 'COM4'  # 자신의 포트에 맞게 수정
BAUD = 115200
CSV_FILE = 'mpu6050_data.csv'

# 시리얼 포트 연결
ser = serial.Serial(PORT, BAUD)
time.sleep(2)  # 아두이노 연결 안정화

# CSV 파일 열기
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"])

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            parts = line.split(',')

            if len(parts) == 7:
                writer.writerow(parts)
                print(parts)  # 실시간 확인용 출력

    except KeyboardInterrupt:
        print("\n[종료됨] CSV 저장 완료")

ser.close()
