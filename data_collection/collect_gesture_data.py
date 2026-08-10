# -*- coding: utf-8 -*-
"""
손동작 데이터 수집 코드 (수집 전용 - 학습은 Colab에서 진행)

[전체 파이프라인]
    1. (로컬, 이 코드)  데이터 수집 -> gesture_features_index.csv
    2. (Colab)          CSV 업로드 -> MLP 학습 -> gesture_weights.npz 등 저장
    3. (Jetson)         npz 가중치 로드 -> 실시간 인식

[사용법 - Anaconda Prompt]
    conda activate gesture
    python collect_gesture_data.py

[조작법]
    SPACE : 녹화 시작 (그 상태로 동작을 수행)
    1~8   : 녹화 종료 + 해당 라벨로 저장
            1: 확대(zoom_in)          2: 축소(zoom_out)
            3: 위로 스크롤(scroll_up)  4: 아래로 스크롤(scroll_down)
            5: 왼쪽 이동(move_left)    6: 오른쪽 이동(move_right)
            7: 클릭(click)             8: 무동작(idle)
    ESC   : 종료

[결과물]
    gesture_features_index.csv (라벨 + 14개 특징, 한 동작 = 한 줄)
    -> 이 파일을 Google Drive에 업로드해서 Colab 학습에 사용
"""

import csv
import os
import math
import time

import cv2
import mediapipe as mp
import pandas as pd


# =========================================================
# 파일/클래스 설정
# =========================================================

CSV_PATH = "gesture_features_index.csv"

GESTURES = {
    ord("1"): "zoom_in",
    ord("2"): "zoom_out",
    ord("3"): "scroll_up",
    ord("4"): "scroll_down",
    ord("5"): "move_left",
    ord("6"): "move_right",
    ord("7"): "click",
    ord("8"): "idle",
}

# Colab 학습 코드와 Jetson 추론 코드 모두 이 순서를 그대로 사용해야 함
FEATURE_NAMES = [
    "wrist_dx",
    "wrist_dy",
    "index_dx",
    "index_dy",
    "pinch_delta",
    "pinch_speed",
    "duration",
    "wrist_speed_x",
    "wrist_speed_y",
    "index_speed_x",
    "index_speed_y",
    "pinch_min",
    "pinch_max",
]


# =========================================================
# CSV 파일 자동 생성
# =========================================================

def create_csv_if_needed():
    if os.path.exists(CSV_PATH):
        return

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + FEATURE_NAMES)

    print(f"CSV 자동 생성 완료: {CSV_PATH}")


# =========================================================
# 두 관절 사이 거리
# =========================================================

def calculate_distance(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    dz = point1.z - point2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


# =========================================================
# 현재 프레임에서 필요한 좌표 추출
# =========================================================

def extract_frame_data(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]

    pinch_distance = calculate_distance(thumb_tip, index_tip)

    return {
        "wrist_x": wrist.x,
        "wrist_y": wrist.y,
        "index_x": index_tip.x,
        "index_y": index_tip.y,
        "pinch_distance": pinch_distance,
        "timestamp": time.time(),
    }


# =========================================================
# 녹화 구간 전체에서 특징 14개 계산
# (Jetson 추론 코드와 완전히 동일해야 함)
# =========================================================

def calculate_segment_features(buffer):
    if len(buffer) < 2:
        return None

    first = buffer[0]
    last = buffer[-1]

    wrist_dx = last["wrist_x"] - first["wrist_x"]
    wrist_dy = last["wrist_y"] - first["wrist_y"]

    index_dx = last["index_x"] - first["index_x"]
    index_dy = last["index_y"] - first["index_y"]

    pinch_delta = last["pinch_distance"] - first["pinch_distance"]

    duration = last["timestamp"] - first["timestamp"]
    duration = max(duration, 0.001)

    pinch_speed = pinch_delta / duration

    wrist_speed_x = wrist_dx / duration
    wrist_speed_y = wrist_dy / duration

    index_speed_x = index_dx / duration
    index_speed_y = index_dy / duration

    pinch_distances = [d["pinch_distance"] for d in buffer]
    pinch_min = min(pinch_distances)
    pinch_max = max(pinch_distances)

    return [
        wrist_dx,
        wrist_dy,
        index_dx,
        index_dy,
        pinch_delta,
        pinch_speed,
        duration,
        wrist_speed_x,
        wrist_speed_y,
        index_speed_x,
        index_speed_y,
        pinch_min,
        pinch_max,
    ]


# =========================================================
# 특징을 CSV에 저장
# =========================================================

def save_sample(label, features):
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([label] + features)

    print(f"[저장 완료] {label}")


# =========================================================
# 현재 클래스별 데이터 개수 확인
# =========================================================

def load_class_counts():
    counts = {gesture: 0 for gesture in GESTURES.values()}

    if not os.path.exists(CSV_PATH):
        return counts

    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) == 0:
            return counts

        value_counts = df["label"].value_counts()
        for label, count in value_counts.items():
            counts[label] = int(count)
    except Exception:
        pass

    return counts


# =========================================================
# 화면 안내 출력
# =========================================================

def draw_guide(frame, recording, buffer_size, counts):
    if recording:
        status = f"RECORDING - frames: {buffer_size}"
        status_color = (0, 0, 255)
    else:
        status = "READY"
        status_color = (0, 255, 0)

    cv2.putText(frame, status, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    guide_text = [
        "SPACE : Start recording",
        "1 : Zoom In",
        "2 : Zoom Out",
        "3 : Scroll Up",
        "4 : Scroll Down",
        "5 : Move Left",
        "6 : Move Right",
        "7 : Click",
        "8 : Idle",
        "ESC : Exit",
    ]

    y_position = 65
    for text in guide_text:
        cv2.putText(frame, text, (20, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_position += 25

    count_text = (
        f"ZI:{counts['zoom_in']} "
        f"ZO:{counts['zoom_out']} "
        f"UP:{counts['scroll_up']} "
        f"DOWN:{counts['scroll_down']}"
    )
    cv2.putText(frame, count_text, (20, frame.shape[0] - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    count_text2 = (
        f"LEFT:{counts['move_left']} "
        f"RIGHT:{counts['move_right']} "
        f"CLICK:{counts['click']} "
        f"IDLE:{counts['idle']}"
    )
    cv2.putText(frame, count_text2, (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# =========================================================
# 데이터 수집 메인
# =========================================================

def collect():
    create_csv_if_needed()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    recording = False
    segment_buffer = []

    counts = load_class_counts()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while True:
            ret, frame = camera.read()
            if not ret:
                print("카메라 프레임 읽기 실패")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if recording:
                    segment_buffer.append(
                        extract_frame_data(hand_landmarks))

            draw_guide(frame, recording, len(segment_buffer), counts)

            cv2.imshow("Gesture Data Collection", frame)

            key = cv2.waitKey(1) & 0xFF

            # ESC 종료
            if key == 27:
                break

            # 스페이스바: 구간 녹화 시작
            if key == ord(" "):
                recording = True
                segment_buffer = []
                print()
                print("녹화 시작")
                print("동작을 수행한 후 1~8 키로 라벨을 지정하세요.")

            # 숫자 키: 녹화 종료 후 저장
            if key in GESTURES:
                if not recording:
                    print("먼저 SPACE를 눌러 녹화를 시작하세요.")
                    continue

                label = GESTURES[key]
                features = calculate_segment_features(segment_buffer)

                if features is None:
                    print("프레임이 부족합니다. 다시 수집하세요.")
                else:
                    save_sample(label, features)
                    counts[label] += 1

                recording = False
                segment_buffer = []

    camera.release()
    cv2.destroyAllWindows()

    print()
    print("수집 종료. 최종 클래스별 개수:")
    for name, cnt in load_class_counts().items():
        print(f"  {name}: {cnt}")
    print()
    print(f"-> {CSV_PATH} 를 Google Drive에 업로드한 뒤 Colab에서 학습을 진행하세요.")


if __name__ == "__main__":
    collect()