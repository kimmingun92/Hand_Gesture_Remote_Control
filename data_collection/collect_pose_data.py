# -*- coding: utf-8 -*-
"""
준비 자세(포즈) 데이터 수집 코드

동작 인식과 별개로, "정지 상태의 준비 자세"를 분류하는 두 번째 모델용
데이터를 수집합니다. 규칙 기반 자세 판별이 손 모양/각도 차이로 계속
어긋나는 문제를, 실제 본인 손 데이터 학습으로 해결하기 위함입니다.

[특징]
    21개 관절의 손목 기준 상대좌표를 손 크기로 정규화한 63개 값.
    -> 손이 화면 어디에 있든, 카메라와 거리가 어떻든 "자세 모양"만 남음.

[사용법 - Anaconda Prompt]
    conda activate gesture
    python collect_pose_data.py

[조작법]
    숫자 키를 "꾹 누르고 있으면" 그 자세로 연사 저장됩니다.
    자세를 유지한 채 손 위치/각도를 조금씩 바꿔가며 몇 초씩 누르세요.

    1 : fist          (주먹 - 중립)
    2 : pinch_closed  (엄지·검지 붙임 - zoom_in 준비)
    3 : pinch_open    (엄지·검지 벌림 - zoom_out 준비)
    4 : index_ready   (검지 자세 - move/scroll 준비)
    5 : click_ready   (따봉 - click 준비)
    ESC : 종료

[결과물]
    pose_features.csv -> Google Drive에 올려 train_pose_colab.ipynb 로 학습
"""

import csv
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

CSV_PATH = "pose_features.csv"

POSES = {
    ord("1"): "fist",
    ord("2"): "pinch_closed",
    ord("3"): "pinch_open",
    ord("4"): "index_ready",
    ord("5"): "click_ready",
}

NUM_LANDMARKS = 21
FEATURE_NAMES = []
for i in range(NUM_LANDMARKS):
    FEATURE_NAMES += [f"x{i}", f"y{i}", f"z{i}"]  # 63개


def extract_pose_features(hand_landmarks):
    """
    21개 관절 -> 손목 기준 상대좌표 -> 손 크기(손목~중지MCP)로 정규화 -> 63개 값.
    Jetson 추론 코드와 완전히 동일해야 함.
    """
    joint = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        joint[i] = [lm.x, lm.y, lm.z]

    wrist = joint[0].copy()
    rel = joint - wrist
    hand_size = np.linalg.norm(joint[9] - wrist) + 1e-6
    rel = rel / hand_size
    return rel.flatten()  # (63,)


def create_csv_if_needed():
    if os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + FEATURE_NAMES)
    print(f"CSV 자동 생성 완료: {CSV_PATH}")


def load_class_counts():
    counts = {p: 0 for p in POSES.values()}
    if not os.path.exists(CSV_PATH):
        return counts
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) == 0:
            return counts
        for label, count in df["label"].value_counts().items():
            counts[label] = int(count)
    except Exception:
        pass
    return counts


def main():
    create_csv_if_needed()

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    counts = load_class_counts()

    csv_file = open(CSV_PATH, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_landmarks = None
            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            saved_label = None
            if key in POSES and hand_landmarks is not None:
                label = POSES[key]
                features = extract_pose_features(hand_landmarks)
                writer.writerow([label] + features.tolist())
                counts[label] += 1
                saved_label = label

            # ---- 화면 표시 ----
            cv2.putText(frame, "Hold 1-5 to record pose / ESC to quit",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "1:fist 2:pinch_closed 3:pinch_open",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, "4:index_ready 5:click_ready",
                        (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            if saved_label:
                cv2.putText(frame, f"SAVING: {saved_label}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            y = frame.shape[0] - 20
            counts_text = " ".join(f"{k}:{v}" for k, v in counts.items())
            cv2.putText(frame, counts_text,
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Pose Data Collection", frame)

    csv_file.close()
    camera.release()
    cv2.destroyAllWindows()

    print()
    print("수집 종료. 최종 클래스별 개수:")
    for name, cnt in load_class_counts().items():
        print(f"  {name}: {cnt}")
    print()
    print(f"-> {CSV_PATH} 를 Google Drive에 업로드한 뒤 train_pose_colab.ipynb 로 학습하세요.")


if __name__ == "__main__":
    main()