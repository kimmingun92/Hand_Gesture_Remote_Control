# -*- coding: utf-8 -*-
"""
gesture_worker.py - 손동작 인식 백엔드 (QThread)

jetson_realtime_stream.py 의 인식 로직을 그대로 옮기되,
cv2.imshow/putText 대신 Qt 시그널로 프레임과 상태를 GUI(main_window.py)에 전달합니다.

특징 계산, 모델 구조, 임계값은 기존 검증된 값과 완전히 동일합니다.
"""

import json
import math
import time
import socket
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

from PyQt5.QtCore import QThread, pyqtSignal


# =========================================================
# 파일 경로
# =========================================================

WEIGHTS_PATH = "gesture_weights.npz"
SCALER_PATH = "scaler_params.json"
LABELS_PATH = "labels.json"
POSE_WEIGHTS_PATH = "pose_weights.npz"
POSE_LABELS_PATH = "pose_labels.json"


# =========================================================
# 임계값 (jetson_realtime_stream.py 와 동일 - 여기서 튜닝)
# =========================================================

POSE_CONF_THRESHOLD = 0.80

MOTION_START_THRESHOLD = 0.014
MOTION_STOP_THRESHOLD = 0.010
FRAMES_TO_CONFIRM_START = 2
FRAMES_TO_CONFIRM_STOP = 3
MIN_SEGMENT_FRAMES = 4
MAX_SEGMENT_SECONDS = 2.5
CONFIDENCE_THRESHOLD = 0.80
PRE_ROLL_FRAMES = 6
FINAL_STABILITY_RANGE = 0.015

POSE_SWITCH_FRAMES = 4
POSE_SETTLE_FRAMES = 6

ALLOWED_ACTIONS = {
    "fist":              ["idle"],
    "pinch_closed":      ["idle", "zoom_in"],
    "pinch_open":        ["idle", "zoom_out"],
    "index_ready":       ["idle", "move_left", "move_right", "scroll_up", "scroll_down"],
    "click_ready":       ["idle", "click"],
    None:                ["idle"],
}

FEATURE_NAMES = [
    "wrist_dx", "wrist_dy", "index_dx", "index_dy",
    "pinch_delta", "pinch_speed", "duration",
    "wrist_speed_x", "wrist_speed_y", "index_speed_x", "index_speed_y",
    "pinch_min", "pinch_max",
]

NUM_LANDMARKS = 21
POSE_FEATURE_DIM = NUM_LANDMARKS * 3  # 63


# =========================================================
# 공통 유틸 (jetson_realtime_stream.py 와 완전히 동일)
# =========================================================

def calculate_distance(point1, point2):
    dx = point1.x - point2.x
    dy = point1.y - point2.y
    dz = point1.z - point2.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


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


def instant_motion(prev, curr):
    return (
        abs(curr["wrist_x"] - prev["wrist_x"])
        + abs(curr["wrist_y"] - prev["wrist_y"])
        + abs(curr["index_x"] - prev["index_x"])
        + abs(curr["index_y"] - prev["index_y"])
        + abs(curr["pinch_distance"] - prev["pinch_distance"])
    )


def is_final_position_stable(tail_frames, max_range=FINAL_STABILITY_RANGE):
    if len(tail_frames) < 2:
        return False
    for key in ("wrist_x", "wrist_y", "index_x", "index_y"):
        values = [f[key] for f in tail_frames]
        if (max(values) - min(values)) > max_range:
            return False
    return True


def extract_pose_features(hand_landmarks):
    joint = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks.landmark):
        joint[i] = [lm.x, lm.y, lm.z]

    wrist = joint[0].copy()
    rel = joint - wrist
    hand_size = np.linalg.norm(joint[9] - wrist) + 1e-6
    rel = rel / hand_size
    return rel.flatten()


def build_pose_model(num_classes):
    return Sequential([
        Input(shape=(POSE_FEATURE_DIM,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])


def detect_ready_pose(pose_model, pose_labels, hand_landmarks):
    feat = extract_pose_features(hand_landmarks)[None, :]
    proba = pose_model.predict(feat, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    conf = float(proba[pred_idx])

    pose = pose_labels[pred_idx] if conf >= POSE_CONF_THRESHOLD else None

    diag = {
        "pose_conf": conf,
        "pose_top": pose_labels[pred_idx],
    }
    return pose, diag


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
        wrist_dx, wrist_dy, index_dx, index_dy,
        pinch_delta, pinch_speed, duration,
        wrist_speed_x, wrist_speed_y, index_speed_x, index_speed_y,
        pinch_min, pinch_max,
    ]


def build_model(num_classes):
    return Sequential([
        Input(shape=(len(FEATURE_NAMES),)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])


def load_everything():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = {int(k): v for k, v in raw.items()}
    num_classes = len(labels)

    with open(SCALER_PATH, "r", encoding="utf-8") as f:
        sp = json.load(f)
    scaler_mean = np.array(sp["mean"], dtype=np.float32)
    scaler_scale = np.array(sp["scale"], dtype=np.float32)

    model = build_model(num_classes)
    data = np.load(WEIGHTS_PATH)
    weights = [data[f"arr_{i}"] for i in range(len(data.files))]
    model.set_weights(weights)

    with open(POSE_LABELS_PATH, "r", encoding="utf-8") as f:
        raw_pose = json.load(f)
    pose_labels = {int(k): v for k, v in raw_pose.items()}

    pose_model = build_pose_model(len(pose_labels))
    pose_data = np.load(POSE_WEIGHTS_PATH)
    pose_weights = [pose_data[f"arr_{i}"] for i in range(len(pose_data.files))]
    pose_model.set_weights(pose_weights)

    return model, labels, scaler_mean, scaler_scale, pose_model, pose_labels


def predict_gesture_masked(model, labels, scaler_mean, scaler_scale,
                            features, allowed_names):
    x = np.array([features], dtype=np.float32)
    x = (x - scaler_mean) / scaler_scale
    proba = model.predict(x, verbose=0)[0]

    name_to_idx = {v: k for k, v in labels.items()}
    allowed_idx = [name_to_idx[n] for n in allowed_names if n in name_to_idx]

    masked = np.zeros_like(proba)
    for i in allowed_idx:
        masked[i] = proba[i]

    total = masked.sum()
    if total <= 1e-9:
        return "idle", 0.0, proba

    masked_norm = masked / total
    pred_idx = int(np.argmax(masked_norm))
    return labels[pred_idx], float(masked_norm[pred_idx]), proba


# =========================================================
# QThread - 인식 백엔드
# =========================================================

class GestureWorker(QThread):
    """
    카메라 캡처 + mediapipe + 자세/동작 모델 + 서버 전송을 전부 담당하는 스레드.
    GUI(main_window.py)는 아래 시그널만 받아서 화면을 갱신하면 됨.
    """

    # 매 프레임: OpenCV BGR numpy 배열 (랜드마크가 이미 그려진 상태)
    frame_ready = pyqtSignal(np.ndarray)

    # 상태 갱신용 딕셔너리:
    #   state, pose, armed_pose, armed_frames, motion, pose_conf, pose_top,
    #   settling, settle_progress, settle_total,
    #   last_result, last_conf, last_fired, server_connected
    status_update = pyqtSignal(dict)

    # 콘솔에 남기던 로그 문자열 (원하면 GUI에도 표시 가능)
    log_message = pyqtSignal(str)

    def __init__(self, server_host, server_port, client_id, client_password,
                 camera_index=0, parent=None):
        super().__init__(parent)
        self.server_host = server_host
        self.server_port = server_port
        self.client_id = client_id
        self.client_password = client_password
        self.camera_index = camera_index
        self._running = True
        self.server_sock = None

    def stop(self):
        """GUI 종료 시 호출 - 루프를 안전하게 빠져나가게 함"""
        self._running = False

    # ---------------------------------------------------
    # 서버 연결
    # ---------------------------------------------------

    def connect_to_server(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((self.server_host, self.server_port))
            s.settimeout(None)
            s.send(f"[{self.client_id}:{self.client_password}]\n".encode("utf-8"))
            self.log_message.emit(f"서버 연결 성공: {self.server_host}:{self.server_port}")
            return s
        except Exception as e:
            self.log_message.emit(f"서버 연결 실패 ({e}) - 로컬 인식만 진행합니다.")
            return None

    def send_gesture(self, gesture_name):
        if self.server_sock is None:
            return
        try:
            self.server_sock.send(f"[GESTURE]{gesture_name}\n".encode("utf-8"))
        except Exception as e:
            self.log_message.emit(f"[전송 실패] {e}")

    # ---------------------------------------------------
    # 메인 루프
    # ---------------------------------------------------

    def run(self):
        model, labels, scaler_mean, scaler_scale, pose_model, pose_labels = load_everything()
        self.log_message.emit("모델 로드 완료")

        self.server_sock = self.connect_to_server()

        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils

        camera = cv2.VideoCapture(self.camera_index)
        if not camera.isOpened():
            self.log_message.emit("웹캠을 열 수 없습니다.")
            return

        state = "IDLE_WATCH"
        recent_frames = deque(maxlen=PRE_ROLL_FRAMES + 1)
        segment_buffer = []
        prev_frame = None

        start_streak = 0
        stop_streak = 0
        ignore_next_segment = False

        current_pose = None
        raw_pose = None
        candidate_pose = None
        candidate_streak = 0
        frames_since_pose_change = 0
        armed_pose = None
        need_fist_return = False

        live_label, live_conf = "-", 0.0
        last_fired_label = "-"
        diag = {}
        motion = 0.0

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:

            while self._running:
                ret, frame = camera.read()
                if not ret:
                    self.log_message.emit("카메라 프레임을 읽지 못했습니다.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    curr_frame = extract_frame_data(hand_landmarks)
                    recent_frames.append(curr_frame)

                    motion = 0.0
                    if prev_frame is not None:
                        motion = instant_motion(prev_frame, curr_frame)

                    if state == "IDLE_WATCH":
                        new_raw, diag = detect_ready_pose(
                            pose_model, pose_labels, hand_landmarks)
                        raw_pose = new_raw

                        if new_raw is None or new_raw == current_pose:
                            candidate_pose = None
                            candidate_streak = 0
                        else:
                            if new_raw == candidate_pose:
                                candidate_streak += 1
                            else:
                                candidate_pose = new_raw
                                candidate_streak = 1

                            if candidate_streak >= POSE_SWITCH_FRAMES:
                                current_pose = candidate_pose
                                candidate_pose = None
                                candidate_streak = 0
                                frames_since_pose_change = 0
                                if current_pose == "fist":
                                    need_fist_return = False

                        frames_since_pose_change += 1
                        pose_settled = frames_since_pose_change >= POSE_SETTLE_FRAMES

                        if pose_settled and motion >= MOTION_START_THRESHOLD:
                            start_streak += 1
                        else:
                            start_streak = 0

                        if start_streak >= FRAMES_TO_CONFIRM_START:
                            state = "RECORDING"
                            segment_buffer = list(recent_frames)
                            armed_pose = current_pose
                            stop_streak = 0
                            start_streak = 0

                    else:  # RECORDING
                        segment_buffer.append(curr_frame)

                        if motion <= MOTION_STOP_THRESHOLD:
                            stop_streak += 1
                        else:
                            stop_streak = 0

                        segment_duration = (
                            curr_frame["timestamp"] - segment_buffer[0]["timestamp"]
                        )

                        tail_stable = False
                        if stop_streak >= FRAMES_TO_CONFIRM_STOP:
                            tail_frames = segment_buffer[-FRAMES_TO_CONFIRM_STOP:]
                            tail_stable = is_final_position_stable(tail_frames)

                        should_finish = (
                            tail_stable
                            or segment_duration >= MAX_SEGMENT_SECONDS
                        )

                        if should_finish:
                            if len(segment_buffer) >= MIN_SEGMENT_FRAMES:
                                if need_fist_return:
                                    self.log_message.emit(
                                        "[무시] 주먹 복귀 대기 중")
                                elif ignore_next_segment:
                                    self.log_message.emit(
                                        "[무시] 복귀 동작으로 판단, 판단 생략")
                                    ignore_next_segment = False
                                else:
                                    features = calculate_segment_features(segment_buffer)
                                    allowed = ALLOWED_ACTIONS.get(armed_pose, ["idle"])

                                    live_label, live_conf, _ = predict_gesture_masked(
                                        model, labels, scaler_mean, scaler_scale,
                                        features, allowed)

                                    self.log_message.emit(
                                        f"[구간 종료] 자세={armed_pose} "
                                        f"-> 예측: {live_label} ({live_conf*100:.1f}%)")

                                    if (live_conf >= CONFIDENCE_THRESHOLD
                                            and live_label != "idle"):
                                        last_fired_label = live_label
                                        self.log_message.emit(
                                            f"[FIRED] {live_label} "
                                            f"({live_conf*100:.1f}%)")
                                        self.send_gesture(live_label)
                                        need_fist_return = True
                                    else:
                                        self.log_message.emit(
                                            f"[무시] 확신도 부족 또는 idle "
                                            f"({live_label}, {live_conf*100:.1f}%)")
                            else:
                                self.log_message.emit(
                                    f"[버림] 너무 짧은 구간 "
                                    f"({len(segment_buffer)}개)")

                            state = "IDLE_WATCH"
                            segment_buffer = []
                            stop_streak = 0

                    prev_frame = curr_frame
                else:
                    state = "IDLE_WATCH"
                    segment_buffer = []
                    recent_frames.clear()
                    prev_frame = None
                    start_streak = 0
                    stop_streak = 0
                    current_pose = None
                    raw_pose = None
                    candidate_pose = None
                    candidate_streak = 0
                    frames_since_pose_change = 0
                    need_fist_return = False
                    ignore_next_segment = False

                # ---- GUI로 프레임 + 상태 전달 ----
                self.frame_ready.emit(frame)

                self.status_update.emit({
                    "state": state,
                    "pose": current_pose,
                    "armed_pose": armed_pose,
                    "armed_frames": len(segment_buffer) if state == "RECORDING" else 0,
                    "motion": motion,
                    "pose_conf": diag.get("pose_conf", 0.0),
                    "pose_top": diag.get("pose_top", "-"),
                    "settling": frames_since_pose_change < POSE_SETTLE_FRAMES,
                    "settle_progress": frames_since_pose_change,
                    "settle_total": POSE_SETTLE_FRAMES,
                    "need_fist_return": need_fist_return,
                    "last_result": live_label,
                    "last_conf": live_conf,
                    "last_fired": last_fired_label,
                    "server_connected": self.server_sock is not None,
                })

        camera.release()
        if self.server_sock is not None:
            self.server_sock.close()
        self.log_message.emit("인식 스레드 종료")
