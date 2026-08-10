# -*- coding: utf-8 -*-
"""
main_window.py - MainWindow 클래스

gesture_ui.ui 를 불러와서 위젯을 배치하고,
GestureWorker(gesture_worker.py)가 보내는 시그널을 받아 화면을 갱신한다.
"""

import os

import cv2
import numpy as np

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

from gesture_worker import GestureWorker

UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_ui.ui")

# 다크 테마 (mockup 1과 동일한 톤)
STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #1e1f24;
    color: #e8e8ea;
    font-size: 13px;
}
QLabel#video_label {
    background-color: #14151a;
    border: 1px solid #33343a;
    border-radius: 12px;
    color: #6b6c72;
    font-size: 14px;
}
QFrame#card_state, QFrame#card_pose, QFrame#card_armed,
QFrame#card_fired, QFrame#card_server {
    background-color: #2a2b31;
    border-radius: 12px;
}
QLabel#state_title, QLabel#pose_title, QLabel#armed_title,
QLabel#fired_title, QLabel#server_title {
    color: #8a8b91;
    font-size: 11px;
    font-weight: 500;
}
QLabel#state_value, QLabel#pose_value, QLabel#armed_value,
QLabel#fired_value {
    font-size: 18px;
    font-weight: 500;
    color: #e8e8ea;
}
QLabel#server_value {
    font-size: 14px;
    font-weight: 500;
}
QLabel#fired_icon {
    font-size: 24px;
}
QProgressBar#armed_progress {
    background-color: #14151a;
    border: none;
    border-radius: 3px;
}
QProgressBar#armed_progress::chunk {
    background-color: #4d8dff;
    border-radius: 3px;
}
QPushButton#quit_button {
    background-color: #2a2b31;
    border: 1px solid #3a3b41;
    border-radius: 8px;
    padding: 8px;
    font-weight: 500;
}
QPushButton#quit_button:hover {
    background-color: #35363c;
}
"""

# 동작별 화살표/아이콘 문자 (Tabler 폰트를 데스크톱에 따로 설치할 필요 없이
# 유니코드 문자로 간단히 표현. 원하면 QIcon으로 교체 가능)
ACTION_ICON = {
    "move_left": "←",
    "move_right": "→",
    "scroll_up": "↑",
    "scroll_down": "↓",
    "zoom_in": "+",
    "zoom_out": "-",
    "click": "●",
    "idle": "•",
    "-": "•",
}


class MainWindow(QMainWindow):

    def __init__(self, server_host, server_port, client_id, client_password,
                 camera_index=0):
        super().__init__()
        uic.loadUi(UI_PATH, self)
        self.setStyleSheet(STYLE_SHEET)

        self.worker = GestureWorker(
            server_host, server_port, client_id, client_password,
            camera_index=camera_index,
        )
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.status_update.connect(self.on_status_update)
        self.worker.log_message.connect(self.on_log_message)

        self.quit_button.clicked.connect(self.close)

        self.worker.start()

    # ---------------------------------------------------
    # 슬롯
    # ---------------------------------------------------

    def on_frame_ready(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # 버벅이면 Qt.SmoothTransformation -> Qt.FastTransformation 으로 바꾸면
        # 화질은 살짝 떨어지지만 CPU 부담이 줄어듭니다.
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def on_status_update(self, status: dict):
        # STATE
        state = status["state"]
        self.state_value.setText(state)
        self.state_value.setStyleSheet(
            "color: #ff5c5c;" if state == "RECORDING" else "color: #4d8dff;")

        # POSE
        pose = status["pose"] or "-"
        if status.get("need_fist_return"):
            self.pose_value.setText(f"{pose}  (주먹 복귀 필요)")
            self.pose_value.setStyleSheet("color: #ffab40;")
        elif status.get("settling"):
            p, t = status["settle_progress"], status["settle_total"]
            self.pose_value.setText(f"{pose}  (안정화 {p}/{t})")
            self.pose_value.setStyleSheet("color: #ffab40;")
        else:
            self.pose_value.setText(pose)
            self.pose_value.setStyleSheet("color: #b58cff;")

        # ARMED
        if state == "RECORDING":
            armed_pose = status["armed_pose"] or "-"
            frames = status["armed_frames"]
            self.armed_value.setText(f"{armed_pose} · {frames} frames")
            self.armed_progress.setValue(min(frames, self.armed_progress.maximum()))
        else:
            self.armed_value.setText("-")
            self.armed_progress.setValue(0)

        # LAST FIRED
        last_fired = status["last_fired"]
        self.fired_value.setText(last_fired)
        self.fired_icon.setText(ACTION_ICON.get(last_fired, "•"))

        # SERVER
        if status["server_connected"]:
            self.server_value.setText("connected")
            self.server_value.setStyleSheet("color: #4dd48a;")
        else:
            self.server_value.setText("not connected")
            self.server_value.setStyleSheet("color: #ff5c5c;")

    def on_log_message(self, message: str):
        # 콘솔에도 그대로 출력 (터미널에서 디버깅용으로 확인 가능)
        print(message)

    # ---------------------------------------------------
    # 종료 처리
    # ---------------------------------------------------

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()
