# -*- coding: utf-8 -*-
"""
main.py - Qt 손동작 인식 앱 진입점

[준비물 - 이 파일과 같은 폴더 (총 8개 파일)]
    gesture_weights.npz / scaler_params.json / labels.json      (동작 모델)
    pose_weights.npz / pose_labels.json                          (자세 모델)
    gesture_ui.ui                                                (화면 레이아웃)
    gesture_worker.py                                            (인식 백엔드)
    main_window.py                                               (화면 로직)

[사용법]
    source ~/tf241/bin/activate
    pip install PyQt5
    python3 main.py <서버IP> <포트> <내ID> [비밀번호]

    예: python3 main.py 10.10.16.39 5000 KMG_LIN
    (인자 없이 실행하면 아래 기본값 사용)
"""

import sys

from PyQt5.QtWidgets import QApplication

from main_window import MainWindow

DEFAULT_SERVER_HOST = "10.10.16.39"
DEFAULT_SERVER_PORT = 5000
DEFAULT_CLIENT_ID = "KMG_LIN"
DEFAULT_PASSWORD = "PASSWD"
DEFAULT_CAMERA_INDEX = 0


def parse_args():
    if len(sys.argv) == 1:
        return (DEFAULT_SERVER_HOST, DEFAULT_SERVER_PORT,
                DEFAULT_CLIENT_ID, DEFAULT_PASSWORD)
    if len(sys.argv) in (4, 5):
        host = sys.argv[1]
        port = int(sys.argv[2])
        client_id = sys.argv[3]
        password = sys.argv[4] if len(sys.argv) == 5 else DEFAULT_PASSWORD
        return host, port, client_id, password

    print(f"Usage: python3 {sys.argv[0]} <서버IP> <포트> <내ID> [비밀번호]")
    print("예: python3 main.py 10.10.16.39 5000 KMG_LIN")
    sys.exit(1)


def main():
    server_host, server_port, client_id, password = parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(
        server_host, server_port, client_id, password,
        camera_index=DEFAULT_CAMERA_INDEX,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
