# -*- coding: utf-8 -*-
"""
손동작 명령 수신 서버 (Ubuntu, 10.10.16.39 에서 실행)

Jetson(gesture 클라이언트)으로부터 소켓으로 동작 이름을 받아서,
pyautogui로 실제 키보드/마우스 동작을 수행합니다.

[동작 -> 실행 매핑]
    zoom_in      : Ctrl + '+'
    zoom_out     : Ctrl + '-'
    scroll_up    : ↑ (방향키) x20회 연속 (빠른 스크롤용)
    scroll_down  : ↓ (방향키) x20회 연속 (빠른 스크롤용)
    move_left    : ← (방향키)
    move_right   : → (방향키)
    click        : 마우스 좌클릭

[사용법 - Ubuntu 터미널]
    pip install pyautogui
    python3 gesture_server.py <port>

    예: python3 gesture_server.py 5000

[인증]
    같은 폴더의 idpasswd.txt 파일에서 "ID 비밀번호" 목록을 읽어옵니다.
    클라이언트가 접속하며 보내는 "[ID:비밀번호]" 메시지를 이 목록과 대조해서
    일치하지 않으면 연결을 거부합니다.

    idpasswd.txt 예시:
        KMG_LIN PASSWD

[프로토콜]
    클라이언트 접속 시 인증 메시지: "[클라이언트ID:PASSWD]"
    이후 동작 메시지:               "[GESTURE]동작이름"
"""

import socket
import threading
import sys
import time

import pyautogui

IDPASSWD_PATH = "idpasswd.txt"

# 스크롤 동작 시 방향키를 여러 번 누른 것처럼 동작 (웹툰 등 빠른 스크롤용)
SCROLL_REPEAT = 9
SCROLL_INTERVAL = 0.03  # 각 키 입력 사이 간격(초)

# pyautogui 안전장치: 마우스를 화면 구석으로 보내면 강제 중단됨 (필요시 False로)
pyautogui.FAILSAFE = True


def execute_gesture(gesture_name):
    """실제 키보드/마우스 동작 수행"""
    print(f"[실행] {gesture_name}")

    if gesture_name == "zoom_in":
        pyautogui.hotkey("ctrl", "+")
    elif gesture_name == "zoom_out":
        pyautogui.hotkey("ctrl", "-")
    elif gesture_name == "scroll_up":
        pyautogui.press("up", presses=SCROLL_REPEAT, interval=SCROLL_INTERVAL)
    elif gesture_name == "scroll_down":
        pyautogui.press("down", presses=SCROLL_REPEAT, interval=SCROLL_INTERVAL)
    elif gesture_name == "move_left":
        pyautogui.press("left")
    elif gesture_name == "move_right":
        pyautogui.press("right")
    elif gesture_name == "click":
        pyautogui.click()
    else:
        print(f"[알 수 없는 동작] {gesture_name}")


def load_idpasswd(path=IDPASSWD_PATH):
    """
    idpasswd.txt 로드. 한 줄에 "ID 비밀번호" 형식.
    예: KMG_LIN PASSWD
    """
    idpasswd = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    idpasswd[parts[0]] = parts[1]
    except FileNotFoundError:
        print(f"경고: {path} 파일이 없습니다. 인증이 전부 실패합니다.")
    return idpasswd


def parse_message(raw_text):
    """
    '[GESTURE]move_left' 같은 형식을 파싱.
    반환: (태그, 내용) 예: ('GESTURE', 'move_left')
    """
    results = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line or "]" not in line or not line.startswith("["):
            continue
        tag, _, payload = line[1:].partition("]")
        results.append((tag, payload))
    return results


def handle_client(conn, addr, idpasswd):
    print(f"[접속] {addr}")
    buffer = ""
    authenticated = False
    client_id = None

    try:
        while True:
            data = conn.recv(1024)
            if not data:
                print(f"[종료] {addr} 연결 끊김")
                break

            buffer += data.decode("utf-8", errors="ignore")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                for tag, payload in parse_message(line + "\n"):

                    if not authenticated:
                        # 최초 메시지는 반드시 "[ID:비밀번호]" 형식이어야 함
                        cid, sep, cpw = tag.partition(":")
                        if sep and idpasswd.get(cid) == cpw:
                            authenticated = True
                            client_id = cid
                            print(f"[인증 성공] {cid} ({addr})")
                            conn.send(f"[AUTH]OK\n".encode("utf-8"))
                        else:
                            print(f"[인증 실패] {tag} ({addr}) - 연결 종료")
                            conn.send(f"[AUTH]FAIL\n".encode("utf-8"))
                            conn.close()
                            return
                        continue

                    if tag == "GESTURE":
                        execute_gesture(payload.strip())
                    else:
                        print(f"[알 수 없는 태그] {tag}: {payload}")

    except ConnectionResetError:
        print(f"[종료] {addr} 강제 종료")
    finally:
        conn.close()
        if client_id:
            print(f"[연결 해제] {client_id} ({addr})")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} <port>")
        print("예: python3 gesture_server.py 5000")
        sys.exit(1)

    port = int(sys.argv[1])
    idpasswd = load_idpasswd()
    print(f"인증 등록 계정: {list(idpasswd.keys())}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", port))
    s.listen(5)
    print(f"서버 시작 - 포트 {port}에서 대기 중...")

    try:
        while True:
            conn, addr = s.accept()
            t = threading.Thread(
                target=handle_client, args=(conn, addr, idpasswd), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n서버 종료")
    finally:
        s.close()


if __name__ == "__main__":
    main()
