# server/

Ubuntu PC에서 실행하는 명령 수신 서버. Jetson(`jetson_client/`)이 소켓으로 보낸
동작 이름을 받아서 인증한 뒤, `pyautogui`로 실제 키보드/마우스 입력을 실행한다.

## 파일

| 파일 | 역할 |
|---|---|
| `gesture_server.py` | 소켓 서버. 인증 + 동작 실행 |
| `idpasswd.example.txt` | 인증 파일 예시. 실제 사용 시 `idpasswd.txt`로 복사해 값 수정 (`.gitignore` 처리됨) |

## 환경 설정

```bash
pip install pyautogui
```

## 인증 파일 설정

```bash
cp idpasswd.example.txt idpasswd.txt
```

`idpasswd.txt`에 `ID 비밀번호` 형식으로 한 줄에 하나씩 적는다 (구분자는 스페이스/탭
상관없음). 계정을 추가하려면 줄만 더 쓰면 되고, 개수 제한은 없다 (서버 재시작 필요).

```
KMG_LIN yourpassword
YGY_LIN yourpassword
```

클라이언트가 접속 시 보내는 `[ID:비밀번호]` 메시지를 이 목록과 대조해서, 일치하지
않으면 연결을 거부한다.

## 실행

```bash
python3 gesture_server.py <port>
# 예: python3 gesture_server.py 5000
```

연결 안 되면 방화벽 확인:

```bash
sudo ufw allow 5000
```

## 동작 → 실행 매핑

| 동작 | 실행되는 것 |
|---|---|
| `zoom_in` | `Ctrl` + `+` |
| `zoom_out` | `Ctrl` + `-` |
| `scroll_up` | `↑` 방향키 9회 연속 (0.03초 간격) |
| `scroll_down` | `↓` 방향키 9회 연속 (0.03초 간격) |
| `move_left` | `←` 1회 |
| `move_right` | `→` 1회 |
| `click` | 마우스 좌클릭 |

`SCROLL_REPEAT`(연속 입력 횟수)와 `SCROLL_INTERVAL`(입력 간격)은 `gesture_server.py`
상단에서 조정 가능. 앱이 키 입력 속도를 못 따라가면 `SCROLL_INTERVAL`을 늘린다.

## 프로토콜

| 방향 | 메시지 | 설명 |
|---|---|---|
| 클라이언트 → 서버 | `[ID:PASSWORD]\n` | 최초 인증 |
| 서버 → 클라이언트 | `[AUTH]OK\n` / `[AUTH]FAIL\n` | 인증 결과 |
| 클라이언트 → 서버 | `[GESTURE]scroll_down\n` | 동작 명령 |

## 안전장치

`pyautogui.FAILSAFE = True`로 설정되어 있어, 마우스를 화면 모서리로 보내면 실행 중인
동작이 강제 중단된다 (오동작 시 비상 정지용).
