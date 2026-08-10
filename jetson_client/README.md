# jetson_client/

Jetson Nano에서 실행하는 실시간 인식 클라이언트. 웹캠 → MediaPipe → 자세/동작 모델
추론 → 서버로 명령 전송까지 전부 담당하며, PyQt5로 만든 대시보드 화면에 인식 상태를
실시간으로 보여준다.

## 파일

| 파일 | 역할 |
|---|---|
| `main.py` | 진입점. 실행 인자(서버 IP/포트/ID/비밀번호) 처리 후 앱 실행 |
| `main_window.py` | 화면 로직. `.ui` 로드, `GestureWorker`의 시그널을 받아 화면 갱신만 담당 |
| `gesture_worker.py` | **인식 백엔드 (QThread)**. 카메라 캡처, 특징 계산, 2단 모델 추론, 서버 전송까지 전부 여기 있음 |
| `gesture_ui.ui` | Qt Designer로 만든 대시보드 레이아웃 (영상 + 상태 카드) |
| `gesture_weights.npz` / `scaler_params.json` / `labels.json` | 동작 모델 (Colab에서 학습됨) |
| `pose_weights.npz` / `pose_labels.json` | 자세 모델 (Colab에서 학습됨) |

인식 로직(모델, 임계값, 상태 머신)은 전부 `gesture_worker.py` 안에 있고,
`main_window.py`는 표시만 담당한다 (관심사 분리). `.ui` 파일 디자인을 바꾸고 싶으면
Qt Designer로 열어서 배치·색·크기를 자유롭게 수정해도 되는데, `main_window.py`가
`objectName`(`state_value`, `pose_value`, `armed_progress`, `fired_value`,
`server_value`, `quit_button`)으로 위젯을 찾아서 쓰기 때문에 **이 이름들만 유지**하면
코드 수정이 필요 없다.

## 환경 설정 (Jetson, tf241 venv 기준)

```bash
source ~/tf241/bin/activate
pip install mediapipe==0.10.14 tensorflow==2.16.1 opencv-python==4.9.0.80

sudo apt update
sudo apt install -y python3-pyqt5 pyqt5-dev-tools
```

PyQt5는 `pip`가 아니라 `apt`로 설치한다 — ARM(Jetson)용 PyQt5 pip wheel이 없어서
pip가 소스 빌드를 시도하다 `qmake` 없음 에러로 실패한다. `tf241` venv가 시스템
패키지에 접근 가능한 구조라, apt로 시스템에 설치하면 venv 안에서도 바로 보인다.

## 실행

```bash
cd jetson_client
python3 main.py <서버IP> <포트> <내ID> [비밀번호]
# 예: python3 main.py 10.10.16.39 5000 KMG_LIN
```

인자 없이 실행하면 `main.py` 상단의 기본값을 사용한다. 서버가 꺼져 있어도 프로그램은
죽지 않고 "서버 연결 실패 - 로컬 인식만 진행합니다"를 띄운 채 인식만 계속한다.

## 화면 표시 항목

| 표시 | 의미 |
|---|---|
| STATE | `IDLE_WATCH`(대기 중) / `RECORDING`(동작 구간 기록 중) |
| POSE | 확정된 준비 자세. 안정화 진행 중이면 "안정화 N/6" 표시 |
| ARMED | 동작 시작 시점에 고정된 자세 + 기록된 프레임 수 |
| LAST FIRED | 마지막으로 실제 서버에 전송된 동작 |
| SERVER | 서버 연결 상태 |

## gesture_worker.py 임계값 (튜닝 시 참고)

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `MOTION_START_THRESHOLD` | 0.014 | 이 값 이상 움직이면 동작 시작으로 판단 |
| `MOTION_STOP_THRESHOLD` | 0.010 | 이 값 이하로 멈추면 동작 종료 판단 시작 |
| `FRAMES_TO_CONFIRM_START` / `STOP` | 2 / 3 | 시작/종료 판정에 필요한 연속 프레임 수 |
| `CONFIDENCE_THRESHOLD` | 0.80 | 동작 모델 확신도 최소값 (미만이면 무시) |
| `POSE_CONF_THRESHOLD` | 0.80 | 자세 모델 확신도 최소값 |
| `POSE_SWITCH_FRAMES` | 4 | 자세를 바꾸려면 필요한 연속 프레임 수 (노이즈 방지) |
| `POSE_SETTLE_FRAMES` | 6 | 자세 확정 직후 움직임 감지를 무시하는 프레임 수 |

임계값은 기기(PC/Jetson)마다 카메라 노이즈 수준이 달라서 그대로 재사용하면 안 맞을
수 있다. `motion` 실측값을 status 로그로 확인하면서 조정하는 걸 권장한다.
