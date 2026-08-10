# data_collection/

로컬 PC(Windows)에서 실행하는 학습 데이터 수집 코드. 웹캠으로 손을 촬영해
MediaPipe로 관절 좌표를 뽑고, 특징을 계산해 CSV로 저장한다. 여기서 만든 CSV를
Google Drive에 올려 `training/`의 Colab 노트북에서 학습에 사용한다.

## 파일

| 파일 | 역할 | 결과물 |
|---|---|---|
| `collect_gesture_data.py` | 동작(움직임) 데이터 수집 | `gesture_features_index.csv` |
| `collect_pose_data.py` | 준비 자세(정지) 데이터 수집 | `pose_features.csv` |

## 환경 설정

```bash
conda create -n gesture python=3.12 -y
conda activate gesture
pip install mediapipe==0.10.14 opencv-python==4.9.0.80 numpy==1.26.4 pandas
```

버전을 반드시 명시해서 한 줄로 설치할 것 — 따로 설치하면 pip가 서로 다른 시점의
최신 버전을 참조해 의존성이 꼬인다.

## collect_gesture_data.py — 동작 데이터 수집

```bash
python collect_gesture_data.py
```

| 키 | 동작 |
|---|---|
| `SPACE` | 녹화 시작 (이 상태로 동작 수행) |
| `1`~`8` | 녹화 종료 + 해당 라벨로 저장 |
| `ESC` | 종료 |

라벨: `1` zoom_in · `2` zoom_out · `3` scroll_up · `4` scroll_down ·
`5` move_left · `6` move_right · `7` click · `8` idle

**수집 팁**
- 클래스당 100개씩. move는 완전히 수평으로, scroll은 완전히 수직으로만 움직일 것
  (대각선으로 새면 두 동작이 헷갈리는 데이터가 된다).
- 화면 위치·카메라 거리를 다양하게 바꿔가며 수집 (특징이 "변화량"이라 상쇄되지만
  MediaPipe 인식 안정성 확보에 도움).
- zoom/click은 손목을 고정하고 손가락만 움직일 것.

결과물 `gesture_features_index.csv`는 라벨 + 13개 특징(동작 구간의 시작·끝 좌표
변화량·속도 등)으로 구성된다. 껐다 켜도 이어서 누적 저장된다.

## collect_pose_data.py — 준비 자세 데이터 수집

```bash
python collect_pose_data.py
```

숫자 키를 **꾹 누르고 있으면** 그 자세로 연속 저장된다. 자세를 유지한 채 손
위치·각도·거리를 조금씩 바꿔가며 몇 초씩 나눠 누른다 (자세당 300장 정도면 충분).

| 키 | 자세 |
|---|---|
| `1` | fist (주먹 - 중립) |
| `2` | pinch_closed (엄지·검지 붙임) |
| `3` | pinch_open (엄지·검지 벌림) |
| `4` | index_ready (검지만 폄) |
| `5` | click_ready (따봉) |
| `ESC` | 종료 |

결과물 `pose_features.csv`는 라벨 + 63개 특징(21개 관절 × x,y,z, 손목 기준
상대좌표를 손 크기로 정규화한 값)으로 구성된다.

## 특징 계산 방식

두 스크립트 모두 좌표를 그대로 쓰지 않고 위치·거리 변화에 무관한 값으로 가공한다.
이 계산 로직(`calculate_segment_features`, `extract_pose_features`)은 `training/`의
학습 코드, `jetson_client/gesture_worker.py`의 추론 코드와 **완전히 동일해야** 한다.
셋 중 하나라도 다르면 학습된 모델이 실제 입력과 다른 형태의 데이터를 받게 되어
정확도가 무너진다.
