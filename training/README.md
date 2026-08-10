# training/

Google Colab에서 실행하는 모델 학습 노트북. `data_collection/`에서 만든 CSV를
입력받아 MLP를 학습하고, Jetson에서 그대로 불러올 수 있는 형태(numpy 가중치 +
JSON)로 저장한다.

## 파일

| 파일 | 입력 | 출력 |
|---|---|---|
| `train_gesture_colab.ipynb` | `gesture_features_index.csv` | `gesture_weights.npz`, `scaler_params.json`, `labels.json` |
| `train_pose_colab.ipynb` | `pose_features.csv` | `pose_weights.npz`, `pose_labels.json` |

## 실행 방법

1. CSV를 Google Drive의 `MyDrive/hand_gesture/`에 업로드 (경로가 다르면 노트북 상단의
   `CSV_PATH` 수정)
2. Colab에서 노트북을 열고 런타임을 GPU(T4)로 설정
3. `런타임 > 모두 실행` (Drive 마운트 시 계정 인증 클릭 필요)
4. 결과 확인 (정확도, Classification Report, Confusion Matrix)
5. `MyDrive/hand_gesture/model_output/`에 저장된 결과물을 `jetson_client/`로 복사

## 모델 구조

두 모델 모두 GPU 전용 특수 연산이 없는 `Dense`만 사용하는 MLP다. 이 구조는
`jetson_client/gesture_worker.py`의 `build_model()` / `build_pose_model()`과
**한 줄도 다르지 않게 동일해야** 한다. 하나라도 다르면 가중치를 불러올 때
shape 에러가 난다.

```
# 동작 모델 (train_gesture_colab.ipynb)
Input(13) → Dense(64, relu) → Dropout(0.2)
          → Dense(32, relu) → Dropout(0.2)
          → Dense(16, relu)
          → Dense(8, softmax)

# 자세 모델 (train_pose_colab.ipynb)
Input(63) → Dense(64, relu) → Dropout(0.2)
          → Dense(32, relu)
          → Dense(5, softmax)
```

## 결과 (마지막 실행 기준)

| 모델 | 데이터 | 테스트 정확도 |
|---|---|---|
| 동작 모델 (8종) | 800개 (클래스당 100개) | 100% (테스트셋 160개 전부 정답) |
| 자세 모델 (5종) | 1,512개 | 100% (테스트셋 303개 전부 정답) |

Confusion Matrix도 두 모델 모두 완전한 대각행렬(오분류 0건)로 나왔다. 다만 이 값은
단일 사용자 데이터 기준이라, 다른 사용자에 대한 일반화 성능은 별도로 검증되지 않았다.

## 저장 방식이 특이한 이유

`model.save()`나 `.keras`가 아니라 `model.get_weights()` + `np.savez()`로 순수
numpy 배열만 저장한다. Keras 3에서 `.keras`/`.h5` 저장 포맷 자체가 바뀌어서, Colab의
최신 TensorFlow에서 저장한 모델을 Jetson의 구버전 TensorFlow(2.4.1)가 아예 못 읽는
문제가 있었기 때문이다 (자세한 내용은 루트 `README.md`의 트러블슈팅 참고). 같은
이유로 `StandardScaler`도 pkl 대신 mean/scale 숫자만 `scaler_params.json`으로
저장해서, Jetson에 sklearn을 설치할 필요 자체를 없앴다.
