# pyright: reportMissingImports=false, reportMissingModuleSource=false

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results_classification"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 실험 재현성을 높이기 위해 난수 시드를 고정한다.
tf.random.set_seed(42)

# MNIST 손글씨 숫자 데이터셋을 훈련/테스트 세트로 로드한다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 픽셀 값을 0~1 범위로 정규화하기 위해 float32로 변환한다.
x_train = x_train.astype(np.float32) / 255.0

# 테스트 데이터도 동일하게 정규화한다.
x_test = x_test.astype(np.float32) / 255.0

# 다중 분류 학습을 위해 레이블을 원-핫 인코딩으로 변환한다.
y_train_cat = to_categorical(y_train, num_classes=10)

# 테스트 레이블도 원-핫 인코딩으로 변환한다.
y_test_cat = to_categorical(y_test, num_classes=10)

# 손글씨 숫자 이미지는 28x28 흑백 이미지이므로 입력 형태를 그대로 유지한다.
input_shape = (28, 28)

# 아주 간단한 완전연결 신경망 모델을 생성한다.
model = Sequential([
    # 입력 텐서의 형태를 명시해 모델 시작점을 정의한다.
    Input(shape=input_shape),
    # 28x28 이미지를 1차원 벡터로 펼쳐 완전연결층에 전달한다.
    Flatten(),
    # 은닉층에서 비선형 표현을 학습한다.
    Dense(128, activation="relu"),
    # 과적합을 줄이기 위해 중간 은닉층을 하나 더 둔다.
    Dense(64, activation="relu"),
    # 10개 숫자 클래스에 대한 확률을 출력한다.
    Dense(10, activation="softmax"),
])

# 다중 클래스 분류에 맞게 모델을 컴파일한다.
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 구조를 텍스트 파일에 저장하기 위한 요약 정보를 준비한다.
model_summary_lines = []

# 모델 구조를 콘솔과 파일에 남기기 위해 요약 문자열을 수집한다.
model.summary(print_fn=model_summary_lines.append)

# 학습 로그를 저장할 히스토리 객체를 준비한다.
history = model.fit(
    x_train,
    y_train_cat,
    epochs=5,
    batch_size=128,
    validation_split=0.1,
    verbose=2,
)

# 테스트 데이터로 최종 성능을 평가한다.
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)

# 테스트 데이터에 대한 예측 확률을 계산한다.
probabilities = model.predict(x_test, verbose=0)

# 가장 높은 확률의 클래스를 최종 예측값으로 변환한다.
predictions = np.argmax(probabilities, axis=1)

# 중간 결과물로 학습 곡선을 시각화하기 위한 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 왼쪽에는 학습/검증 정확도 변화를 그린다.
plt.subplot(1, 2, 1)

# 학습 정확도 추이를 그린다.
plt.plot(history.history["accuracy"], label="Train Accuracy")

# 검증 정확도 추이를 그린다.
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

# 그래프 제목을 설정한다.
plt.title("Accuracy Curve")

# x축 레이블을 설정한다.
plt.xlabel("Epoch")

# y축 레이블을 설정한다.
plt.ylabel("Accuracy")

# 범례를 표시한다.
plt.legend()

# 오른쪽에는 학습/검증 손실 변화를 그린다.
plt.subplot(1, 2, 2)

# 학습 손실 추이를 그린다.
plt.plot(history.history["loss"], label="Train Loss")

# 검증 손실 추이를 그린다.
plt.plot(history.history["val_loss"], label="Validation Loss")

# 그래프 제목을 설정한다.
plt.title("Loss Curve")

# x축 레이블을 설정한다.
plt.xlabel("Epoch")

# y축 레이블을 설정한다.
plt.ylabel("Loss")

# 범례를 표시한다.
plt.legend()

# 중간 결과 그래프의 레이아웃을 정리한다.
plt.tight_layout()

# 중간 결과물인 학습 곡선을 저장한다.
plt.savefig(str(RESULT_DIR / "classification_training_curve.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# 테스트 샘플의 예측 결과를 시각화할 Figure를 생성한다.
plt.figure(figsize=(12, 8))

# 처음 16개의 테스트 이미지를 예측 예시로 사용한다.
for index in range(16):
    # 4x4 그리드의 각 위치를 계산한다.
    plt.subplot(4, 4, index + 1)

    # 테스트 이미지를 표시한다.
    plt.imshow(x_test[index], cmap="gray")

    # 실제 정답과 예측값을 제목에 함께 표시한다.
    plt.title(f"T:{y_test[index]} P:{predictions[index]}")

    # 축 눈금을 제거한다.
    plt.axis("off")

# 결과 이미지의 배치를 정리한다.
plt.tight_layout()

# 최종 결과물로 예측 예시 이미지를 저장한다.
plt.savefig(str(RESULT_DIR / "classification_prediction_grid.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# 모델 구조와 학습/평가 정보를 텍스트로 정리한다.
summary_lines = [
    "MNIST Simple Classifier",
    "",
    "Model Summary:",
    *model_summary_lines,
    "",
    f"Training samples: {x_train.shape[0]}",
    f"Test samples: {x_test.shape[0]}",
    f"Final training accuracy: {history.history['accuracy'][-1]:.4f}",
    f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}",
    f"Test loss: {test_loss:.4f}",
    f"Test accuracy: {test_accuracy:.4f}",
]

# 요약 텍스트 파일을 저장한다.
(RESULT_DIR / "classification_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# 모델도 함께 저장해 재사용 가능하게 한다.
model.save(RESULT_DIR / "classification_model.keras")

# 콘솔에 데이터셋 크기를 출력한다.
print(f"Training samples: {x_train.shape[0]}")

# 콘솔에 테스트 샘플 수를 출력한다.
print(f"Test samples: {x_test.shape[0]}")

# 콘솔에 최종 학습 정확도를 출력한다.
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")

# 콘솔에 최종 검증 정확도를 출력한다.
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

# 콘솔에 테스트 손실을 출력한다.
print(f"Test loss: {test_loss:.4f}")

# 콘솔에 테스트 정확도를 출력한다.
print(f"Test accuracy: {test_accuracy:.4f}")

# 콘솔에 결과 저장 위치를 출력한다.
print(f"Saved results to: {RESULT_DIR}")
