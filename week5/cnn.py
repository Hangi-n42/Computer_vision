from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageOps
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results_cnn"
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "cifar10"
DOG_IMAGE_PATH = SCRIPT_DIR.parent.parent / "images" / "dog.jpg"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# CUDA GPU 사용이 필수이므로 먼저 사용 가능 여부를 확인한다.
if not torch.cuda.is_available():
    # GPU를 사용할 수 없으면 과제 요구사항을 만족할 수 없으므로 중단한다.
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

# 학습에서 사용할 CUDA 장치를 선택한다.
device = torch.device("cuda")

# 실험 재현성을 위해 PyTorch 난수 시드를 고정한다.
torch.manual_seed(42)

# NumPy 난수 시드도 함께 고정한다.
np.random.seed(42)

# GPU 환경에서 추가적인 시드도 고정한다.
torch.cuda.manual_seed_all(42)

# cuDNN의 최적화된 경로를 허용해 학습 속도를 높인다.
torch.backends.cudnn.benchmark = True

# CIFAR-10의 채널별 평균을 설정한다.
cifar10_mean = (0.4914, 0.4822, 0.4465)

# CIFAR-10의 채널별 표준편차를 설정한다.
cifar10_std = (0.2470, 0.2435, 0.2616)

# 학습 데이터용 전처리 파이프라인을 정의한다.
train_transform = transforms.Compose([
    # 원본 32x32 이미지를 약간 여유 있게 패딩한다.
    transforms.RandomCrop(32, padding=4),
    # 좌우 반전을 적용해 일반화 성능을 높인다.
    transforms.RandomHorizontalFlip(),
    # 이미지를 텐서로 변환하고 0~1 범위로 맞춘다.
    transforms.ToTensor(),
    # 채널별 평균과 표준편차로 정규화한다.
    transforms.Normalize(cifar10_mean, cifar10_std),
    # 일부 영역을 무작위로 지워 과적합을 줄인다.
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
])

# 검증/테스트 데이터용 전처리 파이프라인을 정의한다.
test_transform = transforms.Compose([
    # 이미지를 텐서로 변환하고 0~1 범위로 맞춘다.
    transforms.ToTensor(),
    # 채널별 평균과 표준편차로 정규화한다.
    transforms.Normalize(cifar10_mean, cifar10_std),
])

# 시각화용 전처리 파이프라인을 정의한다.
display_transform = transforms.ToTensor()

# dog.jpg를 비율 유지 방식으로 32x32 텐서로 바꾸는 함수를 정의한다.
def preprocess_dog_image(image: Image.Image) -> torch.Tensor:
    # 원본 비율을 유지한 채 32x32 박스 안으로 축소한다.
    resized = ImageOps.contain(image, (32, 32), method=Image.Resampling.BICUBIC)

    # CIFAR-10 평균 색상에 가까운 배경 캔버스를 만든다.
    background_color = tuple(int(value * 255) for value in cifar10_mean)
    canvas = Image.new("RGB", (32, 32), color=background_color)

    # 가운데 정렬을 위해 좌상단 배치 좌표를 계산한다.
    offset_x = (32 - resized.width) // 2
    offset_y = (32 - resized.height) // 2

    # 축소한 원본 이미지를 캔버스 중앙에 붙여 letterbox를 완성한다.
    canvas.paste(resized, (offset_x, offset_y))

    # 학습 때와 동일한 정규화를 적용해 추론 텐서를 만든다.
    tensor = transforms.ToTensor()(canvas)
    tensor = transforms.Normalize(cifar10_mean, cifar10_std)(tensor)

    # 배치 차원을 추가하기 전 텐서를 반환한다.
    return tensor

# CIFAR-10의 클래스 이름을 준비한다.
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# 학습/검증 분할에 사용할 원본 훈련 데이터셋을 불러온다.
raw_train_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=None,
)

# 같은 인덱스를 사용하되 전처리만 다른 학습용 데이터셋을 만든다.
train_dataset_full = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=train_transform,
)

# 검증용 전처리를 적용한 데이터셋을 만든다.
val_dataset_full = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=True,
    download=False,
    transform=test_transform,
)

# 테스트용 데이터셋을 불러온다.
test_dataset = datasets.CIFAR10(
    root=str(DATA_DIR),
    train=False,
    download=True,
    transform=test_transform,
)

# 학습/검증 분할을 위해 전체 훈련 인덱스를 섞는다.
all_indices = torch.randperm(len(raw_train_dataset), generator=torch.Generator().manual_seed(42)).tolist()

# 훈련 샘플 수를 설정한다.
train_size = 45000

# 검증 샘플 수를 설정한다.
val_size = len(raw_train_dataset) - train_size

# 훈련 인덱스 구간을 분리한다.
train_indices = all_indices[:train_size]

# 검증 인덱스 구간을 분리한다.
val_indices = all_indices[train_size:train_size + val_size]

# 훈련 데이터셋에서 필요한 인덱스만 선택한다.
train_dataset = Subset(train_dataset_full, train_indices)

# 검증 데이터셋에서 필요한 인덱스만 선택한다.
val_dataset = Subset(val_dataset_full, val_indices)

# 배치 단위로 데이터를 읽기 위한 훈련 로더를 만든다.
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

# 배치 단위로 데이터를 읽기 위한 검증 로더를 만든다.
val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# 배치 단위로 데이터를 읽기 위한 테스트 로더를 만든다.
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

# 학습 중간 결과를 확인하기 위한 CIFAR-10 샘플 시각화를 만든다.
plt.figure(figsize=(10, 10))

# 처음 16개의 이미지를 4x4 격자로 배치한다.
for index in range(16):
    # 각 샘플을 꺼낸다.
    image, label = raw_train_dataset[index]

    # 4x4 그리드의 현재 칸을 선택한다.
    plt.subplot(4, 4, index + 1)

    # 원본 이미지를 출력한다.
    plt.imshow(image)

    # 클래스 이름을 제목으로 표시한다.
    plt.title(class_names[label])

    # 축 눈금을 제거해 이미지에 집중한다.
    plt.axis("off")

# 샘플 시각화의 배치를 정리한다.
plt.tight_layout()

# 데이터셋 중간 결과 이미지를 저장한다.
plt.savefig(str(RESULT_DIR / "cnn_dataset_preview.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# CIFAR-10용 간단한 합성곱 신경망을 정의한다.
class CIFAR10CNN(nn.Module):
    def __init__(self) -> None:
        # 부모 클래스 초기화를 수행한다.
        super().__init__()

        # 특징 추출을 담당하는 합성곱 블록을 정의한다.
        self.features = nn.Sequential(
            # 3채널 입력에서 64개 특징맵을 생성한다.
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            # 특징 분포를 안정화해 학습을 빠르게 만든다.
            nn.BatchNorm2d(64),
            # 비선형성을 부여해 표현력을 높인다.
            nn.ReLU(inplace=True),
            # 다시 한 번 합성곱을 적용해 저수준 특징을 더 추출한다.
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            # 특징 분포를 안정화해 학습을 빠르게 만든다.
            nn.BatchNorm2d(64),
            # 활성화를 통해 학습 능력을 높인다.
            nn.ReLU(inplace=True),
            # 공간 크기를 절반으로 줄여 계산량을 낮춘다.
            nn.MaxPool2d(kernel_size=2),
            # 과적합을 줄이기 위해 드롭아웃을 적용한다.
            nn.Dropout(p=0.25),
            # 더 많은 채널을 사용해 고수준 특징을 학습한다.
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            # 특징 분포를 안정화해 학습을 빠르게 만든다.
            nn.BatchNorm2d(128),
            # 비선형 변환을 이어간다.
            nn.ReLU(inplace=True),
            # 한 번 더 특징을 정제한다.
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            # 특징 분포를 안정화해 학습을 빠르게 만든다.
            nn.BatchNorm2d(128),
            # 활성화를 적용한다.
            nn.ReLU(inplace=True),
            # 공간 차원을 다시 절반으로 축소한다.
            nn.MaxPool2d(kernel_size=2),
            # 특징 과적합을 줄이기 위해 추가 드롭아웃을 넣는다.
            nn.Dropout(p=0.3),
            # 더 깊은 단계에서 고수준 형태 특징을 학습한다.
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            # 특징 분포를 안정화해 학습을 빠르게 만든다.
            nn.BatchNorm2d(256),
            # 비선형성을 적용한다.
            nn.ReLU(inplace=True),
            # 공간 차원을 다시 축소한다.
            nn.MaxPool2d(kernel_size=2),
            # 과적합을 줄이기 위해 드롭아웃을 넣는다.
            nn.Dropout(p=0.35),
        )

        # 최종 분류를 담당하는 완전연결 블록을 정의한다.
        self.classifier = nn.Sequential(
            # 펼쳐진 특징 벡터를 받아 고차원 표현을 학습한다.
            nn.Linear(256 * 4 * 4, 512),
            # 배치정규화로 분류기 입력 분포를 안정화한다.
            nn.BatchNorm1d(512),
            # 비선형성을 부여한다.
            nn.ReLU(inplace=True),
            # 분류기 단계에서 과적합을 줄인다.
            nn.Dropout(p=0.5),
            # 중간 분류 표현을 추가로 학습한다.
            nn.Linear(512, 256),
            # 비선형성을 부여한다.
            nn.ReLU(inplace=True),
            # 드롭아웃으로 일반화 성능을 높인다.
            nn.Dropout(p=0.4),
            # 10개 클래스의 로짓을 출력한다.
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 합성곱 블록으로부터 특징을 추출한다.
        x = self.features(x)

        # 합성곱 특징을 2차원 형태에서 1차원 벡터로 펼친다.
        x = torch.flatten(x, 1)

        # 완전연결 분류기로 최종 로짓을 계산한다.
        x = self.classifier(x)

        # 각 클래스의 점수를 반환한다.
        return x

# 모델 인스턴스를 생성하고 CUDA 장치로 옮긴다.
model = CIFAR10CNN().to(device)

# 분류 손실 함수로 교차 엔트로피를 사용한다.
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# 최적화 알고리즘으로 AdamW를 사용한다.
optimizer = optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)

# 코사인 스케줄로 학습률을 점진적으로 줄여 수렴을 돕는다.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# 학습 에포크 수를 설정한다.
epochs = 20

# 학습/검증 기록을 저장할 리스트를 준비한다.
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}

# 한 에포크 동안 학습을 수행하는 함수를 정의한다.
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    # 모델을 학습 모드로 전환한다.
    model.train()

    # 누적 손실을 저장한다.
    running_loss = 0.0

    # 정답 개수를 저장한다.
    correct = 0

    # 전체 샘플 수를 저장한다.
    total = 0

    # 데이터 로더에서 배치를 하나씩 꺼낸다.
    for images, labels in loader:
        # 입력과 레이블을 CUDA 장치로 옮긴다.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 이전 배치의 그래디언트를 초기화한다.
        optimizer.zero_grad(set_to_none=True)

        # 순전파를 수행해 로짓을 얻는다.
        outputs = model(images)

        # 손실 값을 계산한다.
        loss = criterion(outputs, labels)

        # 역전파를 수행한다.
        loss.backward()

        # 가중치를 업데이트한다.
        optimizer.step()

        # 현재 배치 손실을 누적한다.
        running_loss += loss.item() * images.size(0)

        # 예측 클래스를 계산한다.
        predictions = outputs.argmax(dim=1)

        # 맞춘 개수를 누적한다.
        correct += (predictions == labels).sum().item()

        # 본 샘플 수를 누적한다.
        total += labels.size(0)

    # 평균 손실을 계산해 반환한다.
    average_loss = running_loss / total

    # 정확도를 계산해 반환한다.
    accuracy = correct / total

    # 에포크 결과를 반환한다.
    return average_loss, accuracy

# 검증 또는 테스트를 수행하는 함수를 정의한다.
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    # 모델을 평가 모드로 전환한다.
    model.eval()

    # 누적 손실을 저장한다.
    running_loss = 0.0

    # 정답 개수를 저장한다.
    correct = 0

    # 전체 샘플 수를 저장한다.
    total = 0

    # 배치 단위로 평가를 수행한다.
    for images, labels in loader:
        # 입력과 레이블을 CUDA 장치로 옮긴다.
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 순전파만 수행한다.
        outputs = model(images)

        # 평가 손실을 계산한다.
        loss = criterion(outputs, labels)

        # 배치 손실을 누적한다.
        running_loss += loss.item() * images.size(0)

        # 예측 클래스를 계산한다.
        predictions = outputs.argmax(dim=1)

        # 맞춘 개수를 누적한다.
        correct += (predictions == labels).sum().item()

        # 샘플 수를 누적한다.
        total += labels.size(0)

    # 평균 손실을 계산한다.
    average_loss = running_loss / total

    # 정확도를 계산한다.
    accuracy = correct / total

    # 결과를 반환한다.
    return average_loss, accuracy

# 각 에포크마다 학습과 검증을 반복한다.
for epoch in range(1, epochs + 1):
    # 한 에포크 학습을 수행한다.
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, device)

    # 검증 데이터로 성능을 평가한다.
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

    # 학습률 스케줄러를 한 단계 진행시킨다.
    scheduler.step()

    # 학습 기록을 저장한다.
    history["train_loss"].append(train_loss)

    # 학습 정확도를 기록한다.
    history["train_accuracy"].append(train_accuracy)

    # 검증 손실을 기록한다.
    history["val_loss"].append(val_loss)

    # 검증 정확도를 기록한다.
    history["val_accuracy"].append(val_accuracy)

    # 현재 에포크의 진행 상황을 출력한다.
    print(
        f"Epoch {epoch}/{epochs} - "
        f"train_loss: {train_loss:.4f} - "
        f"train_accuracy: {train_accuracy:.4f} - "
        f"val_loss: {val_loss:.4f} - "
        f"val_accuracy: {val_accuracy:.4f}"
    )

# 테스트 세트로 최종 성능을 평가한다.
test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

# 테스트 세트 전체에 대한 예측을 수집한다.
all_test_probabilities = []

# 테스트 세트 전체에 대한 정답을 수집한다.
all_test_labels = []

# 모델을 평가 모드로 둔 채 예측을 수행한다.
model.eval()

# 그래디언트를 계산하지 않는다.
with torch.no_grad():
    # 테스트 배치 단위로 반복한다.
    for images, labels in test_loader:
        # 입력을 CUDA 장치로 이동한다.
        images = images.to(device, non_blocking=True)

        # 순전파를 수행해 로짓을 구한다.
        outputs = model(images)

        # 확률로 바꾸기 위해 softmax를 적용한다.
        probabilities = torch.softmax(outputs, dim=1)

        # CPU 메모리로 옮겨 저장한다.
        all_test_probabilities.append(probabilities.cpu())

        # 정답도 함께 저장한다.
        all_test_labels.append(labels)

# 확률 텐서를 하나로 합친다.
all_test_probabilities = torch.cat(all_test_probabilities, dim=0)

# 정답 텐서를 하나로 합친다.
all_test_labels = torch.cat(all_test_labels, dim=0)

# 최종 예측 클래스를 계산한다.
test_predictions = all_test_probabilities.argmax(dim=1)

# 전체 테스트 정확도를 확인한다.
computed_test_accuracy = (test_predictions == all_test_labels).float().mean().item()

# 혹시 모를 차이를 방지하기 위해 계산 정확도를 우선 사용한다.
test_accuracy = computed_test_accuracy

# 학습 곡선을 시각화할 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 왼쪽 그래프에 손실 곡선을 그린다.
plt.subplot(1, 2, 1)

# 학습 손실 추이를 그린다.
plt.plot(history["train_loss"], label="Train Loss")

# 검증 손실 추이를 그린다.
plt.plot(history["val_loss"], label="Validation Loss")

# 그래프 제목을 설정한다.
plt.title("CNN Loss Curve")

# x축 레이블을 설정한다.
plt.xlabel("Epoch")

# y축 레이블을 설정한다.
plt.ylabel("Loss")

# 범례를 표시한다.
plt.legend()

# 오른쪽 그래프에 정확도 곡선을 그린다.
plt.subplot(1, 2, 2)

# 학습 정확도 추이를 그린다.
plt.plot(history["train_accuracy"], label="Train Accuracy")

# 검증 정확도 추이를 그린다.
plt.plot(history["val_accuracy"], label="Validation Accuracy")

# 그래프 제목을 설정한다.
plt.title("CNN Accuracy Curve")

# x축 레이블을 설정한다.
plt.xlabel("Epoch")

# y축 레이블을 설정한다.
plt.ylabel("Accuracy")

# 범례를 표시한다.
plt.legend()

# 그래프 배치를 정리한다.
plt.tight_layout()

# 중간 결과물인 학습 곡선을 저장한다.
plt.savefig(str(RESULT_DIR / "cnn_training_curve.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# dog.jpg를 불러와 추론용 이미지를 준비한다.
dog_image = Image.open(DOG_IMAGE_PATH).convert("RGB")

# 비율 유지된 32x32 이미지를 추론 시작점으로 준비한다.
dog_base_image = ImageOps.contain(dog_image, (32, 32), method=Image.Resampling.BICUBIC)

# 평균 색상 배경의 32x32 캔버스를 준비한다.
dog_canvas = Image.new("RGB", (32, 32), color=tuple(int(value * 255) for value in cifar10_mean))

# 비율 유지된 이미지를 중앙에 배치한다.
dog_canvas.paste(dog_base_image, ((32 - dog_base_image.width) // 2, (32 - dog_base_image.height) // 2))

# TTA로 확률 안정화를 위해 다양한 변형 샘플을 생성한다.
tta_images = [
    dog_canvas,
    F.hflip(dog_canvas),
    F.rotate(dog_canvas, angle=7, interpolation=transforms.InterpolationMode.BICUBIC, fill=tuple(int(value * 255) for value in cifar10_mean)),
    F.rotate(dog_canvas, angle=-7, interpolation=transforms.InterpolationMode.BICUBIC, fill=tuple(int(value * 255) for value in cifar10_mean)),
    F.adjust_brightness(dog_canvas, 0.9),
    F.adjust_brightness(dog_canvas, 1.1),
    F.adjust_contrast(dog_canvas, 0.9),
    F.adjust_contrast(dog_canvas, 1.1),
]

# 각 변형에 대한 확률을 저장할 리스트를 준비한다.
tta_probabilities = []

# TTA 변형 각각에 대해 추론을 수행한다.
with torch.no_grad():
    # 변형 이미지를 하나씩 꺼낸다.
    for tta_image in tta_images:
        # 변형 이미지를 텐서로 바꿔 GPU로 옮긴다.
        dog_tensor = preprocess_dog_image(tta_image).unsqueeze(0).to(device)

        # 순전파를 수행해 로짓을 계산한다.
        dog_logits = model(dog_tensor)

        # softmax 확률을 계산해 저장한다.
        tta_probabilities.append(torch.softmax(dog_logits, dim=1)[0])

# 여러 변형의 확률 평균으로 최종 확률을 얻는다.
dog_probabilities = torch.stack(tta_probabilities, dim=0).mean(dim=0)

# 가장 높은 확률의 클래스를 예측값으로 선택한다.
dog_prediction_index = int(dog_probabilities.argmax().item())

# 예측 클래스 이름을 가져온다.
dog_prediction_label = class_names[dog_prediction_index]

# 상위 5개 확률과 클래스 인덱스를 찾는다.
top_probabilities, top_indices = torch.topk(dog_probabilities, k=5)

# dog.jpg 결과를 시각화할 Figure를 생성한다.
plt.figure(figsize=(12, 5))

# 왼쪽에 dog 이미지를 배치한다.
plt.subplot(1, 2, 1)

# 원본 이미지를 표시한다.
plt.imshow(dog_image)

# 제목에 예측 결과를 표시한다.
plt.title(f"Predicted: {dog_prediction_label}")

# 축 눈금을 제거한다.
plt.axis("off")

# 오른쪽에 상위 클래스 확률 막대그래프를 배치한다.
plt.subplot(1, 2, 2)

# 막대그래프를 수평으로 그린다.
plt.barh(
    [class_names[int(index)] for index in top_indices.flip(0).tolist()],
    top_probabilities.flip(0).cpu().numpy(),
)

# x축 범위를 0~1로 맞춘다.
plt.xlim(0, 1)

# 그래프 제목을 설정한다.
plt.title("Top-5 CIFAR-10 Probabilities")

# x축 레이블을 설정한다.
plt.xlabel("Probability")

# 레이아웃을 정리한다.
plt.tight_layout()

# 최종 결과물인 dog 예측 이미지를 저장한다.
plt.savefig(str(RESULT_DIR / "cnn_dog_prediction.png"), dpi=150)

# Figure 자원을 해제한다.
plt.close()

# 모델 구조를 텍스트로 정리한다.
model_lines = [str(model)]

# 실험 요약 정보를 기록할 리스트를 구성한다.
summary_lines = [
    "CIFAR-10 CNN Classifier",
    "",
    f"Device: {device}",
    f"GPU Name: {torch.cuda.get_device_name(0)}",
    "",
    "Model Architecture:",
    *model_lines,
    "",
    f"Train samples: {len(train_dataset)}",
    f"Validation samples: {len(val_dataset)}",
    f"Test samples: {len(test_dataset)}",
    f"Final train loss: {history['train_loss'][-1]:.4f}",
    f"Final train accuracy: {history['train_accuracy'][-1]:.4f}",
    f"Final validation loss: {history['val_loss'][-1]:.4f}",
    f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}",
    f"Test loss: {test_loss:.4f}",
    f"Test accuracy: {test_accuracy:.4f}",
    f"Dog image prediction: {dog_prediction_label}",
    "Top-5 dog image probabilities:",
]

# 상위 5개 예측 확률을 텍스트에 추가한다.
for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
    # 각 클래스와 확률을 한 줄씩 기록한다.
    summary_lines.append(f"  {class_names[int(index)]}: {probability:.4f}")

# 요약 텍스트 파일을 저장한다.
(RESULT_DIR / "cnn_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

# 학습된 모델 가중치를 저장한다.
torch.save(model.state_dict(), RESULT_DIR / "cnn_model.pth")

# 콘솔에 GPU 이름을 출력한다.
print(f"Device: {device}")

# 콘솔에 GPU 이름을 출력한다.
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# 콘솔에 학습 샘플 수를 출력한다.
print(f"Train samples: {len(train_dataset)}")

# 콘솔에 검증 샘플 수를 출력한다.
print(f"Validation samples: {len(val_dataset)}")

# 콘솔에 테스트 샘플 수를 출력한다.
print(f"Test samples: {len(test_dataset)}")

# 콘솔에 최종 학습 정확도를 출력한다.
print(f"Final train accuracy: {history['train_accuracy'][-1]:.4f}")

# 콘솔에 최종 검증 정확도를 출력한다.
print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")

# 콘솔에 테스트 정확도를 출력한다.
print(f"Test accuracy: {test_accuracy:.4f}")

# 콘솔에 dog 이미지 예측 결과를 출력한다.
print(f"Dog image prediction: {dog_prediction_label}")

# 콘솔에 상위 5개 클래스 확률을 출력한다.
for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
    # 클래스명과 확률을 함께 출력한다.
    print(f"Top probability - {class_names[int(index)]}: {probability:.4f}")

# 콘솔에 결과 저장 위치를 출력한다.
print(f"Saved results to: {RESULT_DIR}")
