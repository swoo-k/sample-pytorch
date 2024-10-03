import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 간단한 신경망 모델 정의
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 입력: 28x28 이미지 -> 128 노드
        self.fc2 = nn.Linear(128, 10)  # 출력: 10개의 클래스로 분류

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 이미지를 1차원으로 변환
        x = torch.relu(self.fc1(x))  # 활성화 함수: ReLU
        x = self.fc2(x)
        return x

# 데이터셋 준비 (MNIST)
def get_data_loader(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

# 모델 학습
def train_model():
    # 학습 설정
    model = SimpleModel()  # 모델 생성
    criterion = nn.CrossEntropyLoss()  # 손실 함수
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 옵티마이저 (SGD)
    
    # 데이터 로더
    train_loader = get_data_loader()
    
    # 학습 루프
    for epoch in range(1):  # 1 에폭만 실행 (테스트 용도)
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()  # 이전 배치의 그래디언트 초기화
            output = model(data)  # 모델에 데이터 입력
            loss = criterion(output, target)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            if batch_idx % 100 == 0:  # 100번째 배치마다 출력
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), 'simple_model.pth')
    print("모델 저장 완료: simple_model.pth")

if __name__ == "__main__":
    train_model()

