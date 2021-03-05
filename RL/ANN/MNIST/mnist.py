# 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np

# gpu가 있는 경우 딥러닝 연산을 gpu로 수행, 그렇지 않은 경우 cpu로 수행
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 파라미터 설정
batch_size = 128            # 한번 학습시 128개의 데이터를 통해 학습
num_epochs = 10             # 모든 데이터에 대해 10번 학습 수행
learning_rate = 0.00025     # 학습 속도 결정
                            # 너무 값이 작으면, 학습 속도가 느림
                            # 너무 값이 크면, 최적으로 학습하지 못합

# MNIST 데이터 다운로드
trn_dataset = datasets.MNIST("./mnist_data/",
                             download=True,
                             train=True,    # 학습 데이터로 사용
                             transform=transforms.Compose([
                                 transforms.ToTensor()]))  # Pytorch 텐서(Tensor)의 형태로 데이터 출력
val_dataset = datasets.MNIST("./mnist_data/",
                              download=False,
                              train=False,
                              transform=transforms.Compose([transforms.ToTensor()]))

# DataLoader 설정
trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# CNN 네트워크
class CNNClassifier(nn.Module):
    def __init__(self):
        # 네트워크 연산에 사용할 구조 설정
        super(CNNClassifier, self).__init__()   # 항상 torch.nn.Module을 상속받고 시작

        # Conv2d(입력의 채널, 출력의 채널, Kernel size, Stride)
        self.conv1 = nn.Conv2d(1, 16, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)

        # Linear(입력 노드의 수, 출력 노드의 수)
        self.fc1 = nn.Linear(64*4*4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10) 

    def forward(self, x):
        # 네트워크 연산 수행

        # Convolution 연산 후 ReLU 연산
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Convolution의 연산 결과 (batch_sizex(64*4*4))를 (batch_sizex64*4*4)로 변환
        x = x.view(x.size(0), -1)

        # Fully connected 연산 수행 후 ReLU 연산
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Fully connected 연산
        x = self.fc3(x)

        # 최종 결과에 Softmax 연산 수행
        # - 출력 결과를 확률로 변환(합=1이 되도록)
        # - 입력에 대한 결과의 확률으 ㄹ알 수 있
        return F.softmax(x, dim=1)
    
# 정확도 도출 함수
# - y: 네트워크의 연산 결과
#   - 각 숫자에 대한 확률을 나타냄
#   - 하나의 입력에 대해 10개의 값을 가짐: batch_sizex10
# - label: 실제 결과
#   - 현재 입력이 어떤 숫자인지의 값을 보여줌: batch_sizex1
def get_accuracy(y, label):
    # argmax 연산을 통해 확률 중 가장 큰 값의 인덱스를 반환하여 label과 동일한 형식으로 변환
    y_idx = torch.argmax(y, dim=1)
    result = y_idx-label

    # 모든 입력에 대해 정답을 맞춘 갯수를 전체 개수로 나눠주어 정확도를 반환
    num_correct = 0
    for i in range(len(result)):
        if result[i] == 0:
            num_correct += 1
    return num_correct / y.shape[0]


# 네트워크, 손실함수, 최적화기 선언

# 네트워크 정의
# - CNNClassifier 클래스 호출
# - 설정한 device에서 딥러닝 네트워크 연산을 하도록 설정
cnn = CNNClassifier().to(device)

# 솔신 함수 설정
# - Cross Entropy 함수: 분류 문제에서 많이 사용하는 손실함수
criterion = nn.CrossEntropyLoss()

# 최적화기(Optimizer) 설정
# - 딥러닝 학습에서 주로 사용하는 Adam Optimizer 사용
# - cnn 네트워크의 파라미터 학습, 학습률 설정
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
# 한 epoch에 대한 전체 미니 배치의 
num_batches = len(trn_loader)


# 학습 및 검증 수행

# 학습 수행
for epoch in range(num_epochs):     # epoch 반복문
    # 학습시 손실함수 값과 정확도를 기록하기 위한 리스트
    trn_loss_list = []
    trn_acc_list = []
    # 1 epoch 연산을 위한 반복문
    # - data: 각 배치로 나누어진 데이터와 정답
    for i, data in enumerate(trn_loader):
        # 데이터 처리
        cnn.train()     # 네트워크를 학습을 위한 모드로 설정

        # 학습 데이터(x: 입력, label: 정답)를 받아온 후, device에 올려줌
        x, label = data
        X = x.to(device)
        label = label.to(device)

        # 네트워크 연산 및 손실함수 계산
        model_output = cnn(x)   # 네트워크 연산 수행 후 출력값 도출
                                # 입력: x, 출력: model_output

        loss = criterion(model_output, label)   # 손실함수 값 계산
                                                # 네트워크 연산 결과와 실제 결과를
                                                # cross entropy 연산하여 손실함수 값 도출

        # 네트워크 업데이트
        optimizer.zero_grad()   # 학습 수행 전 미분값을 0으로 초기화
                                # 학습 전에 꼭 수행
        loss.backward()         # 가중치 W와 b에 대한 기울기 계산
        optimizer.step()       # 가중치와 편향 업데이트

        # 학습 정확도 및 손실함수 값 기록
        trn_acc = get_accuracy(model_output, label) # 네트워크의 연산 결과와 실제 정답 결과를 비교하여 정확도 도출
        trn_loss_list.append(loss.item())           # 손실함수 값을 trn_loss_list에 추가
                                                    # item: 하나의 값으로 된 tensor를 일반 값으로 바꿔줌
        trn_acc_list.append(trn_acc)                # 정확도 값을 trn_acc_list에 추가
        
        # 검증 수행
        # 학습 진행 상황 출력 및 검증셋 연산 수행
        if (i+1) % 100 == 0:    # 매 100번째 미니배치 연산마다 진행상황 출력
            cnn.eval()          # 네트워크를 검증 모드로 설정
            with torch.no_grad():   # 학습에 사용하지 않는 코드들은 해당 블록 내에 기입
                # 검증시 손실함수 값과 정확도를 저장하기 위한 리스
                val_loss_list = []
                val_acc_list = []

                # 검증셋에 대한 연산 수행
                for j, val in enumerate(val_loader):
                    val_x, val_label = val
                    
                    val_x = val_x.to(device)
                    val_label = val_label.to(device)

                    val_output = cnn(val_x)

                    val_loss = criterion(val_output, val_label)
                    val_acc = get_accuracy(val_output, val_label)

                    val_loss_list.append(val_loss.item())
                    val_acc_list.append(val_acc)
            
                # 학습 및 검증 과정에 대한 진행상황 출력
                print("epoch: {}/{} | step: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | trn acc: {:.4f} | val acc: {:.4f}".format(
                    epoch+1, num_epochs, i+1, num_batches, np.mean(trn_loss_list), np.mean(val_loss_list), np.mean(trn_acc_list), np.mean(val_acc_list)))