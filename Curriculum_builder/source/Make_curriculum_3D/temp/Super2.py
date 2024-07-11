import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import datetime
import csv

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # 모델을 평가 모드로 설정

# 데이터 로드
df = pd.read_csv('points7.csv')

# X, Y, Z를 입력 변수로, run_time, distance, delta를 목표 변수로 설정
X = df[['X', 'Y', 'Z']].values
y = df[['run_time', 'distance', 'delta']].values

# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# numpy 배열을 PyTorch 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 신경망 모델 구성
model = nn.Sequential(
    nn.Linear(3, 64),  # 입력 레이어 (3개의 입력을 받아 64개의 출력을 생성)
    nn.ReLU(),  # ReLU 활성화 함수
    nn.Linear(64, 32),  # 숨겨진 레이어 (64개의 입력을 받아 32개의 출력을 생성)
    nn.ReLU(),  # ReLU 활성화 함수
    nn.Linear(32, 3)  # 출력 레이어 (32개의 입력을 받아 3개의 출력을 생성)
)

# 손실 함수와 옵티마이저 설정
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 모델 학습
# for epoch in range(10000):  # 100 에포크동안 반복
#     # Forward pass
#     y_pred = model(X_train)
#     loss = loss_fn(y_pred, y_train)

#     # Backward pass and optimization
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(f"Epoch: {epoch}, Loss: {loss.item()}")
#     if(float(loss.item()) < 50):
#         break

# save_model(model,"./model2.pth")



# 예측과 성능 측정
# y_pred_test = model(X_test)
# mse = loss_fn(y_pred_test, y_test)
# print(f"Mean Squared Error: {mse.item()}")




load_model(model,"./model.pth")



new_data = torch.tensor([-0.05714298,0.012596336,0.77024561])

predict_csv_Xtrain = []
for temp_X in X_train:
    predicted_output = model(temp_X)
    predict_csv_Xtrain.append(temp_X.tolist()+predicted_output.tolist())
print(len(predict_csv_Xtrain))

predict_csv_Xtest = []
for temp_X in X_test:
    predicted_output = model(temp_X)
    predict_csv_Xtest.append(temp_X.tolist()+predicted_output.tolist())
print(len(predict_csv_Xtest))






def make_csv(predict_csv_Xtrain, predict_csv_Xtest):
    # CSV 파일 생성
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M")
    folder_path = './SuperVised_(' + formatted_datetime + ")"
    os.makedirs(folder_path, exist_ok=True)

    file_name = f'predict_csv_X_train.csv'
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in predict_csv_Xtrain:
            writer.writerow(row)    # CSV 데이터 쓰기

    file_name = f'predict_csv_X_test.csv'
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in predict_csv_Xtest:
            writer.writerow(row)    # CSV 데이터 쓰기

make_csv(predict_csv_Xtrain, predict_csv_Xtest)