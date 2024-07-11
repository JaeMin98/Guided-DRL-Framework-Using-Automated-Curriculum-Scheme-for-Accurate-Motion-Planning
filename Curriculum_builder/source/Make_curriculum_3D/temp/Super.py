import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# csv 파일 로드
df = pd.read_csv('points7.csv')

# X, Y, Z를 입력 변수로, run_time, distance, delta를 목표 변수로 설정
x = df[['X', 'Y', 'Z']]
y = df[['run_time', 'distance', 'delta']]


# 데이터를 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# 선형 회귀 모델을 생성하고 학습
model = LinearRegression()
model.fit(X_train, y_train)


# 예측값 생성
y_pred = model.predict(X_test)

# Mean Squared Error 계산
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# 새로운 데이터에 대한 예측
new_data = [[0.1,0.1,0.8]]
predicted_output = model.predict(new_data)
print(predicted_output)
