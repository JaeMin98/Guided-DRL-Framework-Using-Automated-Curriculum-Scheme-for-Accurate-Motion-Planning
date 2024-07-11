import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = 'data_points/normalized_data_points.csv'
data = pd.read_csv(file_path)

# 필요한 컬럼 선택
cols = ['execution_time', 'distance', 'joint1_delta', 'joint2_delta', 'joint3_delta']
data = data[cols]

# 2D 그래프 생성
plt.figure(figsize=(12, 8))

# 산점도 그리기
sc = plt.scatter(data['execution_time'], data['distance'],
                 c=data['joint1_delta'], s=data['joint2_delta']*100,
                 alpha=0.6, cmap='viridis', edgecolors='w', linewidth=0.5)

# 컬러바 추가
cbar = plt.colorbar(sc)
cbar.set_label('Joint 1 Delta')

# 그래프 제목 및 축 레이블 설정
plt.title('5D Data Visualization')
plt.xlabel('Execution Time')
plt.ylabel('Distance')

plt.show()
