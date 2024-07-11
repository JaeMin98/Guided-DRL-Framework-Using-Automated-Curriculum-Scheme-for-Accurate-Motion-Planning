import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
from scipy.stats import rankdata
import os
import datetime
colors = ['red', 'green', 'blue','black','orange','purple','cyan','magenta']

# CSV 파일 열기
with open('points7.csv', 'r') as file:
    reader = csv.reader(file)

    # 각 행들을 저장할 리스트 생성
    rows = []
    row_XYZ = []
    row_DD = []

    for row in reader:
        row_temp = row[:9]
        if(row_temp[0]=="X"):
            continue
        row_temp = list(map(float, row_temp))

        row_xyz = row[:3]
        row_xyz = list(map(float, row_xyz))

        run_dis_delta = [row[3],row[4]]
        run_dis_delta = list(map(float, run_dis_delta))

        rows.append(row_temp)
        row_XYZ.append(row_xyz)
        
        row_DD.append(run_dis_delta)


temp_normalized_list = []
for i in range(len(row_DD[0])):
    my_list = [row[i] for row in row_DD]
    min_L = min(my_list)
    max_L = max(my_list)
    normalized_list = [(x - min_L) / (max_L - min_L) for x in my_list]
    temp_normalized_list.append(normalized_list)

transposed_row_DD = [[row[i] for row in temp_normalized_list] for i in range(len(temp_normalized_list[0]))]
row_DD = transposed_row_DD

target = row_DD

# 훈련 데이터셋
train_data = np.array(target)

# k-means 클러스터링 모델 생성
n_clusters_num = 5
kmeans = KMeans(n_clusters=n_clusters_num)  # 클러스터 개수를 적절히 설정해야 함


# k-means 모델 학습
kmeans.fit(train_data)

centroids = kmeans.cluster_centers_

Kmeans_sequence = []
origin = np.array([0, 0])
for i in range(n_clusters_num):
    xy = np.array([centroids[i][0], centroids[i][1]])
    distance = np.linalg.norm(xy - origin)
    Kmeans_sequence.append(distance)
Kmeans_sequence = list(map(int,rankdata(Kmeans_sequence)))
print(Kmeans_sequence)

# # kmeans 그래프 생성
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# RunT = [row[0] for row in row_DD]
# Dis = [row[1] for row in row_DD]
# Delta = [row[2] for row in row_DD]
# Kmeans_color = []
# for i in range(len(RunT)):
#     Kmeans_color.append(colors[kmeans.predict([[row_DD[i][0],row_DD[i][1]]])[0]])
# ax.scatter(RunT, Dis,Delta, color = Kmeans_color,  s=5, alpha=0.3)  # 전체
# plt.show()












x = [row[0] for row in row_XYZ]
y = [row[1] for row in row_XYZ]
z = [row[2]-0.2 for row in row_XYZ]
color_index = []

for i in range(len(x)):
    color_index.append(kmeans.predict([[row_DD[i][0],row_DD[i][1]]])[0])


Rtime = [row[3] for row in rows]


def make_level_csv():
    global color_index, x, y, z, n_clusters_num,colors
    # 폴더 생성
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M")
    folder_path = './Data_Folder_(' + formatted_datetime + ")"
    os.makedirs(folder_path, exist_ok=True)


    # CSV 파일 생성
    for num in range(0,n_clusters_num):
        file_name = f'level_{Kmeans_sequence[num]}.csv'
        file_path = os.path.join(folder_path, file_name)

        tX,tY,tZ = [],[],[]
        for i in range(len(color_index)):
            if color_index[i] == num:
                tX.append(x[i])
                tY.append(y[i])
                tZ.append(z[i])

        tXYZ = [[tX[i],tY[i],tZ[i]] for i in range(len(tX))]

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for row in tXYZ:
                writer.writerow(row)    # CSV 데이터 쓰기

make_level_csv()


# 그래프 생성
fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')

minT = min(Rtime)
maxT = max(Rtime)

division = 5
tick = (maxT - minT)/(division)
Color_list = []

# 시간 순으로 점 표시
for i in range(len(Rtime)):
    cNum = 100
    for j in range(division):
        if(float(minT + tick*j) <= float(Rtime[i]) <= float(minT + tick*(j+1))):
            cNum = j
            break
    if(cNum == 100):
        Color_list.append("white")
    else:
        Color_list.append(colors[cNum])

ax.scatter(x, y, z,color=Color_list, s=1, alpha=0.3)  # 전체
plt.show()

# #색깔별 출력
# tempx, tempy, tempz, tempcolor = [],[],[],[]
# for i in range(division):
#     temp = []
#     for k in range(len(Color_list)):
#         if(Color_list[k]==colors[i]):
#             temp.append([x[k],y[k],z[k]])
#     tempx.append([row[0] for row in temp])
#     tempy.append([row[1] for row in temp])
#     tempz.append([row[2] for row in temp])
#     tempcolor.append([colors[i] for _ in range(len(tempx[i]))])


# for i in range(division):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.set_xlim(-0.5, 0.5)
#     ax.set_ylim(-0.5, 0.5)
#     ax.set_zlim(0, 0.75)
#     ax.scatter(tempx[i], tempy[i], tempz[i],color=colors[i], s=1, alpha=0.3)
#     plt.show()




# #점점 커지게
# for i in range(division-1):
#     tempx[i+1] += tempx[i]
#     tempy[i+1] += tempy[i]
#     tempz[i+1] += tempz[i]
#     tempcolor[i+1] += tempcolor[i]


# for i in range(division):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1, projection='3d')
#     ax.set_xlim(-0.5, 0.5)
#     ax.set_ylim(-0.5, 0.5)
#     ax.set_zlim(0, 0.75)
#     ax.scatter(tempx[i], tempy[i], tempz[i],color=tempcolor[i], s=1, alpha=0.3)
#     plt.show()



def makeSubplot(num):
    global color_index, x, y, z, n_clusters_num,colors
    tX,tY,tZ = [],[],[]
    for i in range(len(color_index)):
        if color_index[i] == num:
            tX.append(x[i])
            tY.append(y[i])
            tZ.append(z[i])

    Tnum = 1 + num
    ax = fig.add_subplot(3,3,Tnum, projection='3d')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 0.75)
    ax.scatter(tX, tY, tZ, c=colors[num], s=1, alpha=0.3)  # 전체

fig = plt.figure()
for i in range(0,n_clusters_num):
    makeSubplot(i)

plt.show()


def makeSubplot2(num, ax):
    global color_index, x, y, z, n_clusters_num,colors
    tX,tY,tZ = [],[],[]
    for i in range(len(color_index)):
        if color_index[i] == num:
            tX.append(x[i])
            tY.append(y[i])
            tZ.append(z[i])

    ax.scatter(tX, tY, tZ, c=colors[num], s=1, alpha=0.3)  # 전체

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0, 0.75)

for i in range(0,n_clusters_num):
    makeSubplot2(i, ax)

plt.show()



