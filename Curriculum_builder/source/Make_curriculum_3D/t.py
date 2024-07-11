import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import SAMPLING
import Unsupervised_Learning
import math


colors = ['red', 'green', 'blue','black','orange','purple','cyan','magenta']

def euclidean_distance(x1, y1, x2, y2):
    y1 = y1 * 1.5
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rank_list(lst):
    # 순위를 매길 리스트를 정렬하여 새로운 리스트 생성
    sorted_lst = sorted(lst)
    
    # 숫자를 순위로 매길 딕셔너리 생성
    rank_dict = {num: rank for rank, num in enumerate(sorted_lst, start=1)}  # start 값을 0으로 변경
    
    # 숫자의 순위를 담은 리스트 생성
    ranks = [rank_dict[num] for num in lst]
    
    return ranks

class plot():
    def get_plot_of_algorithm_type(self, feature, models, original_row):
        num_row, num_col = 3, 6
        fig, ax = plt.subplots(num_row, num_col, figsize=(30, 15))
        ax = ax.flatten()

        X = np.array(feature)

        MODEL_NAME = ["KMeans", "BisectingKMeans", "MiniBatchKMeans", "Birch", "GaussianMixture", "BayesianGaussianMixture"]

        for i in range(len(models)):
            plot_name = MODEL_NAME[i]

            if(plot_name == "GaussianMixture" or plot_name =="BayesianGaussianMixture"):
                labels = models[i].predict(feature)
            else:
                labels = models[i].labels_

            df = pd.DataFrame({'X1': X[:, 0], 'X2': X[:, 1], 'X3': X[:, 2], 'Cluster': labels})

            ax = fig.add_subplot(num_row, num_col, i + 1, projection='3d')  # 3D 서브플롯 생성
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=0.5, cmap='rainbow')  # 3D 산점도 표시
            ax.set_title(plot_name)
            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('X3')

            # Save the results to a CSV file
            df.to_csv('clustered_data/' + plot_name + '.csv', index=False)

            # 스코어 계산
            # s_score = silhouette_score(X, labels)
            # c_score = calinski_harabasz_score(X, labels)
            # d_score = davies_bouldin_score(X, labels)
            s_score = 1.0
            c_score = 1.0
            d_score = 1.0
            # 표 데이터 준비
            data = [[s_score, c_score, d_score]]

            # 표 생성
            table = ax.table(cellText=data, loc='center', cellLoc='center', colLabels=["Silhouette", "Calinski-Harabasz", "Davies-Bouldin"], colColours=['lightgray']*3)

            # 표에 라벨 추가 (column header)
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.5)

            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)




    def get_plot_of_curriculum_unit(self, feature, models, original_row):
        num_row, num_col = 3, 6
        fig, ax = plt.subplots(num_row, num_col, figsize=(30, 15))
        ax = ax.flatten()

        X = np.array(feature)
        MODEL_NAME = ["KMeans", "MiniBatchKMeans", "GaussianMixture"]

        for i in range(num_row*num_col):
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            ax[i].spines['bottom'].set_visible(False)
            ax[i].spines['left'].set_visible(False)

        for i in range(len(models)):
            plot_name = MODEL_NAME[i]

            try:
                labels = models[i].labels_
                cluster_centers = models[i].cluster_centers_
            except:
                labels = models[i].predict(feature)
                cluster_centers = models[i].means_

            distance_list = []
            for center in cluster_centers:
                distance = euclidean_distance(center[0], center[1], -3, -3)
                distance_list.append(distance)
            cluster_sequence = rank_list(distance_list)


            #--------------------------------------------------------------------------------------------------------------------------------
            main_color = []
            for label in labels:
                main_color.append(colors[cluster_sequence[label]-1])
            axes = fig.add_subplot(3, 6, 1+i+(num_col-1)*i, projection='3d')

            X,Y,Z = [],[],[]
            for origin_roop in range(len(original_row)):
                X.append(original_row[origin_roop][0])
                Y.append(original_row[origin_roop][1])
                Z.append(original_row[origin_roop][2])

            axes.scatter(X, Y, Z, c=main_color, s=0.4, alpha =0.7, cmap='rainbow')
            axes.set_title(plot_name)

            x_lim = axes.get_xlim()
            y_lim = axes.get_ylim()
            z_lim = axes.get_zlim()

            #--------------------------------------------------------------------------------------------------------------------------------
            Unit_X,Unit_Y,Unit_Z = [[],[],[],[],[]],[[],[],[],[],[]],[[],[],[],[],[]]

            
            for k in range(len(labels)):
                Unit_X[int(labels[k])].append(original_row[k][0])
                Unit_Y[int(labels[k])].append(original_row[k][1])
                Unit_Z[int(labels[k])].append(original_row[k][2])

            for j in range(1,6):
                axes = fig.add_subplot(3, 6, 1+i+(num_col-1)*i + cluster_sequence[j-1], projection='3d')
                axes.scatter(Unit_X[j-1], Unit_Y[j-1], Unit_Z[j-1], c=colors[cluster_sequence[j-1]-1], s=0.9, alpha =0.7, cmap='rainbow')
                axes.set_title("level-"+str(cluster_sequence[j-1]))
                axes.set_xlim(x_lim[0],x_lim[1])
                axes.set_ylim(y_lim[0],y_lim[1])
                axes.set_zlim(z_lim[0],z_lim[1])
                
                # if j == 0:  # 첫 번째 서브플롯의 경우
                #     x_lim = axes.get_xlim()
                #     y_lim = axes.get_ylim()
                #     z_lim = axes.get_zlim()
                # else:
                #     print(i)
                #     axes.set_xlim(x_lim[0],x_lim[1])
                #     axes.set_ylim(y_lim[0],y_lim[1])
                #     axes.set_zlim(z_lim[0],z_lim[1])

        fig.suptitle('Unsupervised Learning Result')
        plt.show()


if __name__== "__main__" :
    sample = SAMPLING.SAMPLING()
    original_row, XYZ, feature = sample.get_sample("points7.csv")
    
    # for sublist in feature:
    #     sublist[1] *= 10

    U = Unsupervised_Learning.Unsupervised_Learning()
    KMeans, BisectingKMeans, MiniBatchKMeans, Birch, GaussianMixture, BayesianGaussianMixture = U.get_model(feature,5)

    algorithom_models = [KMeans, BisectingKMeans, MiniBatchKMeans, Birch, GaussianMixture, BayesianGaussianMixture]

    P = plot()
    P.get_plot_of_algorithm_type(feature, algorithom_models, original_row)

    curriculum_models = [KMeans, MiniBatchKMeans, GaussianMixture]
    P.get_plot_of_curriculum_unit(feature, curriculum_models, original_row)