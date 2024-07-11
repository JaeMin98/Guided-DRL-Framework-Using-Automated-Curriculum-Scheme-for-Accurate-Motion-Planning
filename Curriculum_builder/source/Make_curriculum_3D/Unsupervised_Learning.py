import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans, BisectingKMeans, MiniBatchKMeans, Birch, AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import SAMPLING
colors = ['red', 'green', 'blue','black','orange','purple','cyan','magenta']


class Unsupervised_Learning():
    def KMeans(self, X, k):
        print("\n\n----KMeans----")
        # KMeans 모델 초기화
        kmeans = KMeans(n_clusters=k)

        # 모델 학습
        kmeans.fit(X)

        # 클러스터링된 결과 확인
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        return kmeans, centroids, labels
    
    def BisectingKMeans(self, X, k):
        print("\n\n----BisectingKMeans----")
        # BisectingKMeans 모델 초기화
        bisecting_kmeans = BisectingKMeans(n_clusters=k)

        # 모델 학습
        bisecting_kmeans.fit(X)

        # 클러스터링된 결과 확인
        labels = bisecting_kmeans.labels_
        centroids = bisecting_kmeans.cluster_centers_

        return bisecting_kmeans, centroids, labels
    
    def MiniBatchKMeans(self, X, k):
        print("\n\n----MiniBatchKMeans----")
        # MiniBatchKMeans 모델 초기화
        mini_batch_kmeans = MiniBatchKMeans(n_clusters=k)

        # 모델 학습
        mini_batch_kmeans.fit(X)

        # 클러스터링된 결과 확인
        labels = mini_batch_kmeans.labels_
        centroids = mini_batch_kmeans.cluster_centers_

        return mini_batch_kmeans, centroids, labels
    
    def Birch(self, X, k):
        print("\n\n----Birch----")
        # BIRCH 모델 초기화
        birch = Birch(n_clusters=k)

        # 모델 학습
        birch.fit(X)

        # 클러스터링된 결과 확인
        labels = birch.labels_
        cluster_centers = birch.subcluster_centers_

        return birch, cluster_centers, labels
    
    def Ward(self, X, k):
        print("\n\n----Ward----")
        # Ward 클러스터링 모델 초기화
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward')

        # 모델 학습
        ward.fit_predict(X)

        # 클러스터링된 결과 확인
        labels = ward.labels_

        return ward

    def GaussianMixture(self, X, k):
        print("\n\n----GaussianMixture----")
        #full, tied, diag, spherical
        model = GaussianMixture(n_components=k, covariance_type='full')

        # 모델 학습 및 데이터에 대한 클러스터 할당
        model.fit(X)
        labels = model.predict(X)
        cluster_centers = model.means_

        return model, cluster_centers, labels
    
    def BayesianGaussianMixture(self, X, k):
        print("\n\n----BayesianGaussianMixture----")
        # Bayesian Gaussian Mixture 모델 생성
        # n_components: 추정할 클러스터 개수의 상한선을 지정합니다.
        #               적절한 클러스터 개수는 모델이 스스로 결정합니다.
        # covariance_type: 공분산 행렬의 타입을 지정합니다. GaussianMixture와 동일하게 작동합니다.
        model = BayesianGaussianMixture(n_components=k, covariance_type='full')

        # 모델 학습 및 데이터에 대한 클러스터 할당
        model.fit(X)
        labels = model.predict(X)
        cluster_centers = model.means_
        
        return model, cluster_centers, labels
    
    def get_model(self,X,k):
        KMeans = self.KMeans(X,k)
        BisectingKMeans = self.BisectingKMeans(X,k)
        MiniBatchKMeans = self.MiniBatchKMeans(X,k)
        Birch = self.Birch(X,k)
        GaussianMixture = self.GaussianMixture(X,k)
        BayesianGaussianMixture = self.BayesianGaussianMixture(X,k)
        # model7 = self.Ward(X,k)
        return KMeans, BisectingKMeans, MiniBatchKMeans, Birch, GaussianMixture, BayesianGaussianMixture



# sample = SAMPLING.SAMPLING()
# original_row, XYZ, feature = sample.get_sample("points7.csv")

# U = Unsupervised_Learning()
# k = U.get_model(feature,5)

# k = U.KMeans(feature,5)

# print(k.predict([[0.86386465 ,0.81574179]]))