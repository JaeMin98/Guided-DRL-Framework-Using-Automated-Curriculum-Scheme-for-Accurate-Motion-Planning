import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

class SAMPLING():
    def get_csv_data(self, csv_path):
        # CSV 파일 열기
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)

            # 각 행들을 저장할 리스트 생성
            original_row = []
            XYZ = []
            feature = []

            for row in reader:
                row_temp = row[:9]
                if(row_temp[0]=="X"):
                    continue
                row_temp = list(map(float, row_temp))

                row_xyz = list(map(float, row[:3]))
                delta = abs(float(row[6])) + abs(float(row[6])) + abs(float(row[6]))
                row_feature = list(map(float, [row[3],row[4],delta]))

                original_row.append(row_temp)
                XYZ.append(row_xyz)
                feature.append(row_feature)

        return original_row, XYZ, feature
    
    def split_list(self, data_list):
        split_data = [[row[i] for row in data_list] for i in range(len(data_list[0]))]
        return split_data
    
    def combine_list(self, split_data):
        data_list = [list(row) for row in zip(*split_data)]
        return data_list

    def normalization(self, feature):
        feature = self.split_list(feature)
        normalized_feature = []

        for F in feature:
            normalized_feature.append(self.normalize_list(F))
        
        normalized_feature = self.combine_list(normalized_feature)
        return normalized_feature
    
    def normalize_list(self, lst):
        # sorted_lst = sorted(lst)
        sorted_lst = lst
        
        mean_val = np.mean(sorted_lst)
        std_dev = np.std(sorted_lst)
        
        normalized_lst = [(x - mean_val) / std_dev for x in sorted_lst]
        
        return normalized_lst
    
    def get_sample(self, csv_path = "points7.csv"):
        original_row, XYZ, feature = self.get_csv_data(csv_path)
        feature = self.normalization(feature)

        return original_row, XYZ, feature


k = SAMPLING()
k.get_sample()