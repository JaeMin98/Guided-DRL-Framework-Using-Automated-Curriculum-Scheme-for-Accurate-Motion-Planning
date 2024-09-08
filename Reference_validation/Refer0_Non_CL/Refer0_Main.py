import os
import sys
import csv
import random
import numpy as np
import Env
from sac import SAC
import torch
import Config
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
datacsv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../DataCSV'))
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Reference_models'))
sys.path = sys.path + [parent_dir, datacsv_dir, models_dir]

import Validation_config as vc

def set_pwd():
    current_file_path = os.path.abspath(__file__)
    current_file_directory = os.path.dirname(current_file_path)
    parent_directory = os.path.dirname(current_file_directory)

    os.chdir(parent_directory)

def load_curriculum():
    # CSV 파일 열기
    level_point = []
    for i in range(1,6):
        with open('./DataCSV/level_'+str(i)+'.csv', 'r') as file:
            reader = csv.reader(file)

            # 각 행들을 저장할 리스트 생성
            rows = []

            for row in reader:
                row_temp = row[:3]
                rows.append(row_temp)
            level_point.append(rows)

    # Level 3,4 바꾸기
    CSV_change = level_point[2]
    level_point[2] = level_point[3]
    level_point[3] = CSV_change

    return level_point

def run(max_t=200, evaluate=True):
    env = Env.Ned2_control()
    agent = SAC(12, 3, Config)

    checkpoint = torch.load('./Reference_models/Non_CL_SAC/model_NoneCL.tar', map_location=lambda storage, loc: storage)
    agent.policy.load_state_dict(checkpoint['model'])
    agent.policy.eval()

    level_point = load_curriculum()

    success_list = [[] for _ in range(vc.Num_of_UoC)]

    for UoC_level in range(vc.Num_of_UoC):
        for i_episode in range(vc.iteration_per_UoC):
            random_index_list = random.choice(range(len(level_point[UoC_level])))
            target = level_point[UoC_level][random_index_list]
            target = [float(element) for element in target] # 목표 지점 위치

            env.target = target

            env.reset()
            states = env.get_state()

            for timestep in range(max_t):
                actions = agent.select_action(np.array(states), evaluate=evaluate)

                env.action(actions)
                time.sleep(Config.time_sleep_interval)
                next_state, reward, done, success = env.observation()

                states = next_state

                if np.any(done):
                    break

            success_list[UoC_level].append(success)

    return success_list

if __name__ == "__main__":
    print(run())