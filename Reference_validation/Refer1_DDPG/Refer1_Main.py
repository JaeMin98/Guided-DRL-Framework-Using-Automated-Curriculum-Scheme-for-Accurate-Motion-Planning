import os
import sys
import csv
import random
import numpy as np
import Env
from ddpg_agent import Agent 
import torch

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

def run(model_name, max_t=200, evaluate=True):
    if(evaluate) : add_noise = False
    else : add_noise = True

    env = Env.Ned2_control()
    agent = Agent(state_size=15, action_size=3, random_seed=123456)

    actor_model_path = model_name+"-actor.pth"
    critic_model_path = model_name+"-critic.pth"
    agent.actor_local.load_state_dict(torch.load(actor_model_path))
    agent.critic_local.load_state_dict(torch.load(critic_model_path))

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
            agent.reset()

            for timestep in range(max_t):
                actions = agent.act(np.array(states), add_noise=add_noise)

                next_states, rewards, dones, success = env.step(actions)  # 새로운 env.step() 반환 값에 맞춰 수정

                states = next_states

                if np.any(dones):
                    break

            success_list[UoC_level].append(success)

    return success_list

if __name__ == "__main__":
    print(run())