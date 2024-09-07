import datetime
import numpy as np
import itertools
import torch
from sac import SAC
from replay_memory import ReplayMemory
import Config
import Env
import time
import os
import sys
import gc
import csv
import wandb


def Run_Training():
    # 가비지 컬렉터 활성화
    gc.enable()
    # Environment
    env = Env.RobotArmControl()
    env.reset()

    # Agent
    agent = SAC(9, 3, Config)

    # Memory
    memory = ReplayMemory(Config.replay_size, Config.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    episode_success = []

    now = datetime.datetime.now()
    folderName = ('{}_SAC_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), Config.Current_Data_Selection_Ratio, Config.lr,
                                                                Config.Clustering_K, Config.Is_Clearing_Memory, Config.Success_Standard))

    success_rate = 0.0
    success_rate_list =[]

    episode_success_csv_data_list = []

    level_str = 'level_' + str(env.Level_Of_Point)

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        env.reset()
        state = env.get_state()

        while not done:
            if Config.start_steps > total_numsteps:
                random_values = np.random.uniform(-1, 1, size=3)
                random_list = random_values.tolist()
                action = random_list  # random action
            else:
                action = agent.select_action(state)  # action from policy

            env.action(action)
            time.sleep(Config.time_sleep_interval)
            next_state, reward, done, success = env.observation() # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            mask = 1 if episode_steps == Config.max_episode_steps else float(not done)

            memory.push(state, action, reward, next_state, mask) # Append transition to memory

            state = next_state

            if len(memory) > Config.batch_size:
                # Number of updates per step in environment
                for i in range(Config.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, Config.batch_size, updates)

                    updates += 1

            if done:
                episode_success.append(success)
                episode_success_csv_data_list.append([success,len(memory),total_numsteps,env.Level_Of_Point])

                success_rate = np.sum(episode_success[-min(10, len(episode_success)):])/10.0
                success_rate_list.append(success_rate)

                # 저장 경로 확인
                filePath = './models/' + folderName

                if not os.path.isdir(filePath):
                    os.makedirs(filePath)

                torch.save({
                    'model': agent.policy.state_dict(),
                    'optimizer': agent.policy_optim.state_dict()
                }, "./models/" + folderName + "/model_"+str(i_episode)+".tar")

                if(len(success_rate_list) > 4):
                    if (np.sum(success_rate_list[-min(5, len(success_rate_list)):])/5.0) >= Config.Success_Standard:
                        if not(env.Level_Of_Point >= env.MAX_Level_Of_Point):
                            episode_success = []
                            success_rate_list = []
                            if(Config.Is_Clearing_Memory):
                                memory = ReplayMemory(Config.replay_size, Config.seed) # memory reset
                            env.Level_Of_Point += 1
                            level_str = 'level_' + str(env.Level_Of_Point)

                        else:
                            # 최종 학습 파일 추출 추가하기
                            torch.save({
                                'model': agent.policy.state_dict(),
                                'optimizer': agent.policy_optim.state_dict()
                            }, "./models/" + folderName + "/model_final.tar")

                            # CSV 파일을 쓸 때 필요한 열 이름
                            fields = ['episode', 'Success', 'Replay Memory', 'total numsteps','level']
                            
                            if not os.path.isdir("./Episode_Success_Log"):
                                os.makedirs("./Episode_Success_Log")

                            filename = "./Episode_Success_Log/" + folderName + ".csv"  # 저장할 CSV 파일 이름

                            with open(filename, 'w', newline='') as csvfile:
                                CSVwriter = csv.writer(csvfile)
                                
                                # 열 이름을 CSV 파일에 작성
                                CSVwriter.writerow(fields)
                                
                                # 데이터를 CSV 파일에 작성
                                for index, value in enumerate(episode_success_csv_data_list):
                                    CSVwriter.writerow([index, value[0],value[1],value[2],value[3]])

                            # Log 파일 남기기 2
                            py_file = 'Config.py'  # .py 파일명
                            txt_file = "./Episode_Success_Log/" + folderName + ".txt"  # 저장할 CSV 파일 이름

                            # .py 파일을 읽어옵니다.
                            with open(py_file, 'r') as file:
                                content = file.read()

                            # .txt 파일에 내용을 저장합니다.
                            with open(txt_file, 'w') as file:
                                file.write(content)

                            sys.exit(0)

        if total_numsteps > Config.num_steps:
            break

        wandb.log({'Success_rate':success_rate}, step=i_episode)
        wandb.log({f'Success_rate_{level_str}':success_rate}, step=i_episode)
        wandb.log({'Score':episode_reward}, step=i_episode)

        print("Current_Level: {}, Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, Replay Memory length: {}".format(env.Level_Of_Point, i_episode, total_numsteps, episode_steps, round(episode_reward, 2), len(memory)))
        # 가비지 컬렉터 수동으로 실행
        gc.collect()
        try:
            wandb.log({'critic_1_loss':critic_1_loss}, step=i_episode)
            wandb.log({'alpha':alpha}, step=i_episode)
        except:
            continue


if __name__ == '__main__':

    for i in range(3):
        Config.Current_Data_Selection_Ratio = 0.8

        Config.time_sleep_interval = 0.05
        wandb.init(project='H2017')
        wandb.run.name = f'SAC_Robotic_Arm_{round(1.0-Config.Current_Data_Selection_Ratio,1)}_time_{Config.time_sleep_interval}'
        wandb.run.save()

        Run_Training()