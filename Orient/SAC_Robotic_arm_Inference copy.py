import datetime
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import Config
import Env
import time
import csv
import datetime

csv_data = [
    ['rate', 'Level', 'NoneCL', 'CL'],
]

csv_filename = 'TestModel/TestModel_Result_'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.csv'

model_type = [
              0.9]

model_compare = ["model_final_"]



Compare_level_Of_SuccessRate = []


# Environment
env = Env.Ned2_control()
env.reset()

# Agent
agent = SAC(12, 3, Config)

path = "TestModel/model_NoneCL.tar"

checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
agent.policy.load_state_dict(checkpoint['model'])

# Training Loop
total_numsteps = 0
updates = 0
iteration = 100
episode_success = []
temp_level_Of_SuccessRate = []
now = datetime.datetime.now()

print()
print(path)
for level in range(env.MAX_Level_Of_Point+1):

    for i_episode in range(1, iteration):
        episode_reward = 0
        episode_steps = 0
        done = False
        env.reset()
        state = env.get_state()

        while not done:
            action = agent.select_action(state)
            env.action(action)
            time.sleep(Config.time_sleep_interval)

            next_state, reward, done, success = env.observation()
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            if(done):
                episode_success.append(success)

            state = next_state

    SuccessRate = sum(episode_success)/iteration
    SuccessRate = SuccessRate*100
    temp_level_Of_SuccessRate.append(SuccessRate)
    print("level :" + str(level+1) + " -> " + str(SuccessRate))

    if not(env.Level_Of_Point >= env.MAX_Level_Of_Point):
        episode_success = []
        env.Level_Of_Point += 1

Compare_level_Of_SuccessRate.append(temp_level_Of_SuccessRate)

NoneCL = Compare_level_Of_SuccessRate


for type in model_type:

    Compare_level_Of_SuccessRate = NoneCL

    for compare in model_compare:

        # Environment
        env = Env.Ned2_control()
        env.reset()

        # Agent
        agent = SAC(12, 3, Config)

        path = "TestModel/" + compare + str(type) + ".tar"

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        agent.policy.load_state_dict(checkpoint['model'])

        # Training Loop
        total_numsteps = 0
        updates = 0
        iteration = 100
        episode_success = []
        temp_level_Of_SuccessRate = []
        now = datetime.datetime.now()

        print()
        print(path)
        for level in range(env.MAX_Level_Of_Point+1):

            for i_episode in range(1, iteration):
                episode_reward = 0
                episode_steps = 0
                done = False
                env.reset()
                state = env.get_state()

                while not done:
                    action = agent.select_action(state)
                    env.action(action)
                    time.sleep(Config.time_sleep_interval)

                    next_state, reward, done, success = env.observation()
                    episode_steps += 1
                    total_numsteps += 1
                    episode_reward += reward

                    if(done):
                        episode_success.append(success)

                    state = next_state

            SuccessRate = sum(episode_success)/iteration
            SuccessRate = SuccessRate*100
            temp_level_Of_SuccessRate.append(SuccessRate)
            print("level :" + str(level+1) + " -> " + str(SuccessRate))

            if not(env.Level_Of_Point >= env.MAX_Level_Of_Point):
                episode_success = []
                env.Level_Of_Point += 1

        Compare_level_Of_SuccessRate.append(temp_level_Of_SuccessRate)

    for i in range(len(Compare_level_Of_SuccessRate[0])):
        csv_data.append([str(type),str(i+1), Compare_level_Of_SuccessRate[0][i],Compare_level_Of_SuccessRate[1][i]])

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(csv_data)