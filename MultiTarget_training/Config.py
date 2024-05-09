#! /usr/bin/env python3
# -*- coding: utf-8 -*-

## environment parameter
env_name = 'ned2'
Replay_ratio = 0.2
Success_Standard = 0.9
state_size = 10
action_size = 3

## soft actor-critic parameter
policy = "Gaussian"
gamma = 0.99
tau = 0.005
lr = 0.0005 #0.0003
alpha = 0.2
automatic_entropy_tuning = True
seed = 123456 # seed = 123456 -> random
hidden_size = 64
updates_per_step =1 #1 time step에 Update를 몇 번 할것인지
target_update_interval = 1 #Update간의 간격을 나타냄 (ex. 3이면 Update를 요청 받았을 때 3의 배수번째로 요청하는 것만 업데이트함)


## training parameter
eval = False
eval_episode = 3
eval_frequency = 30

Is_Clearing_Memory = True
num_steps = 1000001 #60000 * time
batch_size = 512
start_steps = 10000
max_episode_steps = 64
time_sleep_interval = 0.05 #sec 적절한 값 찾기. 학습이 시작되면 값이 달라짐

average_count_for_successrate = 20
isExit_IfSuccessLearning = True #목표 달성 시(success rate 0.9이상일 때) 학습을 종료할 것인지

replay_size = num_steps #1000000
cuda = "cuda"


args = {
    "env_name": env_name,
    "Replay_ratio": Replay_ratio,
    "Success_Standard": Success_Standard,
    "policy": policy,
    "eval": eval,
    "eval_episode": eval_episode,
    "eval_frequency" : eval_frequency,
    "gamma": gamma,
    "tau": tau,
    "lr": lr,
    "alpha": alpha,
    "automatic_entropy_tuning": automatic_entropy_tuning,
    "seed": seed,
    "hidden_size": hidden_size,
    "updates_per_step": updates_per_step,
    "target_update_interval": target_update_interval,
    "Is_Clearing_Memory": Is_Clearing_Memory,
    "num_steps": num_steps,
    "batch_size": batch_size,
    "start_steps": start_steps,
    "max_episode_steps": max_episode_steps,
    "time_sleep_interval": time_sleep_interval,
    "average_count_for_successrate": average_count_for_successrate,
    "isExit_IfSuccessLearning": isExit_IfSuccessLearning,
    "replay_size": replay_size,
    "state_size": state_size,
    "action_size": action_size,
    "cuda": cuda
}
