#import general libraries
import wandb
import torch
import datetime
import numpy as np
import itertools
import gc
import os
import time
import logging

#import custom library
from sac import SAC
from replay_memory import ReplayMemory
import Config
import env


class Training_Robotic_arm():
    def __init__(self) -> None:
        # Enable garbage collector
        gc.enable()

        # Environment
        self.env = env.Ned2_control()

        # Agent
        self.agent = SAC(Config.state_size, Config.action_size, Config)

        # Memory
        self.memory = ReplayMemory(Config.replay_size, Config.seed)

        self.Set_training_parameters()
        self.Set_log_parameters()

    def Set_training_parameters(self) -> None:
        # Training Loop
        self.total_numsteps = 0
        self.updates = 0
        self.losses = None

    def Set_log_parameters(self) -> None:
        self.episode_success = []
        self.success_rate = 0.0
        self.success_rate_list =[]
        self.episode_success_csv_data_list = []

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        project_name = 'Niryo_NED2_SAC_singletarget'
        run_name = now + "_tau:" + str(Config.tau) + "_lr:" + str(Config.lr) + "_batch:" + str(Config.batch_size)
        wandb.init(project=project_name, # 프로젝트 이름
                name=run_name,
                config=Config.args) # run 이름

        self.folderName = ('./models/{}_RR_{}'.format(now, Config.Replay_ratio))
        if not os.path.exists('./models'): os.mkdir('./models')
        os.mkdir(self.folderName)


    def Run(self):
        for i_episode in itertools.count(1):
            self.i_episode = i_episode

            episode_reward, episode_steps = self.Run_episode()

            self.Log_episode_results(episode_reward, episode_steps)
            if(self.losses != None) : self.Log_losses(*self.losses)
            self.Log_successrate_per_UoC()

            if (i_episode != 0) and (i_episode%Config.eval_frequency == 0):
                self.Evaluate()

            if self.total_numsteps > Config.num_steps:
                break

            # self.memory.save_buffer("Ned")
        wandb.finish()

    def Run_episode(self):
        origin_UoC = self.env.Curriculum_manager.Current_UoC

        episode_reward = 0
        episode_steps = 0
        done = False
        self.env.reset()
        state = self.env.get_state()

        if (origin_UoC != self.env.Curriculum_manager.Current_UoC) : self.memory = ReplayMemory(Config.replay_size, Config.seed)

        while not done:
            action = self.Decide_action(state)

            self.env.action(action)
            next_state, reward, done, success = self.env.observation()

            episode_steps += 1
            self.total_numsteps += 1
            episode_reward += reward
            self.Process_step(state, action, reward, next_state, done)
            
            state = next_state

        if success :
            self.Save_model()

        return episode_reward, episode_steps

    def Decide_action(self, state):
        if Config.start_steps > self.total_numsteps:
            action = np.random.uniform(-0.1, 0.1, size=Config.action_size).tolist()
        else:
            action = self.agent.select_action(state)
        return action

    def Process_step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        state = next_state
        if len(self.memory) > Config.batch_size:
            self.Update_networks()

    def Update_networks(self):
        for _ in range(Config.updates_per_step):
            self.losses = self.agent.update_parameters(self.memory, Config.batch_size, self.updates)
            self.updates += 1

    def Evaluate(self):
        if Config.eval is True:
            avg_reward = 0.
            episodes = Config.eval_episode

            for _  in range(episodes):
                self.env.reset()
                state = self.env.get_state()
                episode_reward = 0
                done = False

                while not done:
                    action = self.agent.select_action(state, evaluate=True)

                    self.env.action(action)

                    next_state, reward, done, success = self.env.observation()
                    episode_reward += reward
                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes


            wandb.log({'Episodic_logs/evaluate':avg_reward}, step=self.i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

        origin_UoC = self.env.Curriculum_manager.Current_UoC

        episode_reward = 0
        episode_steps = 0
        done = False
        self.env.reset()
        state = self.env.get_state()

        if (origin_UoC != self.env.Curriculum_manager.Current_UoC) : self.memory = ReplayMemory(Config.replay_size, Config.seed)

        while not done:
            action = self.Decide_action(state)

            self.env.action(action)

            next_state, reward, done, success = self.env.observation()
            episode_steps += 1
            self.total_numsteps += 1
            episode_reward += reward
            self.Process_step(state, action, reward, next_state, done)

        if success :
            self.Save_model()

        return episode_reward, episode_steps


    def Log_losses(self, critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha):
        wandb.log({'NN_logs/critic_loss_1':critic_1_loss}, step=self.i_episode)
        wandb.log({'NN_logs/critic_loss_2':critic_2_loss}, step=self.i_episode)
        wandb.log({'NN_logs/policy_loss':policy_loss}, step=self.i_episode)
        wandb.log({'NN_logs/entropy_loss':ent_loss}, step=self.i_episode)
        wandb.log({'NN_logs/entropy':alpha}, step=self.i_episode)

    def Log_episode_results(self, episode_reward, episode_steps):
        wandb.log({'Episodic_logs/reward':round(episode_reward, 2)}, step=self.i_episode)
        wandb.log({'Episodic_logs/episode_steps':episode_steps}, step=self.i_episode)
        wandb.log({'Episodic_logs/UoC':self.env.Curriculum_manager.Current_UoC}, step=self.i_episode)
        wandb.log({'Episodic_logs/total_numsteps':self.total_numsteps}, step=self.i_episode)
        wandb.log({'Episodic_logs/replaymemory_size':len(self.memory)}, step=self.i_episode)
        
    def Log_successrate_per_UoC(self):
        successrate_dict = self.env.Curriculum_manager.success_rate_per_UoC
        phase = self.env.Curriculum_manager.Current_UoC
        if(successrate_dict != None):
            for UoC in range(1, self.env.Curriculum_manager.Max_UoC+1) :
                if(UoC in successrate_dict):
                    successrate = successrate_dict[UoC]
                    log_name = 'Curriculum_logs_phase_'+str(phase)+'/UoC_'+str(UoC)
                    wandb.log({log_name:successrate*100}, step=self.i_episode)

    def Save_model(self):
        torch.save({
            'model': self.agent.policy.state_dict(),
            'optimizer': self.agent.policy_optim.state_dict()
        }, self.folderName + "/model_"+str(self.i_episode)+".tar")

if __name__ == "__main__":
    main = Training_Robotic_arm()
    main.Run()