#! /usr/bin/env python3
# -*- coding: utf-8 -*-

#import general libraries
import sys
import os
import glob
import math
import csv
import random
import time
from collections import defaultdict, deque
import numpy as np

#import ros|movit|gazrbo libraries
import rospy
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from moveit_commander.conversions import pose_to_list
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
# import geometry_msgs.msg
# import moveit_msgs.msg

#import custom library
import Config

class Ned2_control(object):
    def __init__(self):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)

        # 로봇 그룹 초기화
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("ned2")
        rospy.sleep(2)  # Scene이 완전히 로드될 때까지 기다린다.

        self.set_reward_parameters()
        self.set_action_parameters()
        self.set_curriculum_parameters()

        self.target = self.Curriculum_manager.get_target()
        
    def set_reward_parameters(self) -> None:
        self.goal_distance = 0.03 # The unit is "m"
        self.far_distance = 0.7

        self.Is_success = False

        self.IsLimited = False
        self.Is_under_Z = False
        self.Is_far_away = False
        self.Is_timestep_over = False

        self.Is_done = False

    def set_action_parameters(self) -> None:
        self.timestep = 0
        self.action_size = Config.action_size
        self.state_size = Config.state_size
        self.Max_timestep = Config.max_episode_steps
        self.time_sleep_interval = Config.time_sleep_interval
        self.Iswait = True
        Limit_joint = [[-171.88, 171.88], [-105.0, 34.96], [-76.78, 89.96],
                            [-119.75, 119.75], [-110.01, 110.17], [-144.96, 144.96]]
        self.Limit_joint = [[round(math.radians(degree),4) for degree in pair] for pair in Limit_joint]
        self.action_weight = [2.43,  1.0,  1.21,  1.71,  1.57,  2.07]

    def set_curriculum_parameters(self) -> None:
        self.Curriculum_manager = Curriculum_manager()
        self.Selected_UoC = None


    def action(self, action_value) -> None:
        joint = self.current_state[7:10]

        for i in range(len(action_value)):
            joint[i] += action_value[i] * self.action_weight[i]
            
            if(joint[i] < self.Limit_joint[i][0]) :
                joint[i] = max(joint[i], self.Limit_joint[i][0])
                self.IsLimited = True
            elif(joint[i] > self.Limit_joint[i][1]) :
                joint[i] = min(joint[i], self.Limit_joint[i][1])
                self.IsLimited = True

        joint += [0]*(6-self.action_size)

        if self.Iswait :
            self.move_group.go(joint, wait=self.Iswait)
        else : 
            self.move_group.go(joint, wait=self.Iswait)
            time.sleep(self.time_sleep_interval)
            self.move_group.stop()

        self.timestep += 1

    # def action(self, action_value) -> None:
    #     joint = self.current_state[7:10]
    #     joint = [math.degrees(element) for element in joint]

    #     for i in range(len(action_value)):
    #         joint[i] += math.degrees(action_value[i])
    #         if(joint[i] < self.Limit_joint[i][0]) :
    #             joint[i] = max(joint[i], self.Limit_joint[i][0])
    #             self.IsLimited = True
    #         elif(joint[i] > self.Limit_joint[i][1]) :
    #             joint[i] = min(joint[i], self.Limit_joint[i][1])
    #             self.IsLimited = True

    #     #radians to degrees
    #     joint = [math.radians(element) for element in joint]
    #     joint += [0]*(6-self.action_size)

    #     if self.Iswait :
    #         self.move_group.go(joint, wait=self.Iswait)
    #     else : 
    #         self.move_group.go(joint, wait=self.Iswait)
    #         time.sleep(self.time_sleep_interval)
    #         self.move_group.stop()

    #     self.timestep += 1

    def reset(self) -> None:
        if (self.Selected_UoC != None): self.Curriculum_manager.managing_curriculum([self.Is_success, self.Selected_UoC])

        self.timestep = 0
        self.IsLimited = False

        self.Curriculum_manager.select_target_with_replay_ratio()
        self.target, self.Selected_UoC = self.Curriculum_manager.get_target()
        self.reset_target()

        self.move_group.go([0]*6, wait=True)
        self.init_state = self.get_state()
        self.current_state = self.init_state

    def reset_target(self) -> None:
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x, state_msg.pose.position.y, state_msg.pose.position.z = self.target
        state_msg.pose.orientation.x, state_msg.pose.orientation.y,  state_msg.pose.orientation.z,  state_msg.pose.orientation.w = 0,0,0,0
        rospy.wait_for_service('/gazebo/set_model_state')
        #Sometimes this command is ignored by Gazebo, so you'll need to do it N times to get it to work reliably
        for _ in range(100):
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            set_state(state_msg)

    def get_state(self) -> list:
        while(1):
            try:
                joint_values = self.move_group.get_current_joint_values()
                end_effector_XYZ = self.get_pose()

                joint_values = [round(element, 4) for element in joint_values]
                end_effector_XYZ = [round(element, 4) for element in end_effector_XYZ]
                distance = self.euclidean_distance(end_effector_XYZ, self.target)

                state = []
                state.append(distance)
                for i in range(len(self.target)):
                    state.append(end_effector_XYZ[i])
                    state.append(self.target[i])
                for i in range(3):
                    state.append(joint_values[i])

                if(len(state) == self.state_size):
                    if not(np.isnan(state).any()):
                        return state
            except:
                print("get state error")
                
    def euclidean_distance(self, point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance

    def get_pose(self) -> list:
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]

    def calculate_reward(self) -> None:
        end_effector_Z = self.next_state[5] #end_effector_Z
        distance = self.next_state[0] #distance

        self.Is_success = distance < self.goal_distance
        self.Is_under_Z = end_effector_Z <= 0.05
        self.Is_far_away = distance >= self.far_distance
        self.Is_timestep_over = (self.timestep >= (self.Max_timestep-1))
        
        self.Is_done = self.Is_success or self.IsLimited or self.Is_under_Z or self.Is_far_away or self.Is_timestep_over

        rewardD = -1 * distance
        rewardS = 100 if distance <= self.Is_success else 0
        rewardL = -180 if self.IsLimited or self.Is_under_Z or self.Is_far_away else 0

        self.reward = rewardD + rewardS + rewardL

    def observation(self) -> list:
        self.next_state = self.get_state()
        self.calculate_reward()
        self.current_state = self.next_state

        return self.next_state, self.reward, self.Is_done, self.Is_success


class Curriculum_manager():
    def __init__(self, csv_path="./UoC_data") -> None:
        # This path represents the path where the curriculum's UoC data resides
        self.csv_path = csv_path 
        self.set_managing_parameters()
        self.set_UoC_parameters()
        self.UoC_point = [self.read_UoC_data(i) for i in range(self.Min_UoC, self.Max_UoC+1)]

    def count_csv_files(slef, directory) -> int:
        # Finds all files with the .csv extension in the specified directory
        # This automatically adjusts the parameters without you having to enter the number of UoCs present in your curriculum.
        path_pattern = os.path.join(directory, '*.csv')
        files = glob.glob(path_pattern)
        return len(files)

    def set_managing_parameters(self) -> None:
        self.Replay_ratio = Config.Replay_ratio
        self.target = [0,0,0] #[x,y,z]
        self.episode_sucees_log = []
        self.success_rate_per_UoC = None

    def set_UoC_parameters(self) -> None:
        #UoC means Unit of Curriculum
        #The UoC's sequence starts at 1 for human readability, and 0 when actually applied to the list
        self.Max_UoC = self.count_csv_files(self.csv_path)
        self.Min_UoC = 1
        self.Current_UoC = 1
        self.Selected_UoC = 1

    def read_UoC_data(self, UoC) -> list:
        with open(f'./UoC_data/UoC_{UoC}.csv', 'r') as file:
            reader = csv.reader(file)
            return [row[:3] for row in reader]
        
    def append_episode_sucees_log(self, logs) -> None:
        self.episode_sucees_log.append(logs)

    def clear_episode_sucees_log(self) -> None:
        self.episode_sucees_log = []
    
    def calculate_success_rate_per_UoC(self, compare_entries=20) -> dict:
        UoC_dict = defaultdict(deque)

        for item in self.episode_sucees_log:
            UoC = item[1]
            value = item[0]

            # 각 레벨별로 데이터를 저장하되, 최대 20개의 데이터만 유지
            if len(UoC_dict[UoC]) == compare_entries:
                UoC_dict[UoC].popleft()  # 가장 오래된 데이터를 제거
            UoC_dict[UoC].append(value)

        success_rate_per_UoC = {}
        for UoC, values in UoC_dict.items():
            success_rate_per_UoC[UoC] = sum(values) / len(values)  # True는 1, False는 0으로 계산

        return success_rate_per_UoC
    
    def managing_curriculum(self, logs, compare_entries=Config.average_count_for_successrate) -> None:
        self.append_episode_sucees_log(logs)
        is_data_enough = sum(1 for _, level in self.episode_sucees_log if level == self.Current_UoC) >= compare_entries

        if is_data_enough:
            self.success_rate_per_UoC = self.calculate_success_rate_per_UoC(compare_entries)
            
            if(self.success_rate_per_UoC[self.Current_UoC] > 0.89):
                next_UoC = self.Current_UoC + 1
                if(next_UoC < self.Max_UoC):
                    self.Current_UoC = next_UoC
                    self.clear_episode_sucees_log()
                    self.success_rate_per_UoC = None
                else:
                    if(Config.isExit_IfSuccessLearning):
                        exit(0)
        
    def select_target(self, UoC) -> None:
        # select target w/o replay mechanism
        random_index = random.choice(range(len(self.UoC_point[UoC])))
        self.target = [float(element) for element in self.UoC_point[UoC][random_index]]
        self.target = [round(element, 4) for element in self.target]

    def select_target_with_replay_ratio(self) -> None:
        if random.random() > Config.Replay_ratio:
            self.Selected_UoC = self.Current_UoC
        else:
            self.Selected_UoC = random.randint(1, max(1, self.Current_UoC-1))
        self.select_target(self.Selected_UoC - 1)

    def get_target(self) -> list:
        return self.target, self.Selected_UoC
    

class equalization():
    def __init__(self, initial_value):
        self.initial_value = initial_value
        self.mean = 0.5
        self.max = 1.0

    def equalization_function(self, data):
        if data <= self.initial_value:
            return data * (self.mean / self.initial_value)
        else:
            return self.mean + (data - self.initial_value) * ((self.max - self.mean) / (1 - self.initial_value))

    def get_value(self, o):
        return round(self.equalization_function(o), 4)
    

if __name__ == "__main__":

    time_interver = 0.05

    def action_test() -> None:
        env = Ned2_control()
        env.reset()
        print(0, env.observation())
        for i in range(256):
            env.action([0.0,   -0.1,   -0.5,   0.0,   0.0,   0.0])
            next_state, reward, done, success = env.observation()
            if(success == True) or done == True :
                env.reset()
        print(i, env.observation())
        env.reset()

    def success_test() -> None:
        # with curriculum
        env = Ned2_control()
        env.goal_distance = 20
        env.reset()
        for i in range(300):
            env.action([0.0,   0.0,   0.0,   0.0,   0.0,   0.0])
            time.sleep(time_interver)
            result = env.observation()
            print(env.Curriculum_manager.success_rate_per_UoC)
            if(result[-1] == True) :
                env.reset()
        env.reset()

    env = Ned2_control()
    env.reset()
    for i in range(100):
        time.sleep(0.05)
        print(env.observation())

    # success_test()

