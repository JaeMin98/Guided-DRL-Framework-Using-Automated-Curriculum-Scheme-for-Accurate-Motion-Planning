#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander #Python Moveit interface를 사용하기 위한 모듈
import moveit_msgs.msg
import geometry_msgs.msg
import random
import math
from moveit_commander.conversions import pose_to_list
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import os
from datetime import datetime
import Config

# alpha shape #
import math
import csv
# alpha shape #

class Ned2_control(object):
    def __init__(self):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        group_name = "ned2" #moveit의 move_group name >> moveit assitant로 패키지 생성 시 정의
        move_group = moveit_commander.MoveGroupCommander(group_name) # move_group node로 동작을 계획하고,  실행 
        
        self.move_group = move_group

        # CSV 파일 열기
        self.level_point = []
        for i in range(1,6):
            with open('./DataCSV/level_'+str(i)+'.csv', 'r') as file:
                reader = csv.reader(file)

                # 각 행들을 저장할 리스트 생성
                rows = []

                for row in reader:
                    row_temp = row[:3]
                    rows.append(row_temp)
                self.level_point.append(rows)

        # Level 3,4 바꾸기
        CSV_change = self.level_point[2]
        self.level_point[2] = self.level_point[3]
        self.level_point[3] = CSV_change

        self.MAX_Level_Of_Point = 4
        self.Level_Of_Point = 0

        Config.Clustering_K = self.MAX_Level_Of_Point+1

        self.target = []

        self.rotation_target = 0

        self.isLimited = False 
        self.Limit_joint=[[-171.88,171.88],
                            [-105.0,34.96],
                            [-76.78,89.96],
                            [-119.75,119.75],
                            [-110.01,110.17],
                            [-144.96,144.96]]

        

        self.Iswait = False
        if(self.Iswait):
            self.weight = [5,
                            2,
                            2,
                            3,
                            3,
                            4]
        else:
            self.weight = [6.8,
                            3,
                            3.32,
                            4.8,
                            4.4,
                            5.8]
            

        ## reward weight ##
        self.Success_weight = 0.7
        self.Distance_weight = 3
        self.Limited_weight = 0.3
        self.Negative_DF = 1.01
        self.Positive_DF = 0.99

        self.start_GD = 0.04
        self.limit_GD = 0.02
        self.count_complete = 0
        self.goalDistance = self.start_GD

        self.Discount_count = 0

        self.farDistance = 0.999

        self.prev_state = []
        self.joint_error_count = 0
        self.prev_action = []
        self.time_step = 0
        self.MAX_time_step = Config.max_episode_steps

        self.job_list = []
        self.target_directory = ""

    def Degree_to_Radian(self,Dinput):
        Radian_list = []
        for i in Dinput:
            Radian_list.append(i* (math.pi/180.0))
        return Radian_list

    def Radian_to_Degree(self,Rinput):
        Degree_list = []
        for i in Rinput:
            Degree_list.append(i* (180.0/math.pi))
        return Degree_list

    def action(self,angle):  # angle 각도로 이동 (angle 은 크기 6의 리스트 형태)
        joint = self.get_state()[0:6]
        self.job_list.append(joint)
        # self.prev_action = copy.deepcopy(joint)

        joint[0] += angle[0] * self.weight[0]
        joint[1] += angle[1] * self.weight[1]
        joint[2] += angle[2] * self.weight[2]
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0

        for i in range(len(self.Limit_joint)):
            if(self.Limit_joint[i][1] < joint[i]):
                joint[i] = self.Limit_joint[i][1]
                # self.isLimited = True
                # print("OUT OF (Limit_joint), UPPER JOINT"+str(i+1) + ", ", joint)
            elif(self.Limit_joint[i][0] > joint[i]):
                joint[i] = self.Limit_joint[i][0]
                # self.isLimited = True
                # print("OUT OF (Limit_joint), LOWER JOINT"+str(i+1) + ", ", joint)

        try:
            self.move_group.go(self.Degree_to_Radian(joint), wait=self.Iswait)
        except:
            print("move_group.go EXCEPT, ", joint)
            self.isLimited = True

        self.time_step += 1

    def ikaction(self):
        print("hiru")
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = 0.2
        pose_goal.position.y = 0.3
        pose_goal.position.z = 0.4 #static

        self.move_group.set_pose_target(pose_goal)
        self.move_group.go(wait=True)
        self.move_group.stop()
            
    def reset(self):
        # print("Go To Home pose")
        self.time_step = 0
        self.Negative_DF = 1.01
        self.Positive_DF = 0.99
        self.move_group.go([0,0,0,0,0,0], wait=True)
        self.target_reset()
        

    def get_state(self): #joint 6축 각도
        joint = self.move_group.get_current_joint_values()
        state = self.Radian_to_Degree(joint) + self.target + self.get_pose()
        if(len(state) == 12):
            self.prev_state = state
        else:
            state = self.prev_state
            self.joint_error_count += 1
            print(self.joint_error_count)
        return state

    def get_pose(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_reward(self):
        # self.move_group.stop()
        end_effector = self.get_pose()
        
        d = math.sqrt(abs((end_effector[0]-self.target[0])**2 + (end_effector[1]-self.target[1])**2 + (end_effector[2]-self.target[2])**2 ))
        # reward parameter
        rewardS = 0 # 도달 성공 시 부여
        rewardD = -1.5 * d # 거리가 가까울수록 부여
        rewardL = 0 # 로봇팔 동작 가능 범위(각도)를 벗어나면 부여
        totalReward = 0
        isFinished = False
        isComplete = 0

        if(self.time_step >= self.MAX_time_step):
            # print("OUT OF (time_step), ", self.time_step)
            isFinished = True
        
        if not(0.1< end_effector[2] < 0.8):
            # print("OUT OF (end_effector), Z : ", end_effector[2])
            # self.move_group.stop()
            # self.move_group.go(self.Degree_to_Radian(self.prev_action), wait=True)
            isFinished = True

        # 목표 지점 도달 시
        if (d <= self.goalDistance):
            # print("SUCCESS")
            self.count_complete += 1

            #수정함
            # if(self.count_complete >= 5):
            #     self.count_complete = 0
            #     temp = round(self.goalDistance - 0.01, 3)
            #     if( temp < self.limit_GD):
            #         self.goalDistance = self.goalDistance
            #     else:
            #         self.goalDistance = temp

            isFinished = True
            isComplete = 1  
            rewardS = 50 + (12.5 * self.Level_Of_Point)
            

        # 제한 범위 외로 이동 시
        elif (d > self.farDistance):
            print("OUT OF (farDistance), distance : ", d)
            isFinished = True
            rewardL = -10

        # 로봇팔 동작 범위(각도)를 벗어날 시
        if (self.isLimited):
            isFinished = True
            self.isLimited = False
            rewardL = -10

        totalReward += (self.Success_weight * rewardS)
        totalReward += (self.Distance_weight * rewardD)
        totalReward += (self.Limited_weight * rewardL)

        # totalReward += (self.Success_weight * rewardS) * self.Positive_DF
        # totalReward += (self.Distance_weight * rewardD) * self.Negative_DF
        # totalReward += (self.Limited_weight * rewardL) * self.Negative_DF

        # self.Negative_DF *= 1.01
        # self.Positive_DF *= 0.99

        return totalReward,isFinished,isComplete

    def observation(self):
        totalReward,isFinished,isComplete = self.get_reward()
        current_state = self.get_state()

        return current_state,totalReward,isFinished, isComplete
    
    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x = self.target[0]
        state_msg.pose.position.y = self.target[1]
        state_msg.pose.position.z = self.target[2]
        state_msg.pose.orientation.x = 0
        state_msg.pose.orientation.y = 0
        state_msg.pose.orientation.z = 0
        state_msg.pose.orientation.w = 0

        rospy.wait_for_service('/gazebo/set_model_state')
        for i in range(100):
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            resp = set_state(state_msg)    

    def make_job_file(self, current_epi):

        now = datetime.now()
        date_time = "{}.{}.{}.{}".format(now.day, now.hour, now.minute, now.second)

        if not(os.path.isdir(self.target_directory)):
            self.target_directory = "Job_Files/"+ date_time
            if not os.path.exists(self.target_directory):
                os.makedirs(self.target_directory)

        fileName = self.target_directory + "/Episode_" + str(current_epi) + ".JOB"

        # 파일 쓰기
        with open(fileName, 'w') as f:
            f.write("Program File Format Version : 1.6  MechType: 370(HS220-01)  TotalAxis: 6  AuxAxis: 0\n")
            data = ''
            for i in range(0, len(self.job_list)):
                data += "S" + str(i + 1) + "   MOVE P,S=60%,A=3,T=1  ("
                for j in range(0, 6):
                    data += str(round(self.job_list[i][j], 3))
                    if j != 5:
                        data += ","

                    else:
                        data += ")A\n"

            f.write(data)
            f.write("     END")
        f.close()
        self.job_list = []