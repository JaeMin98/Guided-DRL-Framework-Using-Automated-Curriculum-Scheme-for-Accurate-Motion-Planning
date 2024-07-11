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
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import Config
import numpy as np

# alpha shape #
import math
import csv
# alpha shape #

def quaternion_angle_scipy(q1, q2):
    # 쿼터니언을 Rotation 객체로 변환
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    # 두 Rotation 객체의 상대 회전 계산
    relative_rotation = r1.inv() * r2
    # 각도 계산
    angle_radian = relative_rotation.magnitude()

    return angle_radian

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


        self.target_Euler_list = [[0.42,0,1.27],[0.15,0.12,2.24]]
        self.target_Quaternion_list = []
        self.set_target_quaternion()
        self.target_orientation_index = 0

        # Level 3,4 바꾸기
        CSV_change = self.level_point[2]
        self.level_point[2] = self.level_point[3]
        self.level_point[3] = CSV_change

        self.MAX_Level_Of_Point = 4
        self.Level_Of_Point = 0

        Config.Clustering_K = self.MAX_Level_Of_Point+1


        self.rotation_target = 0

        self.isLimited = False 
        self.Limit_joint=[[-171.88,171.88],
                            [-105.0,34.96],
                            [-76.78,89.96],
                            [-119.75,119.75],
                            [-110.01,110.17],
                            [-144.96,144.96]]

        

        self.Iswait = False

        self.weight = [6.8,
                        3,
                        3.32,
                        9.6,
                        8.8,
                        11.6]
            

        ## reward weight ##
        self.Success_weight = 0.7
        self.Distance_weight = 1.5
        self.Limited_weight = 0.3
        self.Negative_DF = 1.01
        self.Positive_DF = 0.99

        self.count_complete = 0
        self.goalDistance = 0.03

        self.Discount_count = 0

        self.farDistance = 0.999

        self.prev_state = []
        self.joint_error_count = 0
        self.prev_action = []
        self.time_step = 0
        self.MAX_time_step = Config.max_episode_steps

        self.job_list = []
        self.target_directory = ""

    def set_target_quaternion(self):
        for euler_angles in self.target_Euler_list:
            # 오일러 각도를 라디안 단위로 설정하여 회전 객체로 변환
            rotation = R.from_euler('xyz', euler_angles, degrees=False)
            # 회전 객체를 쿼터니언으로 변환
            quaternion = rotation.as_quat()
            self.target_Quaternion_list.append(quaternion.tolist())

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
        joint = self.move_group.get_current_joint_values()
        self.job_list.append(joint)
        # self.prev_action = copy.deepcopy(joint)

        for i in range(len(joint)):
            joint[i] += angle[i] * self.weight[i]

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
        
        if random.random() < Config.Current_Data_Selection_Ratio:
            random_index_list = random.choice(range(len(self.level_point[self.Level_Of_Point])))
            self.target = self.level_point[self.Level_Of_Point][random_index_list]
            self.target = [float(element) for element in self.target] # 목표 지점 위치
            self.target_reset()
        else :
            temp_range = max(0,self.Level_Of_Point - 1)
            temp_index = random.randint(0, temp_range)
            random_index_list = random.choice(range(len(self.level_point[temp_index])))
            self.target = self.level_point[temp_index][random_index_list]
            self.target = [float(element) for element in self.target] # 목표 지점 위치
            self.target_reset()
        

    def get_end_effector_XYZ(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_end_effector_Orientation(self):
        pose = self.move_group.get_current_pose().pose
        Quaternion_value = [pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w]

        rotation = R.from_quat(Quaternion_value)
        Euler_value = rotation.as_euler('xyz', degrees=False)

        return Quaternion_value, Euler_value
    
    def get_reward(self, d, end_effector_XYZ, angle_radian):
        # self.move_group.stop()

        # reward parameter
        rewardS = 0 # 도달 성공 시 부여
        rewardD = -1.5 * d # 거리가 가까울수록 부여
        rewardL = 0 # 로봇팔 동작 가능 범위(각도)를 벗어나면 부여
        rewardO = -1 * (angle_radian / 3.0)
        totalReward = 0
        isFinished = False
        isComplete = 0

        if(self.time_step >= self.MAX_time_step):
            isFinished = True
        
        if not(0.1< end_effector_XYZ[2] < 0.8):
            isFinished = True

        # 목표 지점 도달 시
        if (d <= self.goalDistance):
            self.count_complete += 1

            isFinished = True
            isComplete = 1  
            rewardS = 50
            

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
        totalReward += (self.Distance_weight * rewardO)
        totalReward += (self.Limited_weight * rewardL)


        return totalReward,isFinished,isComplete

    def get_state(self, distance, Quaternion_value, Euler_value, angle_degree): #joint 6축 각도
        joint = self.move_group.get_current_joint_values()

        robot_state = self.Radian_to_Degree(joint) + self.get_end_effector_XYZ()  + list(Quaternion_value) + list(Euler_value)
        target_state = self.target + self.target_Quaternion_list[self.target_orientation_index] + self.target_Euler_list[self.target_orientation_index]
        state = robot_state + target_state + [distance] + [angle_degree]

        if(len(state) == 28):
            self.prev_state = state
        else:
            state = self.prev_state
            self.joint_error_count += 1
            print(self.joint_error_count)
        return state
    
    def observation(self):
        end_effector_XYZ = self.get_end_effector_XYZ()
        distance = math.sqrt(abs((end_effector_XYZ[0]-self.target[0])**2 + (end_effector_XYZ[1]-self.target[1])**2 + (end_effector_XYZ[2]-self.target[2])**2 ))
        Quaternion_value, Euler_value = self.get_end_effector_Orientation()
        target_Quaternion_value = self.target_Quaternion_list[self.target_orientation_index]
        angle_radian = quaternion_angle_scipy(Quaternion_value, target_Quaternion_value)

        # print(f'=========================================')
        # print(f"robot : {Quaternion_value}")
        # print(f"robot : {Euler_value}")
        # print(f"target : {target_Quaternion_value}")
        # print(f"target : {self.target_Euler_list[self.target_orientation_index]}")
        # print(f"angle : {angle_radian}")
        # print(f'=========================================')

        totalReward,isFinished,isComplete = self.get_reward(distance, end_effector_XYZ, angle_radian)
        current_state = self.get_state(distance, Quaternion_value, Euler_value, angle_radian)

        return current_state,totalReward,isFinished, isComplete
    
    def target_reset(self):
        state_msg = ModelState()
        state_msg.model_name = 'cube'
        state_msg.pose.position.x = self.target[0]
        state_msg.pose.position.y = self.target[1]
        state_msg.pose.position.z = self.target[2]

        self.target_orientation_index = random.randint(0, len(self.target_Quaternion_list) - 1)

        state_msg.pose.orientation.x = self.target_Quaternion_list[self.target_orientation_index][0]
        state_msg.pose.orientation.y = self.target_Quaternion_list[self.target_orientation_index][1]
        state_msg.pose.orientation.z = self.target_Quaternion_list[self.target_orientation_index][2]
        state_msg.pose.orientation.w = self.target_Quaternion_list[self.target_orientation_index][3]

        rospy.wait_for_service('/gazebo/set_model_state')
        for i in range(300):
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



a = Ned2_control()
print(a.target_Euler_list)
print(a.target_Quaternion_list)