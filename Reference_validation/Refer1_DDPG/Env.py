#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import rospy
import moveit_commander #Python Moveit interface를 사용하기 위한 모듈
import moveit_msgs.msg
import geometry_msgs.msg
import math
from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import RobotState
from tf.transformations import quaternion_matrix
from gazebo_msgs.msg import ModelState 
from gazebo_msgs.srv import SetModelState
import math
import time
import numpy as np
from sensor_msgs.msg import JointState

class Ned2_control(object):
    def __init__(self):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        group_name = "ned2" #moveit의 move_group name >> moveit assitant로 패키지 생성 시 정의
        move_group = moveit_commander.MoveGroupCommander(group_name) # move_group node로 동작을 계획하고,  실행 
        self.robot = moveit_commander.RobotCommander()
        self.move_group = move_group

        self.target = [0,0,0] #target 위치

        # action 관련
        self.isLimited = False
        self.Iswait = False
        self.Limit_joint=[[-171.88,171.88],
                            [-105.0,34.96],
                            [-76.78,89.96],
                            [-119.75,119.75],
                            [-110.01,110.17],
                            [-144.96,144.96]]
        self.weight = [6.8, 3, 3.32, 4.8, 4.4, 5.8]

        # 오류 최소화를 위한 변수
        self.prev_state = []

        # time_step
        self.time_step = 0
        self.MAX_time_step = 200

        self.prev_linear_velocity = [0, 0, 0]

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
    
    def calc_distance(self, point1, point2):
        # 각 좌표의 차이를 제곱한 후 더한 값을 제곱근한다.
        distance = math.sqrt((point1[0] - point2[0]) ** 2 +
                            (point1[1] - point2[1]) ** 2 +
                            (point1[2] - point2[2]) ** 2)
        return distance
    

    def get_end_effector_linear_velocity(self, current_pose, previous_pose, dt):
        # 위치 변화 계산
        dx = current_pose[0] - previous_pose[0]
        dy = current_pose[1] - previous_pose[1]
        dz = current_pose[2] - previous_pose[2]

        # 속도 계산
        vx = dx / dt
        vy = dy / dt
        vz = dz / dt

        return [vx, vy, vz]
    
    def get_angle_between_velocity_and_target(self, current_pose, linear_velocity):
        # 현재 위치에서 목표 위치까지의 방향 벡터 계산
        direction_vector = [
            self.target[0] - current_pose[0],
            self.target[1] - current_pose[1],
            self.target[2] - current_pose[2]
        ]

        if(linear_velocity == [0, 0, 0]): linear_velocity = self.prev_linear_velocity
        else : self.prev_linear_velocity = linear_velocity

        # 벡터 정규화
        direction_vector = direction_vector / np.linalg.norm(direction_vector)
        linear_velocity = linear_velocity / np.linalg.norm(linear_velocity)
        
        # 두 벡터 사이의 각도 계산 (라디안)
        dot_product = np.dot(direction_vector, linear_velocity)
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        # 라디안을 도로 변환
        angle_degrees = np.degrees(angle)
        
        return angle_degrees

    def action(self,angle):  # angle 각도로 이동 (angle 은 크기 6의 리스트 형태)
        joint = self.move_group.get_current_joint_values()
        angle = self.Degree_to_Radian(angle)

        joint[0] += (angle[0]) * self.weight[0]
        joint[1] += (angle[1]) * self.weight[1]
        joint[2] += (angle[2]) * self.weight[2]
        joint[3] = 0
        joint[4] = 0
        joint[5] = 0

        for i in range(len(self.Limit_joint)):
            if(self.Limit_joint[i][1] < joint[i]):
                joint[i] = self.Limit_joint[i][1]
            elif(self.Limit_joint[i][0] > joint[i]):
                joint[i] = self.Limit_joint[i][0]

        try:
            self.move_group.go(joint, wait=self.Iswait)
        except:
            # print("move_group.go EXCEPT, ", joint)
            self.isLimited = True

        self.time_step += 1
            
    def reset(self):
        self.time_step = 0
        self.isLimited = False
        self.move_group.go([0,0,0,0,0,0], wait=True)
        self.target_reset()
        
    def get_joint2_position(self):
        joint3_pose = self.move_group.get_current_pose(end_effector_link="arm_link").pose
        joint3_position = [joint3_pose.position.x, joint3_pose.position.y, joint3_pose.position.z]
        return joint3_position

    def get_joint3_position(self):
        joint3_pose = self.move_group.get_current_pose(end_effector_link="elbow_link").pose
        joint3_position = [joint3_pose.position.x, joint3_pose.position.y, joint3_pose.position.z]
        return joint3_position
    
    def get_endeffector_position(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_state(self):
        joint2_pos = self.get_joint2_position()
        joint3_pos = self.get_joint3_position()
        endeffector_pos = self.get_endeffector_position()
        
        relative_joint2 = [joint2_pos[0] - self.target[0], joint2_pos[1] - self.target[1], joint2_pos[2] - self.target[2]]
        relative_joint3 = [joint3_pos[0] - self.target[0], joint3_pos[1] - self.target[1], joint3_pos[2] - self.target[2]]
        relative_endeffector = [endeffector_pos[0] - self.target[0], endeffector_pos[1] - self.target[1], endeffector_pos[2] - self.target[2]]
        
        state = joint3_pos + endeffector_pos + relative_joint2 + relative_joint3 + relative_endeffector

        if(len(state) == 15): self.prev_state = state
        else : state = self.prev_state

        return state


    def get_reward(self, distance, angle_difference):
        # R(position)

        # R(theta)
        if(angle_difference >= 90): R_theta = -0.001
        elif(90 > angle_difference >= 22.5): R_theta = 0.1
        elif(22.5 > angle_difference >= 11.25): R_theta = 0.3
        elif(11.25 > angle_difference >= 0): R_theta = 0.6

        # R(dinstance)
        df = 0.7
        if(distance >= df): R_distance = 0.0
        elif(df > distance >= df*0.7): R_distance = 0.01
        elif(df*0.7 > distance >= df*0.5): R_distance = 0.06
        elif(df*0.5 > distance >= df*0.1): R_distance = 0.17
        elif(df*0.1 > distance >= 0): R_distance = 1.17

        isDone, IsSuccess = False, False
        # if(self.time_step >= self.MAX_time_step) or (self.get_endeffector_position()[2] < 0.1) : isDone,IsSuccess = True, False
        if(self.time_step >= self.MAX_time_step):
            R_distance += 10
            isDone,IsSuccess = True, False

        if(self.get_endeffector_position()[2] < 0.1) or (self.isLimited == True): 
            R_distance -= 30
            isDone,IsSuccess = True, False

        if(distance <= 0.03):
            R_distance += 30
            isDone,IsSuccess = True,True

        totalReward = R_theta + R_distance
        return totalReward, isDone,IsSuccess
    
    def step(self, angle):
        distance = self.calc_distance(self.target, self.get_endeffector_position())
        # print(distance)

        df = 0.7
        if(distance >= df): time_interver = 0.11
        elif(df > distance >= df*0.7): time_interver = 0.09
        elif(df*0.7 > distance >= df*0.5): time_interver = 0.07
        elif(df*0.5 > distance >= df*0.1): time_interver = 0.05
        elif(df*0.1 > distance >= 0): time_interver = 0.03

        self.action(angle)
        previous_pose = self.move_group.get_current_joint_values()
        time.sleep(time_interver) #거리에 따라 조절
        current_pose = self.move_group.get_current_joint_values()
        linear_velocity = self.get_end_effector_linear_velocity(current_pose, previous_pose, time_interver)
        angle_difference = self.get_angle_between_velocity_and_target(current_pose, linear_velocity)
        # print(f"Linear velocity: {linear_velocity}")
        # print(f"Angle difference: {angle_difference} degrees")

        totalReward,isDone,IsSuccess = self.get_reward(distance, angle_difference)
        current_state = self.get_state()

        return current_state,totalReward,isDone, IsSuccess

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
        for i in range(300):
            set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
            resp = set_state(state_msg)    


if __name__ == "__main__":
    ned2_control = Ned2_control()
    rospy.sleep(1)  # 초기화 시간 대기

    # # 테스트 코드
    ned2_control.reset()

    while not rospy.is_shutdown():
        ned2_control.step([0.0, -0.3, 0.35, 0, 0, 0])
        print(ned2_control.get_joint3_position())
