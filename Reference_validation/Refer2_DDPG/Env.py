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
        self.prev_distance = None

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
        self.prev_distance = None
        self.isLimited = False
        self.move_group.go([0,0,0,0,0,0], wait=True)
        self.move_group.stop()
        self.move_group.go([0,0,0,0,0,0], wait=True)
        self.move_group.stop()
        self.target_reset()
        print('reset')
        
    
    def get_endeffector_position(self):
        pose = self.move_group.get_current_pose().pose
        pose_value = [pose.position.x,pose.position.y,pose.position.z]
        return pose_value
    
    def get_state(self):
        joint = self.move_group.get_current_joint_values()
        state = joint[0:3] + self.target
        # state = joint[0:3] + self.get_endeffector_position() + self.target
        return state

    def get_reward(self):
        distance = self.calc_distance(self.get_endeffector_position(), self.target)

        R_basic= -1 * distance
        R_done= 0
        R_extra = 0
        if (self.prev_distance != None ): R_extra = -1 * (distance - self.prev_distance)
        self.prev_distance = distance

        isDone, isSuccess = False, False

        if(self.get_endeffector_position()[2] < 0.1) or (self.isLimited == True):
            R_done = -50
            isDone,isSuccess = True, False
     
        if(self.time_step >= self.MAX_time_step):
            R_done = 10
            isDone,isSuccess = True, False

        if(distance <= 0.05):
            R_done = 50
            isDone,isSuccess = True,True

        totalReward = R_basic + R_done + R_extra
        return totalReward, isDone,isSuccess
    
    def step(self, angle):
        time_interver = 0.05
        self.action(angle)
        time.sleep(time_interver) #거리에 따라 조절
        self.move_group.stop()

        totalReward,isDone,isSuccess = self.get_reward()
        current_state = self.get_state()

        return current_state,totalReward,isDone, isSuccess
    

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


if __name__ == "__main__":
    ned2_control = Ned2_control()
    rospy.sleep(1)  # 초기화 시간 대기

    # # 테스트 코드
    ned2_control.reset()

    while not rospy.is_shutdown():
        ned2_control.step([0.0, -0.3, 0.35, 0, 0, 0])
        print(ned2_control.get_joint3_position())
