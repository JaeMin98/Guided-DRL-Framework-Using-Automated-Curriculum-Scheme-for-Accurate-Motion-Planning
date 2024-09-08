#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#import general libraries
import sys
import math
#import ros|movit|gazrbo libraries
import rospy
import os
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
import geometry_msgs.msg
import csv
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np



def quaternion_angle_radians(q1, q2):
    # Normalize the quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate the dot product
    dot_product = np.dot(q1, q2)
    
    # Ensure the dot product is within the valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = 2 * np.arccos(dot_product)
    
    return angle

class Ned2_control(object):
    def __init__(self, xyz_list, orientation):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        # Initialize a Robot Group
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("ned2")
        rospy.sleep(2)  # Wait for the Scene to fully load
        self.target_XYZ = []
        self.Is_valid = True
        self.prev_target = None

        self.target_xyz_list, self.target_quaternion_list = xyz_list, orientation
        self.iteration = 0

        self.folder_path = "data_points"
        self.csv_path = os.path.join(self.folder_path, 'data_points_origin.csv')
        self.headers = ['target_X', 'target_Y', 'target_Z', 'qX', 'qY', 'qZ', 'qW', 'execution_time','distance', 'angle',
                        'delta_of_6_axis', 'delta_of_3_axis', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 원하는 열 이름

        # 초기화 시 폴더 및 파일 확인
        self._initialize_csv()

    def reset(self) -> None:
        self.move_group.go([0]*6, wait=True)
        self.init_joint_values = self.move_group.get_current_joint_values()
        self.init_XYZ = self.get_pose()
        self.init_robot_quaternion, self.init_robot_euler_angles = self.get_orientation()

    def reset_target(self) -> None:
        pose = geometry_msgs.msg.Pose()

        self.target_XYZ = self.target_xyz_list[self.iteration]
        self.target_quaternion = self.target_quaternion_list[self.iteration]

        pose.position.x = self.target_XYZ[0]
        pose.position.y = self.target_XYZ[1]
        pose.position.z = self.target_XYZ[2]
        pose.orientation.x = self.target_quaternion[0]
        pose.orientation.y = self.target_quaternion[1]
        pose.orientation.z = self.target_quaternion[2]
        pose.orientation.w = self.target_quaternion[3]

        self.move_group.set_pose_target(pose)
        self.iteration += 1

            
    def action(self):
        start_time = rospy.get_time()
        self.move_group.go(wait=True)
        end_time = rospy.get_time()
        self.move_group.clear_pose_targets()
        execution_time = end_time - start_time
        self.next_joint_values = self.move_group.get_current_joint_values()
        self.robot_XYZ = self.get_pose()

        if(self.compare_lists(self.robot_XYZ, self.target_XYZ) > 0.01):
            self.Is_valid = False
            self.prev_target = None
            print("Is_valid = False")

        return execution_time
    
    def compare_lists(self, list1, list2):
        if len(list1) != len(list2): return 1000
        
        differences = [abs(a - b) for a, b in zip(list1, list2)]
        mae = sum(differences) / len(differences)
        return mae

    def run(self):
        self.reset()
        self.reset_target()
        execution_time = self.action()

        if(self.Is_valid):
            distance = self.euclidean_distance(self.init_XYZ, self.target_XYZ)
            angle = quaternion_angle_radians(self.init_robot_quaternion, self.target_quaternion)
            delta_of_6_axis = self.get_delta(self.init_joint_values, self.next_joint_values,6)
            delta_of_3_axis = self.get_delta(self.init_joint_values, self.next_joint_values,3)
            delta_joints = self.get_joint_delta(self.init_joint_values, self.next_joint_values)
            save_data = self.target_XYZ + self.target_quaternion+ [execution_time] + [distance] +[angle] + [delta_of_6_axis] + [delta_of_3_axis] + delta_joints
            self.csv_save(save_data)
        else:
            self.Is_valid = True

    def get_pose(self) -> list:
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def get_orientation(self) -> list:
        pose = self.move_group.get_current_pose().pose
        quaternion = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        r = R.from_quat(quaternion)
        euler_angles = r.as_euler('xyz', degrees=False)
        return quaternion, euler_angles

    def _initialize_csv(self):
        # 폴더가 없으면 생성
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        
        # 파일이 없으면 생성하고 열 이름 작성
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)
    
    def csv_save(self, data):
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def get_delta(self, a, b, length):
        result = 0
        for i in range(length):
            result += abs(a[i]-b[i])
        return result
    
    def get_joint_delta(self, a, b):
        result = []
        for i in range(len(a)):
            result.append(abs(a[i]-b[i]))
        return result
    
    def euclidean_distance(self, point1, point2):
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance
    

if __name__ == "__main__":
  
    # CSV 파일 경로
    file_path = 'data_points/input_list.csv'

    # CSV 파일을 읽어옴
    df = pd.read_csv(file_path)

    # 0, 1, 2번째 컬럼을 x, y, z로 리스트에 저장
    x = df.iloc[:, 0].tolist()
    y = df.iloc[:, 1].tolist()
    z = df.iloc[:, 2].tolist()

    qx = df.iloc[:, 6].tolist()
    qy = df.iloc[:, 7].tolist()
    qz = df.iloc[:, 8].tolist()
    qw = df.iloc[:, 9].tolist()

    # x, y, z 리스트를 합쳐서 2D 리스트로 변환
    xyz_list = [[x[i], y[i], z[i]] for i in range(len(x))]
    orientation = [[qx[i], qy[i], qz[i], qw[i]] for i in range(len(qx))]

    main = Ned2_control(xyz_list, orientation)
    while(1):
        main.run()