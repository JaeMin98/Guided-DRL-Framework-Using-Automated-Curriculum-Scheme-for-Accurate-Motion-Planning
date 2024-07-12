#! /usr/bin/env python3
# -*- coding: utf-8 -*-
#import general libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import moveit_msgs.msg
import geometry_msgs.msg
import math
#import ros|movit|gazrbo libraries
import rospy
import os
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
import csv


class CylindricalGridPointGenerator:
    def __init__(self):
        # 원기둥 범위 설정
        self.radius = 0.47  # 반지름
        self.height_range = (0.05, 0.64)  # 높이 범위
        self.density_r = 20  # 반지름 방향 분할 수
        self.density_theta = 120  # 각도 방향 분할 수
        self.density_h = 20  # 높이 방향 분할 수

    def generate_cylindrical_grid_points(self):
        r = np.linspace(0, self.radius, self.density_r)
        theta = np.linspace(0, 2*np.pi, self.density_theta)
        h = np.linspace(self.height_range[0], self.height_range[1], self.density_h)
        
        r, theta, h = np.meshgrid(r, theta, h)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = h
        
        grid_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        num_of_points = grid_points.shape[0]
        
        return grid_points, num_of_points

    def visualize_grid_points(self, grid_points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], c='b', marker='o')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    def run(self):
        # 원기둥 그리드 포인트 생성
        grid_points, num_of_points = self.generate_cylindrical_grid_points()
        print(f"총 포인트 개수: {num_of_points}")
        
        # 그리드 포인트 시각화
        # self.visualize_grid_points(grid_points)
        
        return grid_points, num_of_points

class Ned2_control(object):
    def __init__(self, grid_points, num_of_points):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        # Initialize a Robot Group
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("ned2")
        rospy.sleep(2)  # Wait for the Scene to fully load
        self.target = []
        self.Is_valid = True
        self.prev_target = None
        self.grid_points = grid_points
        self.num_of_points = num_of_points
        self.grid_iteration = 11242

    def reset(self) -> None:
        self.move_group.go([0]*6, wait=True)
        self.init_joint_values = self.move_group.get_current_joint_values()
        self.init_XYZ = self.get_pose()

    def reset_target(self) -> None:
        self.target = self.grid_points[self.grid_iteration]
        self.move_group.set_position_target(self.target)
        self.grid_iteration += 1
        print(f"Target: {self.target}, Grid Iteration: {self.grid_iteration}, Progress: {self.grid_iteration/self.num_of_points*100}%")
            
    def action(self):
        start_time = rospy.get_time()
        self.move_group.go(wait=True)
        end_time = rospy.get_time()
        self.move_group.clear_pose_targets()
        execution_time = end_time - start_time
        self.next_joint_values = self.move_group.get_current_joint_values()
        self.next_XYZ = self.get_pose()

        if(self.compare_lists(self.next_XYZ, self.target) > 0.01):
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
            distance = self.euclidean_distance(self.init_XYZ, self.next_XYZ)
            delta_of_6_axis = self.get_delta(self.init_joint_values, self.next_joint_values,6)
            delta_of_3_axis = self.get_delta(self.init_joint_values, self.next_joint_values,3)
            delta_joints = self.get_joint_delta(self.init_joint_values, self.next_joint_values)
            save_data = list(self.target) + [execution_time] + [distance] + [delta_of_6_axis] + [delta_of_3_axis] + delta_joints
            self.save(save_data)
        else:
            self.Is_valid = True

    def get_pose(self) -> list:
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def save(self, data):
        folder_path = "data_points"
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        csv_path = os.path.join(folder_path, 'grid_data_points_origin.csv')

        with open(csv_path, 'a', newline='') as file:
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
    

if __name__ == '__main__':
    # 원기둥 그리드 포인트 생성
    Grid = CylindricalGridPointGenerator()
    grid_points, num_of_points = Grid.run()
    main = Ned2_control(grid_points, num_of_points)
    while(1):
        main.run()