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
from moveit_msgs.msg import Constraints, JointConstraint
import csv
from geometry_msgs.msg import Pose


class Ned2_control(object):
    def __init__(self):
        super(Ned2_control, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        # Initialize a Robot Group
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("h2017")
        rospy.sleep(2)  # Wait for the Scene to fully load
        self.target = []
        self.Is_valid = True
        self.prev_target = None
        
        self.targets = self.load_poses_from_csv('robot_pose_data.csv')
        self.target_count = 0


    def load_poses_from_csv(self, file_path: str) -> list:
        poses = []
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 7:  # Ensure the row has the correct number of elements
                    pose = Pose()
                    pose.position.x = float(row[0])
                    pose.position.y = float(row[1])
                    pose.position.z = float(row[2])
                    pose.orientation.x = float(row[3])
                    pose.orientation.y = float(row[4])
                    pose.orientation.z = float(row[5])
                    pose.orientation.w = float(row[6])
                    poses.append(pose)  # Add each pose to the list
        return poses

    def reset(self) -> None:
        self.move_group.go([0]*6, wait=True)
        self.init_joint_values = self.move_group.get_current_joint_values()
        self.init_XYZ = self.get_pose()

    def reset_target(self) -> None:
        current_row = self.targets[self.target_count]
        self.move_group.set_pose_target(current_row)
        self.target = [current_row.position.x, current_row.position.y, current_row.position.z]
        
        average_node = 0
        for _ in range(10):
            average_node += len(self.move_group.plan()[1].joint_trajectory.points)
        self.average_node = average_node/10

        self.target_count += 1

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
            save_data = self.target + [execution_time] + [distance] + [delta_of_6_axis] + [delta_of_3_axis] + delta_joints + [self.average_node]
            self.save(save_data)
        else:
            self.Is_valid = True

    def get_pose(self) -> list:
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z]
    
    def save(self, data):
        folder_path = "data_points"
        if not os.path.exists(folder_path): os.makedirs(folder_path)
        csv_path = os.path.join(folder_path, 'data_points_origin.csv')

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
    

if __name__ == "__main__":
  main = Ned2_control()
  while(1):
      main.run()