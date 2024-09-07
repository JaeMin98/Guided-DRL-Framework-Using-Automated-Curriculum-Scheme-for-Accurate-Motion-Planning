#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import rospy
import os
import moveit_commander
from moveit_commander import PlanningSceneInterface, RobotCommander, MoveGroupCommander
from moveit_msgs.msg import Constraints, JointConstraint
import csv
import numpy as np
from geometry_msgs.msg import Pose
from tf.transformations import euler_from_quaternion
import random

def radian_to_degree(radian):
    return radian * (180 / math.pi)

def degree_to_radian(degree):
    return degree * (math.pi / 180)

def generate_random_number(a):
    return random.uniform(-a, a)

class Robotarm_contorl():
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)
        # Initialize a Robot Group
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander("h2017")
        rospy.sleep(2)  # Wait for the Scene to fully load

        self.throttle = [180,110,140]

    def reset(self):
        self.move_group.go([0]*6)

    def get_random(self):
        result = []
        for i in range(3):
            result.append(degree_to_radian(generate_random_number(self.throttle[i])))
        return result

    def action(self, angle):
        angle += [0,0,0]
        self.move_group.go(angle)

    def random_action(self):
        self.action(self.get_random())

    def get_pose(self) -> list:
        pose = self.move_group.get_current_pose().pose
        return [pose.position.x, pose.position.y, pose.position.z, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    def run(self):
        while(1):
            self.random_action()
            pose = self.get_pose()
            self.save_pose_to_csv(pose)

    def save_pose_to_csv(self, pose):
        # CSV 파일에 포즈 데이터를 저장
        with open('robot_pose_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(pose)

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


def main():
    h2017 = Robotarm_contorl()
    print(h2017.load_poses_from_csv('robot_pose_data.csv')[0])

if __name__ == '__main__':
    main()