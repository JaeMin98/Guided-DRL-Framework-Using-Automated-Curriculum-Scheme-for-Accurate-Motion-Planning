import SAC_Robotic_arm_Training
import Config

Config.Current_Data_Selection_Ratio = 0.8

for _ in range(5):
    k = SAC_Robotic_arm_Training
    k.Run_Training()