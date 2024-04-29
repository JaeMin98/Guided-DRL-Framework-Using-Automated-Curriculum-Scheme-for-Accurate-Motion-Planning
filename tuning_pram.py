import main as Training
import Config

Config.isExit_IfSuccessLearning = False
Config.num_steps = 20000

tau_list = [0.01, 0.005, 0.003]
lr_list = [0.001, 0.0015, 0.003]
batch_list = [1028, 512, 256, 128, 64]

for T in tau_list:
    Config.tau = T
    for L in lr_list:
        Config.lr = L
        for B in batch_list:
            Config.batch_size = B
            
            M = Training.Training_Robotic_arm()
            M.Run()