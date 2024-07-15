import sys
import os
import csv

def save_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def calculate_success_rate(success_each_UoCs):
    return [(sum(S)/len(S)) * 100 for S in success_each_UoCs]

def process_refer(path, module_name, csv_filename):
    folder_path = os.path.abspath(path)
    if folder_path not in sys.path:
        sys.path.append(folder_path)
    module = __import__(module_name)
    success_each_UoCs = module.run(evaluate=False)
    successrate_each_UoCs = calculate_success_rate(success_each_UoCs)
    save_csv(csv_filename, [[sr] for sr in successrate_each_UoCs])

# process_refer('./Refer1_DDPG', 'Refer1_Main', 'Refer1.csv')
process_refer('./Refer2_DDPG', 'Refer2_Main', 'Refer2.csv')
# process_refer('./Refer3_SAC', 'Refer3_Main', 'Refer3.csv')
