import sys
import os
import csv

def save_csv(filename, data):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def calculate_success_rate(success_each_UoCs):
    return [(sum(S)/len(S)) * 100 for S in success_each_UoCs]

def process_refer(path, module_name, model_name, csv_filename):
    folder_path = os.path.abspath(path)
    if folder_path not in sys.path:
        sys.path.append(folder_path)

    module = __import__(module_name)
    success_each_UoCs = module.run(model_name, evaluate=False)
    successrate_each_UoCs = calculate_success_rate(success_each_UoCs)
    save_csv(csv_filename, [[sr] for sr in successrate_each_UoCs])

# process_refer('./Refer0_Non_CL', 'Refer0_Main', model_name, 'Non_CL.csv')


# models = ['Reference_models/Refer1/Refer1']
# for model_name in models :
#     process_refer('./Refer1_DDPG', 'Refer1_Main', model_name, f'{model_name}.csv')

# models = ['Reference_models/Refer2/Refer2_1',
#           'Reference_models/Refer2/Refer2_2',
#           'Reference_models/Refer2/Refer2_3',
#           'Reference_models/Refer2/Refer2_4',
#           'Reference_models/Refer2/Refer2_5']

models = ['Reference_models/Refer2/Refer2_4',
          'Reference_models/Refer2/Refer2_5']
for model_name in models :
    process_refer('./Refer2_DDPG', 'Refer2_Main', model_name, f'{model_name}.csv')


# models = ['Reference_models/Refer3_algorithm8/Refer3_algorithm8_3.tar',
#           'Reference_models/Refer3_algorithm8/Refer3_algorithm8_4.tar',
#           'Reference_models/Refer3_algorithm8/Refer3_algorithm8_1.tar']
# for model_name in models :
#     process_refer('./Refer3_SAC', 'Refer3_Main', model_name, f'{model_name}.csv')
