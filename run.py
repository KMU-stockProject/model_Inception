import os

current_dir = os.getcwd()
data_list = os.listdir(os.path.join(current_dir, 'data', 'pklData', 'training'))

for i in data_list:
    os.system('python3 main.py {}'.format(i))
