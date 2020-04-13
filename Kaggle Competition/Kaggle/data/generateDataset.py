import shutil
import os 
import numpy as np

path = 'training/'
path_test = 'testing/'

labels = dict()

cnt = 0
with open ('labels_training.csv') as f:
    for i in f:
        cnt += 1
        if cnt == 1:
            continue
        i = i.split(',')
        labels[i[0]] = i[1][0]
        
print (len(labels))


if True:
    files = os.listdir(path)
    print ('files', len(files))

    f1 = 'solar_pv/'
    if not os.path.exists(f1):
        os.makedirs(f1)

    if not os.path.exists(f1 + path):
        os.makedirs(f1 + path)

    t_folder, f_folder = 'is_true/', 'is_false/'
    if not os.path.exists(f1 + path + t_folder):
        os.makedirs(f1 + path + t_folder)

    if not os.path.exists(f1 + path + f_folder):
        os.makedirs(f1 + path + f_folder)

    for i in range(len(files)):
        fi = files[i]
        li = labels[fi[:-4]]

        if li == '1':
            shutil.copyfile(path + fi , f1 + path + t_folder + fi [:-4] + '.png')
            print(path + fi , f1 + path + t_folder + fi [:-4] + '.png')
        elif li == '0':
            shutil.copyfile(path + fi , f1 + path + f_folder + fi [:-4] + '.png')
            print (path + fi , f1 + path + f_folder + fi [:-4] + '.png')
        else:
            exit(-1)
        print ()

