import numpy as np
import os.path as osp
from collections import defaultdict
import numpy as np
import random

img_dir_root = './data/vehicleID'
train_list_dir = osp.join(img_dir_root,'train_test_split')
train_list_path = osp.join(train_list_dir,'train_list.txt')
test_list_path = osp.join(train_list_dir, 'test_list_')

biggest_pid = 0
pset = set()

name_dict = defaultdict()

with open(train_list_path) as ftrain:
    train_data = ftrain.readlines()
    pid_list = []
    name_list = []
    for data in train_data:
        name, pid = data.split(" ")
        pid = int(pid)
        pid_list.append(pid)
        name_list.append((name,pid))

def convert_pid_to_label(pids):
    p_set = set(pids)
    p_dict = defaultdict()
    for index, pid in enumerate(p_set):
        p_dict[pid] = index
    return p_dict

pids = pid_list
label_dict = convert_pid_to_label(pids)


with open(osp.join(img_dir_root,'train.txt'), 'w') as f_train:
    for name, pid in name_list:
        path = osp.join("image",name +'.jpg')
        label = label_dict[pid]
        f_train.write(path+ ' '+ str(label) +'\n')

for item in ['800','1600','2400','3200']:
    pid_dic = defaultdict(list)
    pid_list = []
    name_list = []
    test_path = test_list_path + item + '.txt'
    with open(test_path) as tes:
        test_data = tes.readlines()
        for data_item in test_data:
            name, pid = data_item.split(" ")
            pid = int(pid)
            pid_dic[pid].append((name,pid))
            pid_list.append(pid)

    label_dict = convert_pid_to_label(pid_list)
    query_list = []
    gallery_list = []
    for pid in pid_list:
        name_list = pid_dic[pid]
        name_for_gallery = random.choice(name_list)
        gallery_list.append(name_for_gallery)
        name_list.remove(name_for_gallery)
        query_list.extend(name_list)

    with open(osp.join(img_dir_root,'test_'+item+'.txt'),'w') as f_test:
        for name, pid in query_list:
            path = osp.join("image",name + '.jpg')

            label = label_dict[pid]
            f_test.write(path + " "+ str(label) +'\n')

    with open(osp.join(img_dir_root,'database_'+item + '.txt'),'w') as f_data:
        for name , pid in gallery_list:
            path = osp.join("image", name+'.jpg')
            label = label_dict[pid]
            f_data.write(path + ' '+ str(label) + '\n')


