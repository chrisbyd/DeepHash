import os.path as osp
from collections import defaultdict
import numpy as np
import random
import os

img_dir_root = './data/VeRi'
train_list_dir = osp.join(img_dir_root,'train_test_split')
train_img_path = osp.join(img_dir_root,'image_train')
test_img_path = osp.join(img_dir_root,'image_test')
database_img_path = osp.join(img_dir_root,'image_query')

img_data_list = []
pid_set = set()

for _,_, file_name in os.walk(train_img_path):
    for f_name in file_name:
        pid = f_name[:4]
        camid = f_name[6:9]
        img_data_list.append((f_name,pid,camid))
        pid_set.add(pid)

pid2label = {pid: label for label, pid in enumerate(pid_set)}

train_text_path = osp.join(img_dir_root,'train.txt')
test_text_path = osp.join(img_dir_root,'test.txt')
database_text_path = osp.join(img_dir_root,'database.txt')

with open(train_text_path,'w') as f:
    for fname, pid, camid in img_data_list:
        img_path = osp.join('image_train',fname)
        label = pid2label[pid]
        f.write(img_path+' '+str(label)+ ' '+str(camid)+' '+'\n')

img_data_list = []
pid_set = set()
for _,_, file_name in os.walk(test_img_path):
    for f_name in file_name:
        pid = f_name[:4]
        camid = f_name[6:9]
        img_data_list.append((f_name,pid,camid))
        pid_set.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_set)}

with open(test_text_path,'w') as f:
    for fname, pid, camid in img_data_list:
        img_path = osp.join('image_test',fname)
        label = pid2label[pid]
        f.write(img_path+' '+str(label)+ ' '+str(camid)+' '+'\n')


img_data_list = []
pid_set = set()
for _,_, file_name in os.walk(database_img_path):
    for f_name in file_name:
        pid = f_name[:4]
        camid = f_name[6:9]
        img_data_list.append((f_name,pid,camid))
        pid_set.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_set)}

with open(database_text_path,'w') as f:
    for fname, pid, camid in img_data_list:
        img_path = osp.join('image_query',fname)
        label = pid2label[pid]
        f.write(img_path+' '+str(label)+ ' '+str(camid)+' '+'\n')


