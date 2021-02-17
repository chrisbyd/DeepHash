import numpy as np
import scipy.io as sio
import warnings
import data_provider.image as dataset
import model.dhn.dhn as model
import sys
from pprint import pprint
from util import results_to_excel, make_dirs
import os



warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define input arguments
lr = float(sys.argv[1])
output_dim = int(sys.argv[2])
iter_num = int(sys.argv[3])
cq_lambda = float(sys.argv[4])
alpha = float(sys.argv[5])
_dataset = sys.argv[6]
gpu = sys.argv[7]
log_dir = sys.argv[8]
data_root = sys.argv[9]
test_mode = sys.argv[10]

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_21': 21,
              'nuswide_81': 81, 'coco': 80, 'imagenet': 100, 'cifar10_zero_shot': 10,
              'vehicleID' : 13164, 'VeRi': 576}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000,
      'nuswide_21': 5000, 'imagenet': 5000, 'cifar10_zero_shot': 15000,
      'vehicleID' :5000, 'VeRi' :5000}

config = {
    'device': '/gpu:' + gpu,
    'max_iter': iter_num,
    'batch_size': 800,  # TODO
    'val_batch_size': 100,
    'decay_step': 5000,  # TODO     # Epochs after which learning rate decays.
    'learning_rate_decay_factor': 0.5,   # Learning rate decay factor.
    'learning_rate': lr,                 # Initial learning rate img.

    'output_dim': output_dim,
    'alpha': alpha,

    'R': Rs[_dataset],
    'model_weights': '../../architecture/pretrained_model/reference_pretrain.npy',

    'img_model': 'alexnet',
    'loss_type': 'normed_cross_entropy',  # normed_cross_entropy # TODO

    # if only finetune last layer
    'finetune_all': True,

    # CQ params
    'cq_lambda': cq_lambda,

    'label_dim': label_dims[_dataset],
    'img_tr': "../../data/{}/train.txt".format(_dataset),
    'img_te': "../../data/{}/test.txt".format(_dataset),
    'img_db': "../../data/{}/database.txt".format(_dataset),
    'save_dir': "./models/",
    'log_dir': log_dir,
    'dataset': _dataset
}

pprint(config)
# print(os.getcwd())
# train_img = dataset.import_train(data_root, config['img_tr'])
# model_weights = model.train(train_img, config)
#
#config['model_weights'] = model_weights
config['model_weights'] ='./models/lr_5e-05_cqlambda_0.0_alpha_10.0_dataset_vehicleID_hashbit_512.npy'
query_img, database_img = dataset.import_validation(data_root, config['img_te'], config['img_db'])
cmc, mAP = model.validation(database_img, query_img, config)
print("The cmc: Rank1:{}, Rank5:{}, Rank10:{} and the mAP is {}".format(cmc[0],cmc[4],cmc[9],mAP))
print(
    'The cmc: Rank1:{},Rank2:{},Rank3:{},Rank4:{} Rank5:{},Rank6:{},Rank7:{},Rank8:{},Rank9:{}, Rank10:{}'
    'Rank11:{},Rank12:{},Rank13:{},Rank14{},Rank15:{},Rank16:{},Rank17:{},Rank18:{},Rank19:{},Rank20:{},mAP is {}'.format(
        cmc[0], cmc[1], cmc[2], cmc[3], cmc[4], cmc[5], cmc[6], cmc[7], cmc[8], cmc[9], cmc[10],
        cmc[11], cmc[12],
        cmc[13], cmc[14],
        cmc[15], cmc[16], cmc[17], cmc[18], cmc[19], mAP))
results = [item for item in cmc[:20]] + [mAP]
model_name = 'DHN-{}'.format(config['output_dim'])
results_to_excel(results, model_name, config['dataset'])
# for key in maps:
#     print(("{}: {}".format(key, maps[key])))
pprint(config)
