import os
import argparse
import warnings
import numpy as np
import scipy.io as sio
import model.dch as model
import data_provider.image as dataset
from util import results_to_excel, make_dirs

from pprint import pprint

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Triplet Hashing')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float)
parser.add_argument('--output-dim', default=64, type=int)   # 256, 128
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--bias', default=0.0, type=float)
parser.add_argument('--gamma', default=20, type=float)
parser.add_argument('--iter-num', default=2000, type=int)
parser.add_argument('--q-lambda', default=0, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--gpus', default='0', type=str)
parser.add_argument('--log-dir', default='tflog', type=str)
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-vb', '--val-batch-size', default=16, type=int)
parser.add_argument('--decay-step', default=10000, type=int)
parser.add_argument('--decay-factor', default=0.1, type=float)

parser.add_argument('--test_mode', default= '800', type=str)

tanh_parser = parser.add_mutually_exclusive_group(required=False)
tanh_parser.add_argument('--with-tanh', dest='with_tanh', action='store_true')
tanh_parser.add_argument('--without-tanh', dest='with_tanh', action='store_false')
parser.set_defaults(with_tanh=True)

parser.add_argument('--img-model', default='alexnet', type=str)
parser.add_argument('--model-weights', type=str,
                    default='../../architecture/pretrained_model/reference_pretrain.npy')
parser.add_argument('--finetune-all', default=True, type=bool)
parser.add_argument('--save-dir', default="./models/", type=str)
parser.add_argument('--data-dir', default="../../data/", type=str)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

label_dims = {'cifar10': 10, 'cub': 200, 'nuswide_81': 81, 'coco': 80, 'vehicleID': 13164,
              'VeRi': 576}
Rs = {'cifar10': 54000, 'nuswide_81': 5000, 'coco': 5000,'vehicleID':50, 'VeRi':50}
args.R = Rs[args.dataset]
args.label_dim = label_dims[args.dataset]

args.img_tr = os.path.join(args.data_dir, args.dataset, "train.txt")
args.img_te = os.path.join(args.data_dir, args.dataset, "test.txt")
args.img_db = os.path.join(args.data_dir, args.dataset, "database.txt")

pprint(vars(args))

data_root = os.path.join(args.data_dir, args.dataset)
query_img, database_img = dataset.import_validation(data_root, args.img_te, args.img_db)

# if not args.evaluate:
#     train_img = dataset.import_train(data_root, args.img_tr)
#     model_weights = model.train(train_img, database_img, query_img, args)
#     args.model_weights = model_weights
args.model_weights = './models/lr_0.005_cqlambda_0_alpha_0.5_bias_0.0_gamma_20_dataset_vehicleID_hashbit_512.npy'

#maps = model.validation(database_img, query_img, args)
cmc, mAP = model.validation(database_img, query_img, args)
print(
    'The cmc: Rank1:{},Rank2:{},Rank3:{},Rank4:{} Rank5:{},Rank6:{},Rank7:{},Rank8:{},Rank9:{}, Rank10:{}'
    'Rank11:{},Rank12:{},Rank13:{},Rank14{},Rank15:{},Rank16:{},Rank17:{},Rank18:{},Rank19:{},Rank20:{},mAP is {}'.format(
        cmc[0], cmc[1], cmc[2], cmc[3], cmc[4], cmc[5], cmc[6], cmc[7], cmc[8], cmc[9], cmc[10],
        cmc[11], cmc[12],
        cmc[13], cmc[14],
        cmc[15], cmc[16], cmc[17], cmc[18], cmc[19], mAP))
results = [item for item in cmc[:20]] + [mAP]
model_name = 'DCH-{}'.format(args.output_dim)
results_to_excel(results, model_name, args.dataset)
# for key in maps:
#     print(("{}\t{}".format(key, maps[key])))

pprint(vars(args))
