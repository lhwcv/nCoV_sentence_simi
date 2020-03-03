import os
import  argparse
from cfg.default_config import  get_cfg_defaults
from  utils.comm import  setup_seed
from tools.split_data import split_data
from tools.train import train



def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--cfg',  type = str,
                     default = '/workspace/nCoV_sentence_simi/cfgs/ernie.yml')
    arg.add_argument('--train_val_files', type = list,
                     default=[
                         './data/train_20200228_clean.csv',
                         './data/dev_20200228_clean.csv',
                     ])
    return arg.parse_args()


def kfold_train(cfg, args,n_fold = 6, train_folds= [0,1,2,3,4,5]):

    split_data(args.train_val_files, n_fold,
               save_dir = os.path.dirname(args.train_val_files[0]),
               random_state=666)
    score_avg = 0
    nn = 0
    for i in range(n_fold):
        if i not in train_folds:
            continue
        train_data_path = os.path.join(cfg.DATA.data_dir, 'train_fold_{}.csv'.format(i))
        val_data_path = os.path.join(cfg.DATA.data_dir, 'val_fold_{}.csv'.format(i))
        save_dir = '{}/fold_{}/'.format(cfg.TRAIN.save_dir, i)
        score = train(cfg, train_data_path, val_data_path, save_dir)
        score_avg += score
        nn += 1
    print('avg score: ', score_avg / nn)


if __name__ =='__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    args = get_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    setup_seed(1029)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.device_ids_str
    train_folds = [0]#[0,1,2,3,4,5]
    kfold_train(cfg,args, n_fold=6, train_folds= train_folds)

    #train_data_path = args.train_val_files[0]
    #val_data_path   = args.train_val_files[1]
    #save_dir = cfg.TRAIN.save_dir
    #train(cfg, train_data_path, val_data_path, save_dir)
