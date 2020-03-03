import os
import  argparse
from cfg.default_config import  get_cfg_defaults
from  utils.comm import  setup_seed
from tools.train import train
import  pandas as pd
from tools.infer import infer_with_model_data_build
from sklearn.metrics import  accuracy_score
from dataset.tianchi_2020_dataset import parse_input_file

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--cfg',  type = str,
                     default = '/workspace/nCoV_sentence_simi/cfgs/ernie_for_cleaning_data.yml')
    arg.add_argument('--train_val_files', type = list,
                     default=[
                         './data/train_20200228.csv',
                         './data/dev_20200228.csv',
                     ])
    return arg.parse_args()


def maybe_train_model(args, cfg):
    df = pd.read_csv(args.train_val_files[0])
    for i in range(1, len(args.train_val_files)):
        d = pd.read_csv(args.train_val_files[i])
        df = pd.concat([df, d])
    merged_save_path = os.path.dirname(args.train_val_files[0]) + '/train_val_merge.csv'
    df.to_csv(merged_save_path)
    save_dir = cfg.TRAIN.save_dir
    train(cfg, merged_save_path, merged_save_path, save_dir)

def clean_by_model(cfg,
                   model_path,
                   data_path,
                   save_path,
                   drop_hard = True):
    preds, labels,pred_loggits = infer_with_model_data_build(cfg, model_path,
                                                             test_data_path=data_path)
    acc = accuracy_score(y_pred=preds, y_true=labels)
    print('acc: ', acc)

    all_data = parse_input_file(data_path)
    nn = 0
    hard_n = 0
    all_data_clean = []
    for i, p in enumerate(preds):
        if p == labels[i]:
            all_data_clean.append(all_data[i])
        else:
            v = pred_loggits[i]
            if v[p] > 0.9: ### maybe we can trust the model
                #print('modify', v, all_data[i])
                all_data[i][3] =  p ##modify the label
                all_data_clean.append(all_data[i])
                nn+=1
            else:  ##drop or get?  maybe the hard samples
                #print(v, all_data[i])
                if not drop_hard:
                    all_data_clean.append(all_data[i])
                else:
                    hard_n+=1
    print('we have correct {} samples by model!'.format(nn))
    print('we have drop {} hard samples by model!'.format(hard_n))
    print('get clean sampls: ', len(all_data_clean))
    df = pd.DataFrame(all_data_clean, columns=['category', 'query1', 'query2', 'label'])
    df.to_csv(save_path)
    print('saved in {}'.format(save_path))



if __name__ =='__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    args = get_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    setup_seed(1029)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TRAIN.device_ids_str
    maybe_train_model(args, cfg)

    model_path = cfg.TRAIN.save_dir +'/best.pth'

    file = args.train_val_files[0]
    clean_by_model(cfg, model_path,
                   file, file.replace('.csv','_clean.csv'),
                   drop_hard=True)
    file = args.train_val_files[1]
    clean_by_model(cfg, model_path,
                   file, file.replace('.csv','_clean.csv'),
                   drop_hard=True)




