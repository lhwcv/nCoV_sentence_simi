import  argparse
from cfg.default_config import get_cfg_defaults
import numpy as np
from utils.comm import setup_seed
from tools.infer import infer_with_model_data_build
import pandas as pd
from sklearn.metrics import  accuracy_score,classification_report

def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--data_dir',  type = str,
                     default = '/workspace/nCoV_sentence_simi/data/')
    arg.add_argument('--test_file', type=str,
                     default='/workspace/nCoV_sentence_simi/data/val_fold_0.csv')
    arg.add_argument('--model_pths', type = list,
                     default= [
                         '/workspace/wkdir/ernie_2/fold_0/best.pth',
                         '/workspace/wkdir/ernie_2/fold_1/best.pth',
                         '/workspace/wkdir/ernie_2/fold_2/best.pth',
                         '/workspace/wkdir/ernie_2/fold_3/best.pth',
                         '/workspace/wkdir/ernie_2/fold_4/best.pth',
                         '/workspace/wkdir/ernie_2/fold_5/best.pth'
                    ])
    arg.add_argument('--cfg_files' , type = str,
                     default = [
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                         '/workspace/nCoV_sentence_simi/cfgs/ernie.yml',
                        ])
    arg.add_argument('--save_path', type = str,  default = '/test.pred.csv')
    return arg.parse_args()




if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    setup_seed(1029)
    args = get_args()

    cfg = get_cfg_defaults()
    print(args.cfg_files[0])
    cfg.merge_from_file(args.cfg_files[0])
    cfg.DATA.data_dir = args.data_dir
    cfg.DATA.test_file = args.test_file

    labels = None
    all_pred_loggits = []
    for i,path in enumerate(args.model_pths):
        cfg.merge_from_file(args.cfg_files[i])
        _, labels,pred_loggits = infer_with_model_data_build(cfg, path,args.test_file)
        all_pred_loggits.append(pred_loggits)

    all_pred_loggits = np.array(all_pred_loggits)
    all_pred_loggits = all_pred_loggits.sum(axis=0)
    print(all_pred_loggits.shape)
    preds = np.argmax(all_pred_loggits, axis=1)


    # if True:
    #     acc = accuracy_score(y_pred=preds, y_true=labels)
    #     print('acc: ',acc)

    df = pd.DataFrame({
        'id': range(len(preds)),
        'label': preds
    })
    df.to_csv(args.save_path,index=None)





