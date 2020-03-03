import  pandas as pd
from  sklearn.model_selection import KFold


def split_data(data_file_list, n = 4,
               save_dir = '/workspace/nCoV_sentence_simi/data/',
               random_state = 1029):
    df = pd.read_csv(data_file_list[0])
    for i in range(1, len(data_file_list)):
        d = pd.read_csv(data_file_list[i])
        df = pd.concat([df, d])
    print('total samples: ', len(df))
    kf = KFold(n_splits=n, shuffle=True,random_state=random_state)
    k = 0
    df_indexes = list(range(0, len(df)))
    for train_idx, val_idx in kf.split(df_indexes):
        save_path_train = '{}/train_fold_{}.csv'.format(save_dir,k)
        save_path_val = '{}/val_fold_{}.csv'.format(save_dir, k)
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        train_df.to_csv(save_path_train)
        val_df.to_csv(save_path_val)
        k+=1



if __name__ =='__main__':
    data_file_list = [
        '/workspace/nCoV_sentence_simi/data/dev_20200228_clean.csv',
        '/workspace/nCoV_sentence_simi/data/train_20200228_clean.csv'
    ]
    split_data(data_file_list,4)


