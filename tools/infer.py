import  torch
import  tqdm
import  numpy as np
from models import  build_model_and_tokenizer
from dataset.tianchi_2020_dataset import get_test_dataloader
def _batch_trans(batch, device):
    batch = tuple(t.to(device) for t in batch)
    batch_data ={
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'token_type_ids': batch[2],
        'labels': batch[3]
    }
    return batch_data


def infer_dataloader(models_list, dloader, device, return_loggits = False):
    for m  in models_list:
        m.eval()
        m.to(device)
    all_preds_loggits = None
    all_labels = None
    for batch in tqdm.tqdm(dloader):
        with torch.no_grad():
            batch_data = _batch_trans(batch,device)
            outputs = models_list[0](
                input_ids=batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                token_type_ids=batch_data['token_type_ids'],
            )
            pred_loggits = outputs[0]
            for i in range(1, len(models_list)):
                pred_loggits += models_list[i](
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    token_type_ids=batch_data['token_type_ids'],
                )[0]
            pred_loggits = pred_loggits.softmax(dim=-1)
        if all_preds_loggits is None:
            all_preds_loggits = pred_loggits.detach().cpu().numpy()
            all_labels = batch_data['labels'].detach().cpu().numpy()
        else:
            all_preds_loggits = np.append(all_preds_loggits, pred_loggits.detach().cpu().numpy(), axis=0)
            all_labels = np.append(all_labels, batch_data['labels'].detach().cpu().numpy(), axis=0)
    all_preds = np.argmax(all_preds_loggits, axis=1)

    if return_loggits:
        return  all_preds, all_labels,all_preds_loggits
    return  all_preds,all_labels


def infer_with_model_data_build(cfg, model_path, test_data_path):
    model_type = cfg.MODEL.model_type
    model, tokenizer = build_model_and_tokenizer(model_type)
    dataloader = get_test_dataloader(cfg, tokenizer, test_data_path)
    print('samples: ', len(dataloader.dataset))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = []
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path), strict=True)
    else:
        map_location = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=map_location),
                                  strict=True)
    models.append(model)
    preds, labels,pred_loggits = infer_dataloader(models, dataloader, device,return_loggits=True)
    return  preds, labels,pred_loggits


