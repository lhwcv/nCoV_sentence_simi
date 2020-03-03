import  os
import  numpy as np
import torch
from torch.optim.optimizer import Optimizer
from utils.logger import TxtLogger
from torch.utils.data import  Dataset, DataLoader
import  tqdm
from sklearn.metrics import matthews_corrcoef, f1_score
from utils.meter import AverageMeter
from  cfg.default_config import  get_cfg_defaults
from sklearn.metrics import classification_report,accuracy_score

class Trainer():
    def __init__(self, model : torch.nn.Module,
                 loss_fn : torch.nn.Module,
                 optimizer: Optimizer,
                 scheduler,
                 logger : TxtLogger,
                 save_dir : str,
                 val_steps = 1000,
                 log_steps = 100,
                 device_ids = [0,1],
                 gradient_accum_steps = 1,
                 max_grad_norm = 1.0,
                 batch_to_model_inputs_fn = None,
                 early_stop_n = 30,
                 ):
        self.model  = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.val_steps = val_steps
        self.log_steps = log_steps
        self.logger = logger
        self.device_ids =device_ids
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn  = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0

    def step(self,step_n,  batch_data : dict):
        outputs = self.model(
            input_ids = batch_data['input_ids'],
            attention_mask = batch_data['attention_mask'],
            token_type_ids = batch_data['token_type_ids'],
            labels = batch_data['labels']
        )
        label = batch_data['labels']
        loggits = outputs[1]
        loss = self.loss_fn(loggits, label)
        loss = loss.mean() #+ outputs[0].mean()## add inner loss
        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if (step_n + 1) % self.gradient_accum_steps == 0:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        return  loss

    def _accuracy(self,preds, labels):
        return accuracy_score(y_pred=preds, y_true=labels)

    def _acc_and_f1(self, preds, labels):
        acc = self._accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_f1_avg": (acc + f1) / 2,
        }

    def _batch_trans(self, batch):
        batch = tuple(t.to(self.device_ids[0]) for t in batch)
        if self.batch_to_model_inputs_fn is None:
            batch_data = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }
        else:
            batch_data = self.batch_to_model_inputs_fn(batch)
        return  batch_data

    def val(self, val_dataloader : DataLoader):
        eval_loss = 0.0
        all_preds = None
        all_labels = None

        self.model.eval()
        for batch in tqdm.tqdm(val_dataloader):
            batch = tuple(t.to(self.device_ids[0]) for t in batch)
            with torch.no_grad():
                batch_data = self._batch_trans(batch)
                outputs = self.model(
                    input_ids=batch_data['input_ids'],
                    attention_mask=batch_data['attention_mask'],
                    token_type_ids=batch_data['token_type_ids'],
                    labels = batch_data['labels']
                )
                loss, pred_loggits = outputs[:2]
                #loss = self.loss_fn(pred_loggits, batch_data["labels"])
                eval_loss += loss.mean().item()
            if all_preds is None:
                all_preds = pred_loggits.detach().cpu().numpy()
                all_labels = batch_data['labels'].detach().cpu().numpy()
            else:
                all_preds = np.append(all_preds, pred_loggits.detach().cpu().numpy(), axis=0)
                all_labels = np.append(all_labels, batch_data['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / len(val_dataloader)
        all_preds = np.argmax(all_preds, axis=1)
        self.logger.write("steps: {} ,mean eval loss : {:.4f} ". \
                                      format(self.global_step, eval_loss))
        print("all preds shape: ", all_preds.shape)
        result =   self._acc_and_f1(all_preds, all_labels)
        return result

    def train(self, train_dataloader : DataLoader,
              val_dataloader : DataLoader,
              epoches = 100):
        best_score = 0
        early_n = 0

        for epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()
            train_data_iter = tqdm.tqdm(train_dataloader)
            for batch in train_data_iter:
                self.model.train()
                batch_data = self._batch_trans(batch)
                train_loss = self.step(step_n, batch_data)
                train_avg_loss.update(train_loss.item(),1)
                status = '[{0}] lr = {1:.7f} batch_loss = {2:.3f} avg_loss = {3:.3f} '.format(
                    epo + 1, self.scheduler.get_lr()[0],
                    train_loss.item(), train_avg_loss.avg )
                #if step_n%self.log_steps ==0:
                #    print(status)
                train_data_iter.set_description(status)
                step_n +=1
                if self.global_step % self.val_steps == 0:
                    ## val
                    m = self.val(val_dataloader)
                    acc = m['acc']
                    if best_score < acc:
                        early_n = 0
                        best_score = acc
                        model_path = os.path.join(self.save_dir, 'best.pth')
                        torch.save(self.model.state_dict(), model_path)
                    else:
                        early_n += 1
                    self.logger.write("steps: {} ,mean ap : {:.4f} , best ap: {:.4f}". \
                                      format(self.global_step, acc, best_score))
                    self.logger.write(str(m))
                    self.logger.write("=="*50)

                    if early_n > self.early_stop_n:
                        return best_score
        return  best_score








