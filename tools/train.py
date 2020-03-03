import os
import  torch
from dataset.tianchi_2020_dataset import  get_train_val_dataloader,get_test_dataloader
from cfg.default_config import  get_cfg_defaults
from tools.simple_trainer import Trainer
from  loss.softmaxloss import  CrossEntropyLabelSmooth,CrossEntropyLabelSmooth_OHEM
from  torch.optim.lr_scheduler import  LambdaLR
from  torch.optim import Adam
from  utils.logger import  TxtLogger
from  utils.comm import  setup_seed,create_dir
from optim.optimizer.adamw import AdamW
from optim.optimizer.nadam import  Nadam
from optim.optimizer.lookahead import  Lookahead

from optim.lr_scheduler import (get_linear_schedule_with_warmup,
                                get_cosine_schedule_with_warmup,
                                get_cosine_with_hard_restarts_schedule_with_warmup)
import  numpy as np
import  random
from transformers import BertTokenizer, BertModel,BertForSequenceClassification
from models import  *
from tools.split_data import split_data


def train(cfg, train_data_path, val_data_path, save_dir):

    model, tokenizer = build_ernie_model_and_tokennizer()
    model = model.cuda()
    #model, tokenizer = build_roberta_model_and_tokenizer()
    train_dataloader, val_dataloader = \
        get_train_val_dataloader(cfg, tokenizer, train_data_path,val_data_path)
    print('train samples: ', len(train_dataloader.dataset))

    loss_fn = CrossEntropyLabelSmooth_OHEM(2,ohem_ratio=0.7)
    #loss_fn = CrossEntropyLabelSmooth(2, 0.0)

    num_training_steps = len(train_dataloader) // cfg.TRAIN.gradient_accumulation_steps * cfg.TRAIN.num_train_epochs
    cfg.TRAIN.warmup_steps = int(num_training_steps * cfg.TRAIN.warmup_proportion)

    finetuned_parameters = list(map(id, model.classifier.parameters()))
    other_parameters = (p for p in model.parameters() if id(p) not in finetuned_parameters)
    parameters = [
        {'params': other_parameters, 'lr': 1.0 * cfg.TRAIN.learning_rate},
        {'params': model.classifier.parameters()}
        ]

    optimizer_grouped_parameters = parameters
    #optimizer = AdamW(params=optimizer_grouped_parameters,
    #                  lr = cfg.TRAIN.learning_rate)
    optimizer = Nadam(params=optimizer_grouped_parameters,
                    lr=cfg.TRAIN.learning_rate)
    #optimizer = Lookahead(optimizer)

    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps = cfg.TRAIN.warmup_steps,
    #                                             num_training_steps = num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=cfg.TRAIN.warmup_steps,
                                                num_training_steps=num_training_steps,
                                                num_cycles = 0.5)

    # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=cfg.TRAIN.warmup_steps,
    #                                             num_training_steps=num_training_steps,
    #                                             num_cycles = 1.0)
    create_dir(save_dir)
    logger = TxtLogger(save_dir+"/logger.txt")


    # if cfg.MODEL.load_trained:
    #     logger.write("load from  {}".format(cfg.MODEL.load_trained_path))
    #     model.load_state_dict(torch.load(cfg.MODEL.load_trained_path),
    #                           strict = False)


    trainer = Trainer(
        model = model,
        loss_fn  = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        logger = logger,
        save_dir = save_dir,
        val_steps = len(train_dataloader),
        log_steps = cfg.TRAIN.log_steps,
        device_ids = cfg.TRAIN.device_ids,
        gradient_accum_steps = 1,
        max_grad_norm = 10.0,
        batch_to_model_inputs_fn = None )

    score = trainer.train(train_dataloader, val_dataloader, cfg.TRAIN.num_train_epochs)
    return score
