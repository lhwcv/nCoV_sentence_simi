import os
import torch
import csv
import json
from  dataset.base import  TextExampleForClassify
import logging
import tqdm
import pickle
import  pandas as pd
from  torch.utils.data import  TensorDataset,RandomSampler,DataLoader,SequentialSampler
logger = logging.getLogger(__name__)


def parse_input_file(path):
    df = pd.read_csv(path)
    lines = []
    for i, d in df.iterrows():
        try:
            label = str(int(d['label']))
            if label not in ['0', '1']:
                label = '0'
            lines.append([d['category'], d['query1'], d['query2'], label])
        except:
            print("WARN, line: {} data format not compelete!".format(int(i) + 1))
            continue
    return lines


class TianchiProcessor():
    """Processor for the Tianchi data set."""

    def _parse_input_file(self, path):
        df = pd.read_csv(path)
        lines = []
        for i, d in df.iterrows():
            try:
                label = str(int(d['label']))
                if label not in ['0', '1']:
                    label = '0'
                lines.append([d['category'], d['query1'], d['query2'], label ])
            except:
                print("WARN, line: {} data format not compelete!".format(int(i)+1)  )
                continue
        return lines

    def get_examples(self, data_path, tag ='train'):

        exmaples=self._create_examples(self._parse_input_file(data_path),
                                       set_type  = tag,with_flip = False)
        return  exmaples


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, with_flip = False):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            text_c = line[2]
            if set_type == 'test222':
                label = '0'
            else:
                label = line[3]
            examples.append(
                TextExampleForClassify(
                    guid=guid,
                    text_a=text_b,
                    text_b=text_c,
                    label=label,
                    whichCatgory = None#text_a
                ))
            if with_flip:
                guid = "%s-flip-%s" % (set_type, i)
                examples.append(
                    TextExampleForClassify(
                        guid=guid,
                        text_a=text_c,
                        text_b=text_b,
                        label=label,
                        whichCatgory=None#text_a
                    ))
        return examples

def load_and_cache_dataset(config, tokenizer, data_path,  tag='train'):

    processor = TianchiProcessor()
    examples = processor.get_examples(data_path, tag)
    features = []
    for example in tqdm.tqdm(examples):
        fea = example.tonkenize_to_feature(tokenizer, config.MODEL.max_seq_lenth)
        features.append(fea)

    # logger.info("Saving features into cached file %s", cached_features_file)
    # torch.save(features, cached_features_file)
    # logger.info("Saved!")

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_lens, all_labels)
    return dataset


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def get_train_val_dataloader(config, tokenizer, train_data_path, val_data_path):
    train_dataset = load_and_cache_dataset(config, tokenizer,train_data_path, tag='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=config.TRAIN.batch_size,
                                  collate_fn=collate_fn,
                                  num_workers = config.SYSTEM.NUM_WORKERS)

    val_dataset = load_and_cache_dataset(config, tokenizer,val_data_path, tag='dev')
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler,
                                  batch_size=config.TRAIN.batch_size,
                                  collate_fn=collate_fn,
                                  num_workers = config.SYSTEM.NUM_WORKERS)
    return  train_dataloader, val_dataloader

def get_test_dataloader(config, tokenizer, test_data_pah):
    test_dataset = load_and_cache_dataset(config, tokenizer, test_data_pah, tag='test')
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler,
                                  batch_size=config.TRAIN.batch_size,
                                  collate_fn=collate_fn,
                                  num_workers = config.SYSTEM.NUM_WORKERS)
    return  test_dataloader


