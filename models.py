import os
from transformers import BertTokenizer, BertForSequenceClassification

DIR = os.path.dirname(__file__)

def build_ernie_model():
    model = BertForSequenceClassification.from_pretrained(DIR+'/pretrained/ernie/')
    return model

def build_ernie_model_and_tokennizer():
    model = BertForSequenceClassification.from_pretrained(DIR+'/pretrained/ernie/')
    tokenizer = BertTokenizer.from_pretrained(DIR+'/pretrained/ernie/')
    return model, tokenizer

def build_roberta_model_and_tokenizer():
    model = BertForSequenceClassification.from_pretrained(DIR+'/pretrained/chinese_roberta_wwm_large_ext_pytorch/')
    tokenizer = BertTokenizer.from_pretrained(DIR+'/pretrained/chinese_roberta_wwm_large_ext_pytorch/')
    return model, tokenizer

def build_model_and_tokenizer(model_type):
    if model_type=='ernie':
        return  build_ernie_model_and_tokennizer()
    else:
        return  build_roberta_model_and_tokenizer()