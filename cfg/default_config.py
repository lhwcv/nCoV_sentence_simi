from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 6

_C.DATA = CN()
_C.DATA.VOCAB_FILE = ''
_C.DATA.vocab_size = 21128
_C.DATA.data_dir = '/workspace/nCoV_sentence_simi/data/'
_C.DATA.test_file = '/workspace/nCoV_sentence_simi/data/dev_20200228.csv'

_C.TRAIN = CN()
_C.TRAIN.do_train = True
_C.TRAIN.batch_size = 16
_C.TRAIN.save_dir = '/workspace/wkdir/albert_base/'
_C.TRAIN.gradient_accumulation_steps = 1
_C.TRAIN.num_train_epochs = 5
_C.TRAIN.warmup_proportion = 0.1
_C.TRAIN.learning_rate = 2*1e-5
_C.TRAIN.weight_decay = 0.0
_C.TRAIN.device_ids_str = "0"
_C.TRAIN.device_ids = [0]
_C.TRAIN.log_steps = 200
_C.TRAIN.adam_epsilon = 1e-6


_C.MODEL = CN()
_C.MODEL.model_type = 'ernie'
_C.MODEL.load_trained = False
_C.MODEL.load_trained_path = ''
_C.MODEL.num_labels = 2
_C.MODEL.max_seq_lenth = 64


def get_cfg_defaults(merge_from = None):
  cfg =  _C.clone()
  if merge_from is not None:
      cfg.merge_from_other_cfg(merge_from)
  return cfg