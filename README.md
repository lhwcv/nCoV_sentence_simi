# nCoV-2019 related sentence similarity
If useful for you, maybe a star to encourage our work.
### introduce
ERNIE , RoBerta based model for sentence similarity

For example: 
 ```
387,支原体肺炎,支原体肺炎的症状及治疗方法是什么,肺炎衣原体与肺炎支原体有什么区别？,0
388,支原体肺炎,支原体肺炎的症状及治疗方法是什么,肺炎支原体培养及药敏的检验单怎么看？,0
389,支原体肺炎,支原体肺炎的症状及治疗方法是什么,小儿支原体与小儿支原体肺炎相同吗？,0
390,支原体肺炎,宝宝支原体肺炎感染的症状有哪些？,宝宝肺炎支原体感染的症状是什么？,1
391,支原体肺炎,宝宝支原体肺炎感染的症状有哪些？,宝宝支原体肺炎感染有什么症状？,1
```

### 95.2 acc online (simply choose the 1st fold, 1/6)
 - ERNIE 1.0
 - Nadam with 2.0*1e-5 lr
 - OHEM CE, with label smoothing
 - cosine lr scheduler with warmup
 - clean noise data by an overfitted model
 
### more tricks maybe 
 - simply change the model
 - add any 'word2vec' features
 - split into multipiece data,get N bert,<br/>
   using multiple  feature to train a tree based<br/>
   model, lightGBM, Xgboost...
 - for those hard example, maybe add the nearest sentence<br>
   (pair with label) for reference info, into bert
 - pseudo label
 
 - more open data(e.g  ping an CHIP 2019)
 
 - ...
 
## denpendency
 - opencv-python
 - pytorch >= 1.4
 - pandas
 - yacs
 - sklearn
 
## prepare
 - download the ernie (128 length) model from  https://github.com/nghuyong/ERNIE-Pytorch
 - but using the config in this repo at pretrained/ernie/
 
## train 
you maye change the data path, have a look at train.py test.py
```
export PYTHONPATH=./
sh train_pipeline.sh
```

## ref 
https://tianchi.aliyun.com/competition/entrance/231776/introduction?spm=5176.12281949.1003.4.21eb2448atCLQk