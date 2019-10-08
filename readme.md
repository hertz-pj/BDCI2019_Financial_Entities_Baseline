## 参考说明
BERT-BiLSTM-CRF-NER模型来源于：https://github.com/macanv/BERT-BiLSTM-CRF-NER  
数据处理部分参考水哥：https://github.com/finlay-liu/kaggle_public  
感谢各位大佬的开源

## 说明
随着互联网的飞速进步和全球金融的高速发展，金融信息呈现爆炸式增长。投资者和决策者在面对浩瀚的互联网金融信息时常常苦于如何高效的获取需要关注的内容。针对这一问题，金融实体识别方案的建立将极大提高金融信息获取效率，从而更好的为金融领域相关机构和个人提供信息支撑。  
比赛地址：https://www.datafountain.cn/competitions/361

## 运行说明
#### 环境
python 3.7.3  
ubuntu 18.04  
tensorflow 1.13.0  
关于tensorflow1.14会报'tensorflow.contrib.estimator' has no attribute 'stop_if_no_decrease_hook'的问题，原因是tf.contrib在1.14中已经被弃用。  
解决方案：将BERT-BiLSTM-CRF-NER/bert_base/train/bert_lstm_ner.py文件内第620行的tf.contrib.estimator.stop_if_no_decrease_hook替换为 tf.estimator.experimental.stop_if_no_decrease_hook即可正常运行。
#### 硬件
V100, 如果没有大显存，调小maxlen和batch size, 正常来说1080Ti够用
#### 运行
修改run.sh里面的各种路径，直接bash就可以了

## 结果
该版本线上0.277+，因为没有设置随机种子，所以应该会有所波动。
