#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import json, os
import numpy as np
from bert_serving.client import BertClient

'''
通过生成的word_index_dict.json={词1:id1,词2:id2,...}，将每个词
通过BERT转换为定长768向量，然后按照{id1:vector1,id2:vector2,...}存储在index_bert_dict.json
'''

bc = BertClient()

index_bert_vec_dic = {}  # 存储全部文档中出现的词及对应的bert embedding vector
with open('word_index_dict.json', 'r', encoding='utf-8') as f:
    word_index_dic = json.load(f)
    # 将出现的词转为bert embedding vector
    for key, values in word_index_dic.items():
        if values not in index_bert_vec_dic and key is not ' ':
            # print(type(bc.encode([key])[0]), key, values)
            index_bert_vec_dic[values] = bc.encode([key])[0].tolist()

if not os.path.exists("index_bert_dict.json"):
    with open('index_bert_dict.json', 'w', encoding='utf-8') as f:
        json.dump(index_bert_vec_dic, f, ensure_ascii=False)
print('step3：分词结束得到%s个词，分词结果及编码保存在word_bert_dict.npy' % (len(word_index_dic)))
