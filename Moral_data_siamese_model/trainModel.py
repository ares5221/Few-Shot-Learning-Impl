#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import timeit, os
import json
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import pkuseg, jieba
import pandas as pd
import csv
from keras.layers import Input, Embedding, LSTM, Dense, MaxPooling1D,concatenate, GlobalMaxPooling1D
from keras.models import Model



def read_data():
    print('step1：read data...')
    df1 = pd.read_csv('./processLSTM_MLP_train_data/same_label_sentences_pairs_10w.csv',
                     header=None,sep=' ',error_bad_lines=False)
    df2 = pd.read_csv('./processLSTM_MLP_train_data/no_same_label_sentences_pairs_10w.csv',
                      header=None, sep=' ', error_bad_lines=False)
    # print(df1.head())
    # print(df2.head())
    res = pd.concat([df1, df2], axis=0,
                    ignore_index=True)  # 0表示竖项合并 1表示横项合并 ingnore_index重置序列index index变为0 1 2 3 4 5 6 7 8
    # print(res)
    res = np.array(res)
    indices = np.arange(res.shape[0])  # shuffle
    np.random.shuffle(indices)
    # print(indices)
    res = res[indices]
    print(len(res),type(res), res)
    np.save("data.npy", res)
    return res


def peredata(content_data):
    print('step2: 数据预处理获取label信息及分词处理...')
    data = content_data.tolist()
    train_label = []# 用于存储每对句子的是否相似的标签信息1，0
    index_bert_vec_dic = {}  # 存储全部文档中出现的词及对应的bert embedding vector
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    sentences1_seg_list = [[] for index in range(len(data))]#存储第一个句子和第二个句子的分词结果
    sentences2_seg_list = [[] for index in range(len(data))]
    is_get_data_info = False #flag set
    if is_get_data_info:
        get_data_info(data)

    # seg_text_data = [[] for index in range(len(content_data))]  # 存储文档分词后结果用于word2vec
    # 这里采用pkuseg的分词，如果要改jieba，不需要前面实例化声明, jieba.cut(sentence)
    qinghuaSeg = pkuseg.pkuseg()

    for i in range(len(data)):
        sen1 = data[i][0].split(',')[0]
        sen2 = data[i][0].split(',')[1]
        # print(len(data[i][0].split(',')),data[i][0].split(','))
        train_label.append( int(data[i][0].split(',')[2]) )
        seg_content_data1 = qinghuaSeg.cut(sen1)
        seg_content_data2 = qinghuaSeg.cut(sen2)
        #将分词结果保存到对应数组
        sentences1_seg_list[i] = seg_content_data1
        sentences2_seg_list[i] = seg_content_data2
        #将出现的词通过BERT转为向量
        #将出现的词编码，并将词与id的对应关系保存在word_index_dic
        for word1 in seg_content_data1:
            if word1 not in word_index_dic:
                word_index_dic[word1] = len(word_index_dic) + 1
        for word2 in seg_content_data2:
            if word2 not in word_index_dic:
                word_index_dic[word2] = len(word_index_dic) + 1
        if i % 1000 == 0:
            print('已经完成进度', i/200000)
    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)
    # save label info
    if not os.path.exists("train_label.npy"):
        np.save("train_label.npy", np.array(train_label))

    # 分词结束得到7282个词，分词结果及编码保存在word_bert_dict.json
    sentences1_data = []
    sentences2_data = []

    for ssl1 in range(len(sentences1_seg_list)):
        word_vec_list1 = []
        for word in sentences1_seg_list[ssl1]:
            word_vec_list1.append(word_index_dic[word])
        sentences1_data.append(word_vec_list1)
    for ssl2 in range(len(sentences2_seg_list)):
        word_vec_list2 = []
        for word in sentences1_seg_list[ssl2]:
            word_vec_list2.append(word_index_dic[word])
        sentences2_data.append(word_vec_list2)
    if not os.path.exists("sentence_data1.npy"):
        np.save("sentence_data1.npy", sentences1_data)
    if not os.path.exists("sentence_data2.npy"):
        np.save("sentence_data2.npy", sentences2_data)
    return sentences1_data, sentences2_data, train_label


def get_data_info(data):
    '''
    用于统计待处理的句子对的句子的信息，包括句子的最大分词后的词的个数，所有句子分词后词的平均个数
    :param data:
    :return:统计得到句子分词后的最大词的个数： 539 平均词的个数： 40.038145
    '''
    max_sentences_words_num = 0
    sentences_words_count = 0  # 统计全部句子的分词长度，计算句子分词后的平均词的个数
    qinghuaSeg = pkuseg.pkuseg()
    for i in range(len(data)):
        sen1 = data[i][0].split(',')[0]
        sen2 = data[i][0].split(',')[1]
        seg_content_data1 = qinghuaSeg.cut(sen1)
        seg_content_data2 = qinghuaSeg.cut(sen2)
        sentences_words_count += len(seg_content_data1)
        sentences_words_count += len(seg_content_data2)
        # print(seg_content_data1, seg_content_data2)
        if len(seg_content_data1) > max_sentences_words_num:
            max_sentences_words_num = len(seg_content_data1)
        if len(seg_content_data2) > max_sentences_words_num:
            max_sentences_words_num = len(seg_content_data2)
        if i %10000 ==0:
            print('已经完成进度',i/200000)
    print('统计得到句子分词后的最大词的个数：', max_sentences_words_num, '平均词的个数：', sentences_words_count/400000)



def build_model(sentences1_data, sentences2_data, train_label, test_data1, test_data2, test_label):
    print('step4: start build model...')
    batch_size_num = 100
    embed_size = 768  # 词向量维度
    max_len = 500 # 每句话的最大长度539，平均句子长度40
    max_words = 7282+1   # 统计得到该文档用到的词的个数7282
    sentences1_data = keras.preprocessing.sequence.pad_sequences(sentences1_data,
                                                                 padding='post',
                                                                 maxlen=max_len)
    sentences2_data = keras.preprocessing.sequence.pad_sequences(sentences2_data,
                                                                 padding='post',
                                                                 maxlen=max_len)
    # print(len(sentences1_data),sentences1_data[0])
    #load test data100
    test_data1 = keras.preprocessing.sequence.pad_sequences(test_data1,
                                                                 padding='post',
                                                                 maxlen=max_len)
    test_data2 = keras.preprocessing.sequence.pad_sequences(test_data2,
                                                                 padding='post',
                                                                 maxlen=max_len)


    embedding_matrix = np.zeros((max_words, embed_size))
    with open('index_bert_dict.json', 'r', encoding='utf-8') as f:
        index_to_bert_vector = json.load(f)
        # embedding_matrix[0] = np.zeros(768)
        for key in index_to_bert_vector:
            index = int(key)
            if index == max_words:
                embedding_matrix[0] = np.array(index_to_bert_vector.get(key))
            else:
                embedding_matrix[index] =np.array(index_to_bert_vector.get(key))

    input1 = Input(batch_shape=(batch_size_num,max_len))
    embed1 = Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input1)
    input2 = Input(batch_shape=(batch_size_num,max_len))
    embed2 = Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input2)
    print(input1.shape, input2.shape, embed1.shape, embed2.shape)
    shared_lstm = LSTM(64, return_sequences=True)  # 设置共享参数的LSTM层
    lstm_out1 = shared_lstm(embed1)
    lstm_out2 = shared_lstm(embed2)

    max_pooling1 = GlobalMaxPooling1D()(lstm_out1)
    max_pooling2 = GlobalMaxPooling1D()(lstm_out2)
    merge_vector = concatenate([max_pooling1, max_pooling2],axis=-1)
    mlp_out = Dense(1, activation=tf.nn.sigmoid)(merge_vector)
    model = Model([input1, input2], mlp_out)

    # model.save('m1.h5')
    model.summary()
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # train_test_split = int(len(sentences1_data) * 0.8)
    partial_x_train = [sentences1_data[:160000], sentences2_data[:160000]]
    partial_y_train = train_label[:160000]
    #使用20w构造数据中的一部分4w数据作为test数据
    # test_data = [sentences1_data[160000:], sentences2_data[160000:]]
    # test_labels = train_label[160000:]
    # 使用人工构造的论坛问答数据中的100条数据作为test数据 理论上效果会比val_acc差
    test_data = [test_data1, test_data2]
    test_labels = test_label

    history = model.fit(partial_x_train, partial_y_train,
                        epochs=50,
                        batch_size=batch_size_num,
                        validation_split=0.2
                        )

    model.save('m2.h5')
    model.save_weights('m3.h5')
    results = model.evaluate(test_data, test_labels, batch_size=batch_size_num)
    print('step5: 评估模型效果(损失-精度）：...', results)

    print('step6: predict test data for count...')
    predictions = model.predict(test_data, batch_size=batch_size_num)
    predict = np.argmax(predictions, axis=1)
    print(predict)
    with open('02bert_lstm_mlp_predict.csv', 'w', newline='', encoding='utf-8') as csvwriter:
        spamwriter = csv.writer(csvwriter, delimiter=' ')
        for pre_val in predict:
            spamwriter.writerow([pre_val])

    # print('step7: 开始绘图...')
    # history_dict = history.history
    # history_dict.keys()
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(acc) + 1)
    # plt.plot(epochs, loss, 'bo', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    # plt.clf()  # clear figure
    # plt.plot(epochs, acc, 'bo', label='Training acc')
    # plt.plot(epochs, val_acc, 'ro', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    # print('模型训练结束！！！！！')



def execute():
    #step 1
    if os.path.exists('data.npy'):
        data_list = np.load("data.npy",allow_pickle=True)
    else:
        data_list = read_data()
    # step 2.1 load ann data
    if os.path.exists('sentence_data1.npy') and os.path.exists('sentence_data2.npy') and os.path.exists('train_label.npy'):
        sentences1_data = np.load("sentence_data1.npy",allow_pickle=True)
        sentences2_data = np.load("sentence_data2.npy",allow_pickle=True)
        train_label = np.load("train_label.npy",allow_pickle=True)
    else:
        sentences1_data, sentences2_data, train_label = peredata(data_list)

    # step 2.2 load 100 条论坛问答数据作为test data
    if os.path.exists('test_data1.npy') and os.path.exists('test_data2.npy') and os.path.exists(
            'test_label.npy'):
        test_data1 = np.load("test_data1.npy", allow_pickle=True)
        test_data2 = np.load("test_data2.npy", allow_pickle=True)
        test_label = np.load("test_label.npy", allow_pickle=True)
    else:
        pass

    # step 3
    build_model(sentences1_data, sentences2_data, train_label, test_data1, test_data2, test_label)


if __name__ == '__main__':
    execute()
