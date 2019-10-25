#!/usr/bin/env python
# _*_ coding:utf-8 _*_
from bert_serving.client import BertClient
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import pkuseg
import json
import random

#BERT将一个句子分词后，每个词转为768向量后，输入BiLSTM+attation+polling+MLP做分类

def create_data():
    train_path = './../data/sentences_background'
    train_file_list = os.listdir(train_path)
    print('$$$$$$$$$', train_path, train_file_list)
    loop_num = 100 #循环产生随机句子，循环100次，后面可调整训练数据规模
    pair_num_each = 300 #每次生成300对，共3w对训练数据
    way_num, shot_num = 5, 5 #当前采用5-way-5-shot采样策略，每次取5类，每类取5个句子
    for iter_index in range(loop_num):
        selected_labels_indexes = [random.randint(0, len(train_file_list) - 1) for i in range(way_num)]
        label_sens = [] #存储类名-句子，每次选出25个句子
        for index in selected_labels_indexes:
            curr_file_name = train_file_list[index]
            cur_dir = os.path.join(train_path,curr_file_name)
            curr_file_sens = []
            with open(cur_dir, 'r', encoding='utf-8') as fcsv:
                csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
                for data_line in csv_reader:
                    curr_file_sens.append(data_line)
            selected_sens_index = [random.randint(0, len(curr_file_sens) - 1) for i in range(shot_num)]
            for sen_index in selected_sens_index:
                label_sens.append([curr_file_name, curr_file_sens[sen_index]])
        # print(len(label_sens), label_sens)
        #从25个句子中随机选两个作为一对，选出300对作为训练数据
        for pair_num_index in range(pair_num_each):
            selected_sen_idx = [random.randint(0, len(label_sens) - 1) for i in range(2)]
            pair1 = label_sens[selected_sen_idx[0]]
            pair2 = label_sens[selected_sen_idx[1]]
            if pair1[0] == pair2[0]:#siamese_pairs.csv
                # print(pair2[0], pair2[1], pair2[1][0])
                with open('siamese_pairs.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ')
                    spamwriter.writerow([pair1[1][0] + ',' + pair2[1][0] + ',' + '1'])
            else:
                with open('siamese_pairs.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ')
                    spamwriter.writerow([pair1[1][0] + ',' + pair2[1][0] + ',' + '0'])


def create_test_data():
    test_path = './../data/sentences_evaluation'
    test_file_list = os.listdir(test_path)
    print('$$$$$$$$$', test_path, test_file_list)
    loop_num = 100  # 循环产生随机句子，循环100次，后面可调整训练数据规模
    pair_num_each = 300  # 每次生成300对，共3w对训练数据
    way_num, shot_num = 5, 5  # 当前采用5-way-5-shot采样策略，每次取5类，每类取5个句子
    for iter_index in range(loop_num):
        selected_labels_indexes = [random.randint(0, len(test_file_list) - 1) for i in range(way_num)]
        label_sens = []  # 存储类名-句子，每次选出25个句子
        other_sentences = []
        for index in selected_labels_indexes:
            curr_file_name = test_file_list[index]
            cur_dir = os.path.join(test_path, curr_file_name)
            curr_file_sens = []
            with open(cur_dir, 'r', encoding='utf-8') as fcsv:
                csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
                for data_line in csv_reader:
                    curr_file_sens.append(data_line)
            selected_sens_index = [random.randint(0, len(curr_file_sens) - 1) for i in range(shot_num)]
            for sen_index in selected_sens_index:
                label_sens.append([curr_file_name, curr_file_sens[sen_index]])

            for oth_sen_idx in range(len(curr_file_sens)):
                if oth_sen_idx not in selected_sens_index:
                    other_sentences.append([curr_file_name, curr_file_sens[oth_sen_idx]])
        # print(len(label_sens), label_sens)
        # print(len(other_sentences), other_sentences)
        # 从选中的5类中，除去已经选择的25个句子外，随机选一个与这25个句子组合为25对，选出2500对作为训练数据
        selected_sen_idx = [random.randint(0, len(other_sentences) - 1) for i in range(1)]
        for pair_num_index in range(len(label_sens)):
            pair1 = label_sens[pair_num_index]
            pair2 = other_sentences[selected_sen_idx[0]]
            if pair1[0] == pair2[0]:
                # print(pair2[0], pair2[1], pair2[1][0])
                with open('test_siamese_pairs.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ')
                    spamwriter.writerow([pair1[1][0] + ',' + pair2[1][0] + ',' + '1'])
            else:
                with open('test_siamese_pairs.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=' ')
                    spamwriter.writerow([pair1[1][0] + ',' + pair2[1][0] + ',' + '0'])


def read_data(is_train_file, is_shuffle=False ):
    if is_train_file:
        csv_file_name = 'siamese_pairs.csv'
    else:
        csv_file_name = 'test_siamese_pairs.csv'
    df1 = pd.read_csv(csv_file_name,header=None,sep=' ',error_bad_lines=False)
    res = np.array(df1)
    # print(res.shape)
    # print(res[0])
    if is_shuffle:
        indices = np.arange(res.shape[0])  # shuffle
        np.random.shuffle(indices)
        res = res[indices]
        print(len(res),type(res), res)
    if is_train_file:
        np.save("train_data.npy", res)
    else:
        np.save('test_data.npy', res)
    return res


def peredata(content_data):
    data = content_data.tolist()
    train_label = []# 用于存储每对句子的是否相似的标签信息1，0
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    sentences1_seg_list = [[] for index in range(len(data))]#存储第一个句子和第二个句子的分词结果
    sentences2_seg_list = [[] for index in range(len(data))]
    is_get_data_info = False #flag set
    if is_get_data_info:
        get_data_info(data)

    qinghuaSeg = pkuseg.pkuseg()
    for i in range(len(data)):
        sen1 = data[i][0].split(',')[0]
        sen2 = data[i][0].split(',')[1]
        train_label.append( int(data[i][0].split(',')[2]) )
        seg_content_data1 = qinghuaSeg.cut(sen1)
        seg_content_data2 = qinghuaSeg.cut(sen2)
        #将分词结果保存到对应数组
        sentences1_seg_list[i] = seg_content_data1
        sentences2_seg_list[i] = seg_content_data2
        #将出现的词编码，并将词与id的对应关系保存在word_index_dic
        for word1 in seg_content_data1:
            if word1 not in word_index_dic:
                word_index_dic[word1] = len(word_index_dic) + 1
        for word2 in seg_content_data2:
            if word2 not in word_index_dic:
                word_index_dic[word2] = len(word_index_dic) + 1
        if i % 1000 == 0:
            print('已经完成进度', i/len(data))
    if not os.path.exists('word_index_dict.json'):
        with open('word_index_dict.json', 'w', encoding='utf-8') as f:
            json.dump(word_index_dic, f, ensure_ascii=False)
    # save label info
    if not os.path.exists("train_label.npy"):
        np.save("train_label.npy", np.array(train_label))
    print('分词结束得到%d个词，分词结果及编码保存在word_bert_dict.json'%len(word_index_dic))#6872
    print(len(train_label))
    print(train_label[0])

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
        np.save("train_data1.npy", sentences1_data)
    if not os.path.exists("sentence_data2.npy"):
        np.save("train_data2.npy", sentences2_data)
    return sentences1_data, sentences2_data, train_label


def get_data_info(data):
    '''
    用于统计待处理的句子对的句子的信息，包括句子的最大分词后的词的个数，所有句子分词后词的平均个数
    :param data:
    :return:统计得到句子分词后的最大词的个数： 313 平均词的个数： 5.6437375
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
        if i %1000 ==0:
            print('已经完成进度',i/len(data))
    print('统计得到句子分词后的最大词的个数：', max_sentences_words_num, '平均词的个数：', sentences_words_count/400000)


def pere_test_data(content_data):
    data = content_data.tolist()
    test_label = []  # 用于存储每对句子的是否相似的标签信息1，0
    word_index_dic = {}  # 存储全部文档中出现的词及对应的编码
    sentences1_seg_list = [[] for index in range(len(data))]  # 存储第一个句子和第二个句子的分词结果
    sentences2_seg_list = [[] for index in range(len(data))]

    is_get_data_info = False  # flag SET
    if is_get_data_info:
        get_data_info(data)
    # 对于test data，采用train data时候得到的word_index_dict信息
    with open('word_index_dict.json', 'r', encoding='utf-8') as f:
        word_index_dic = json.load(f)
    qinghuaSeg = pkuseg.pkuseg()
    for i in range(len(data)):
        sen1 = data[i][0].split(',')[0]
        sen2 = data[i][0].split(',')[1]
        test_label.append(int(data[i][0].split(',')[2]))
        seg_content_data1 = qinghuaSeg.cut(sen1)
        seg_content_data2 = qinghuaSeg.cut(sen2)
        # 将分词结果保存到对应数组
        sentences1_seg_list[i] = seg_content_data1
        sentences2_seg_list[i] = seg_content_data2
        if i % 1000 == 0:
            print('已经完成进度', i / len(data))

    # save label info
    if not os.path.exists("test_label.npy"):
        np.save("test_label.npy", np.array(train_label))
    print(len(test_label))
    print(test_label[0])

    sentences1_data = []
    sentences2_data = []
    for ssl1 in range(len(sentences1_seg_list)):
        word_vec_list1 = []
        for word in sentences1_seg_list[ssl1]:
            if word in word_index_dic:
                word_vec_list1.append(word_index_dic[word])
            else:
                word_vec_list1.append(0)
        sentences1_data.append(word_vec_list1)
    for ssl2 in range(len(sentences2_seg_list)):
        word_vec_list2 = []
        for word in sentences1_seg_list[ssl2]:
            if word in word_index_dic:
                word_vec_list2.append(word_index_dic[word])
            else:
                word_vec_list2.append(0)
        sentences2_data.append(word_vec_list2)
    if not os.path.exists("test_data1.npy"):
        np.save("test_data1.npy", sentences1_data)
    if not os.path.exists("test_data2.npy"):
        np.save("test_data2.npy", sentences2_data)
    return sentences1_data, sentences2_data, test_label


def build_model(sentences1_data, sentences2_data, train_label, test_data1, test_data2, test_label):
    epochs_num = 1000
    embed_size = 768  # 词向量维度
    max_len = 100 # 每句话的最大长度100，平均句子长度6
    max_words = 6872+1   # 统计得到该文档用到的词的个数7282
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
        for key in index_to_bert_vector:
            index = int(key)
            if index == max_words:
                embedding_matrix[0] = np.array(index_to_bert_vector.get(key))
            else:
                embedding_matrix[index] =np.array(index_to_bert_vector.get(key))

    input1 = tf.keras.layers.Input(shape=(100))
    embed1 = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input1)
    input2 = tf.keras.layers.Input(shape=(100))
    embed2 = tf.keras.layers.Embedding(max_words, embed_size, weights=[embedding_matrix], trainable=False)(input2)
    print(input1.shape, input2.shape, embed1.shape, embed2.shape)
    lstm_out1 = tf.keras.layers.LSTM(64, return_sequences=True)(embed1)
    lstm_out2 = tf.keras.layers.LSTM(64, return_sequences=True)(embed2)
    max_pooling1 = tf.keras.layers.GlobalMaxPooling1D()(lstm_out1)
    max_pooling2 = tf.keras.layers.GlobalMaxPooling1D()(lstm_out2)
    merge_vector = tf.keras.layers.concatenate([max_pooling1, max_pooling2], axis=-1)
    mlp_out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(merge_vector)
    model = tf.keras.Model([input1, input2], mlp_out)

    # model.save('m1.h5')
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_test_split = int(len(sentences1_data) * 0.8)
    partial_x_train = [sentences1_data[:train_test_split], sentences2_data[:train_test_split]]
    partial_y_train = train_label[:train_test_split]

    history = model.fit(partial_x_train, partial_y_train,
                        epochs=epochs_num,
                        validation_split=0.2
                        )

    results = model.evaluate([test_data1, test_data2], test_label)
    print('step5: 评估模型效果(损失-精度）：...', results)

    # print('step6: predict test data for count...')
    # predictions = model.predict(test_data, batch_size=batch_size_num)
    # predict = np.argmax(predictions, axis=1)
    # print(predict)
    # with open('02bert_lstm_mlp_predict.csv', 'w', newline='', encoding='utf-8') as csvwriter:
    #     spamwriter = csv.writer(csvwriter, delimiter=' ')
    #     for pre_val in predict:
    #         spamwriter.writerow([pre_val])


if __name__ == '__main__':
    print('采用meta learning模型对标注文本做分类...')
    print('step1：create train and test data...')
    if os.path.exists('siamese_pairs.csv'):
        print('train data have create ok')
    else:
        create_data()
    if os.path.exists('test_siamese_pairs.csv'):
        print('test data have create ok')
    else:
        create_test_data()

    print('step2：read train and test data...')
    if os.path.exists('train_data.npy'):
        train_data_list = np.load("train_data.npy",allow_pickle=True)
    else:
        is_train_file,is_shuffle = True, False
        train_data_list = read_data(is_train_file, is_shuffle)

    if os.path.exists('test_data.npy'):
        test_data_list = np.load("test_data.npy",allow_pickle=True)
    else:
        is_train_file, is_shuffle = False, False
        test_data_list = read_data(is_train_file, is_shuffle)

    print('step3：pere train and test data...')
    if os.path.exists('train_data1.npy') and os.path.exists('train_data2.npy') and os.path.exists(
            'train_label.npy'):
        train_data1 = np.load("train_data1.npy", allow_pickle=True)
        train_data2 = np.load("train_data2.npy", allow_pickle=True)
        train_label = np.load("train_label.npy", allow_pickle=True)
    else:
        train_data1, train_data2, train_label = peredata(train_data_list)

    if os.path.exists('test_data1.npy') and os.path.exists('test_data2.npy') and os.path.exists(
            'test_label.npy'):
        test_data1 = np.load("test_data1.npy", allow_pickle=True)
        test_data2 = np.load("test_data2.npy", allow_pickle=True)
        test_label = np.load("test_label.npy", allow_pickle=True)
    else:
        test_data1, test_data2, test_label = pere_test_data(test_data_list)

    print('step 4: start build model...')
    build_model(train_data1, train_data2, train_label, test_data1, test_data2, test_label)
