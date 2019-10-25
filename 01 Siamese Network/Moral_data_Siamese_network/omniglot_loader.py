import csv
import os
import random
import numpy as np
import math
from PIL import Image
from bert_serving.client import BertClient
# from image_augmentor import ImageAugmentor


class OmniglotLoader:
    """Class that loads and prepares the ann sentences dataset

    This Class was constructed to read the ann sentences, separate the
    training, validation and evaluation test. It also provides function for
    geting one-shot task batches.

    Attributes:
        dataset_path: path of Omniglot Dataset
        train_dictionary: dictionary of the files of the train set (background set). 
            This dictionary is used to load the batch for training and validation.
        evaluation_dictionary: dictionary of the evaluation set.
        batch_size: size of the batch to be used in training
        use_augmentation: boolean that allows us to select if data augmentation is 
            used or not
        image_augmentor: instance of class ImageAugmentor that augments the images
            with the affine transformations referred in the paper

    """

    def __init__(self, dataset_path, use_augmentation, batch_size):
        """Inits OmniglotLoader with the provided values for the attributes.
        Arguments:
            dataset_path: path of Omniglot dataset
            use_augmentation: boolean that allows us to select if data augmentation 
                is used or not       
            batch_size: size of the batch to be used in training     
        """

        self.dataset_path = dataset_path
        self.train_dictionary = {}
        self.evaluation_dictionary = {}

        self.bert_encode_length = 768
        self.batch_size = batch_size
        self.use_augmentation = use_augmentation
        self._train_alphabets = []
        self._validation_alphabets = []
        self._evaluation_alphabets = []
        # self._current_train_alphabet_index = 0
        self._current_validation_alphabet_index = 0
        self._current_evaluation_alphabet_index = 0

        self.load_dataset()
        # if (self.use_augmentation):
        #     self.image_augmentor = self.createAugmentor()
        # else:
        #     self.use_augmentation = []

    def load_dataset(self):
        """Loads the ann sentences into dictionaries
        Loads the ann sentences dataset and stores the available images for each
        alphabet for each of the train and evaluation set.
        """

        train_path = os.path.join(self.dataset_path, 'sentences_background')
        validation_path = os.path.join(self.dataset_path, 'sentences_evaluation')

        # First let's take care of the train sentences存储句子
        for alphabet in os.listdir(train_path):
            alphabet_path = os.path.join(train_path, alphabet)
            current_label_sentences = []
            with open(alphabet_path, 'r', encoding='utf-8') as fcsv:
                csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
                for data_line in csv_reader:
                    current_label_sentences.append(data_line)
            self.train_dictionary[alphabet] = current_label_sentences

        # Now it's time for the validation alphabets
        for alphabet in os.listdir(validation_path):
            alphabet_path = os.path.join(validation_path, alphabet)
            current_label_sentences = []
            with open(alphabet_path, 'r', encoding='utf-8') as fcsv:
                csv_reader = csv.reader(fcsv)  # 使用csv.reader读取csvfile中的文件
                for data_line in csv_reader:
                    current_label_sentences.append(data_line)
            self.evaluation_dictionary[alphabet] = current_label_sentences


    def split_train_datasets(self):
        """ Splits the train set in train and validation
        Divide the 30 train alphabets in train and validation with
        # a 80% - 20% split (24 vs 6 alphabets)
        """
        available_alphabets = list(self.train_dictionary.keys())
        number_of_alphabets = len(available_alphabets)
        # 从用于训练的30个文件中随机选24个index作为train indexes
        train_indexes = random.sample(
            range(0, number_of_alphabets - 1), int(0.8 * number_of_alphabets))
        # print(train_indexes)#[18, 15, 6, 14, 9, 1, 19, 24, 2, 26, 16, 11, 22, 7, 4, 12, 28, 23, 0, 20, 21, 8, 25, 3]
        # If we sort the indexes in reverse order we can pop them from the list
        # and don't care because the indexes do not change 对train的index逆序排序
        train_indexes.sort(reverse=True)
        # print(train_indexes)#[28, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 15, 14, 12, 11, 9, 8, 7, 6, 4, 3, 2, 1, 0]
        for index in train_indexes:
            self._train_alphabets.append(available_alphabets[index])
            available_alphabets.pop(index)

        # The remaining alphabets are saved for validation
        self._validation_alphabets = available_alphabets

        # The evaluation_alphabets are from sentences evaluation file for evaluation
        self._evaluation_alphabets = list(self.evaluation_dictionary.keys())


    def _convert_batch_sentences_to_sentences_and_labels(self, sentences_list, is_one_shot_task):
        """ Loads the sentence and its correspondent labels
        并将其通过bert +lstm转换为对应的向量

        Take the list with the path from the current batch, read the images and
        return the pairs of images and the labels
        If the batch is from train or validation the labels are alternately 1's and
        0's. If it is a evaluation set only the first pair has label 1

        Arguments:
            sentences_list: list of sentences to be loaded in this batch
            is_one_shot_task: flag sinalizing if the batch is for one-shot task or if
                it is for training

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """
        # print(sentences_list)
        number_of_pairs = int(len(sentences_list) / 2)
        pairs_of_sentences = [np.zeros(
            (number_of_pairs, self.bert_encode_length)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))
        bc = BertClient()
        for pair in range(number_of_pairs):
            sen = sentences_list[pair*2]
            sen_vec = bc.encode([sen])
            pairs_of_sentences[0][pair,:] = sen_vec[0]

            sen1 = sentences_list[pair * 2 + 1]
            sen_vec1 = bc.encode([sen1])
            pairs_of_sentences[1][pair,:] = sen_vec1[0]

            if not is_one_shot_task:
                if (pair + 1) % 2 == 0:
                    labels[pair] = 0
                else:
                    labels[pair] = 1
            else:
                if pair == 0:
                    labels[pair] = 1
                else:
                    labels[pair] = 0
        # shuffle
        if not is_one_shot_task:
            '''
            # indices = np.arange(len(labels))
            # np.random.shuffle(indices)
            # pairs_of_sentences = pairs_of_sentences[indices]
            # labels = labels[indices]
            '''
            random_permutation = np.random.permutation(number_of_pairs)
            labels = labels[random_permutation]
            pairs_of_sentences[0][:, :] = pairs_of_sentences[0][random_permutation, :]
            pairs_of_sentences[1][:, :] = pairs_of_sentences[1][random_permutation, :]

        return pairs_of_sentences, labels


    def get_train_batch(self):
        """ Loads and returns a batch of train sentences
        从训练集获取batch对，
        Get a batch of pairs from the training set. Each batch will contain
        images from a single alphabet. I decided to select one single example
        from random n/2 characters in each alphabet. If the current alphabet
        has lower number of characters than n/2 (some of them have 14) we
        sample repeated classed for that batch per character in the alphabet
        to pair with a different categories. In the other half of the batch
        I selected pairs of same characters. In resume we will have a batch
        size of n, with n/2 pairs of different classes and n/2 pairs of the same
        class. Each batch will only contains samples from one single alphabet.

        Returns:
            pairs_of_sentences: pairs of images for the current batch
            labels: correspondent labels 1 for same class, 0 for different classes

        """
        # print('**')
        # print(self._train_alphabets) #所以训练的label
        # print(self.train_dictionary)  #所以label下对应的句子
        available_labels = self._train_alphabets
        number_of_labels = len(self._train_alphabets)

        selected_labels_indexes = [random.randint(0, number_of_labels - 1) for i in range(self.batch_size)]
        bacth_sentences = []

        # 从训练集的24个label中随机选一个，从这个文件中随机选三个句子，
        # 从剩下的label中随机选一个，随机选一个句子共4条句子

        for index in selected_labels_indexes:
            current_label = available_labels[index]
            available_sentences = (self.train_dictionary[current_label])
            number_of_sen = len(available_sentences)

            # Random select a 3 indexes of images from the same label
            select_sen_indexes = random.sample(range(0, number_of_sen-1), 3)
            sen_1 = available_sentences[select_sen_indexes[0]][0]
            sen_2 = available_sentences[select_sen_indexes[1]][0]
            sen_3 = available_sentences[select_sen_indexes[2]][0]

            bacth_sentences.append(sen_1)
            bacth_sentences.append(sen_2)

            # Now let's take care of the pair of sentence from different label
            bacth_sentences.append(sen_3)
            # 从其他没被选择的label中随机选一个label 在其中随机选一个句子
            different_labels = available_labels[:]
            different_labels.pop(index)

            different_label_index = random.sample(range(0, len(different_labels) - 1), 1)
            different_label_sen = self.train_dictionary[available_labels[different_label_index[0]]]
            select_diff_indexes = random.sample(range(0, len(different_label_sen) - 1), 1)
            sen_diff = different_label_sen[select_diff_indexes[0]][0]
            bacth_sentences.append(sen_diff)

        # print(len(bacth_sentences))
        # 将句子转换为句子对及对应的label
        sentences, labels = self._convert_batch_sentences_to_sentences_and_labels(
            bacth_sentences, is_one_shot_task=False)

        return sentences, labels

    def get_one_shot_batch(self, support_set_size, is_validation):
        """ Loads and returns a batch for one-shot task images

        Gets a one-shot batch for evaluation or validation set, it consists in a
        single image that will be compared with a support set of images. It returns
        the pair of images to be compared by the model and it's labels (the first
        pair is always 1) and the remaining ones are 0's

        Returns:
            pairs_of_images: pairs of images for the current batch
            labels: correspondent labels -1 for same class, 0 for different classes

        """

        # Set some variables that will be different for validation and evaluation sets
        if is_validation:
            alphabets = self._validation_alphabets
            dictionary = self.train_dictionary
        else:
            alphabets = self._evaluation_alphabets
            dictionary = self.evaluation_dictionary

        current_label_index = random.sample(range(0, len(alphabets)-1),1)
        available_sentences = dictionary[alphabets[current_label_index[0]]]
        number_of_sentences = len(available_sentences)

        bacth_sentences = []
        # Get test sentences
        sentence_indexes = random.sample(range(0, number_of_sentences), 2)
        test_sentence = available_sentences[sentence_indexes[0]][0]
        bacth_sentences.append(test_sentence)
        sent1 = available_sentences[sentence_indexes[1]][0]
        bacth_sentences.append(sent1)

        # Let's get our test image and a pair corresponding to
        if support_set_size == -1:
            number_of_support_labels = support_set_size
        else:
            number_of_support_labels = support_set_size

        different_labels = []
        for ss in alphabets:
            if ss == alphabets[current_label_index[0]]:
                pass
            else:
                different_labels.append(ss)

        support_labels_indexes = []
        for ran_index in range(number_of_support_labels - 1):
            support_labels_indexes.append(random.randint(0, len(different_labels) - 1))

        for index in support_labels_indexes:
            support_labels = different_labels[index]
            diff_sents = dictionary[support_labels]
            diff_sen_index = random.sample(range(0,len(diff_sents)), 1)
            sent2 = diff_sents[diff_sen_index[0]][0]
            bacth_sentences.append(test_sentence)
            bacth_sentences.append(sent2)

        sentences, labels = self._convert_batch_sentences_to_sentences_and_labels(
            bacth_sentences, is_one_shot_task=True)
        return sentences, labels

    def one_shot_test(self, model, support_set_size, number_of_tasks_per_alphabet,
                      is_validation):
        """ Prepare one-shot task and evaluate its performance

        Make one shot task in validation and evaluation sets
        if support_set_size = -1 we perform a N-Way one-shot task with
        N being the total of characters in the alphabet

        Returns:
            mean_accuracy: mean accuracy for the one-shot task
        """

        # Set some variables that depend on dataset
        if is_validation:# 在验证集做one shot task
            print('\nMaking One Shot Task on validation alphabets:')
        else:# 在测试集做one shot task，用于最终的测试
            print('\nMaking One Shot Task on evaluation alphabets:')
        mean_alphabet_accuracy = 0
        for num_index in range(number_of_tasks_per_alphabet):
            images, _ = self.get_one_shot_batch(support_set_size, is_validation=is_validation)
            probabilities = model.predict_on_batch(images)

            # Added this condition because noticed that sometimes the outputs
            # of the classifier was almost the same in all images, meaning that
            # the argmax would be always by defenition 0.
            if np.argmax(probabilities) == 0 and probabilities.std()>0.01:
                accuracy = 1.0
            else:
                accuracy = 0.0

            mean_alphabet_accuracy += accuracy
        mean_alphabet_accuracy /= number_of_tasks_per_alphabet
        print(' 本次validation的平均' + ', accuracy: ' + str(mean_alphabet_accuracy))

        return mean_alphabet_accuracy
