import collections
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from config import (BATCH_SIZE, CLASS_NUM, EPOCH_REPEAT_NUM, MOD_NUM, LOG_PATH,
                    MODEL_PATH, SIZE, SUMMARY_INTERVAL, VER_BATCH_SIZE)
from utils.logger import Logger
from utils.tfrecord import generate_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'is_first', 'args'])):
    'A named tuple describing a ResNet block'


class Net(object):
    def __init__(self, class_num=CLASS_NUM):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.allow_growth = True

        self.__loaded = False
        self.logger = Logger('net')
        self.class_num = class_num
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.img = tf.placeholder(
                tf.float32, [None, SIZE, SIZE, MOD_NUM], name='img')
            self.label = tf.placeholder(
                tf.float32, [None, CLASS_NUM], name='label')
            # self.inputs = tf.reshape(self.img, [-1, SIZE, SIZE, MOD_NUM])
            self.inputs = tf.identity(self.img, 'inputs')
            self.prediction = None
            self.accuracy = None
            self.loss = None
            self.trainer = None
            self.summary = None
            self.writer = None
            self.saver = None

            self.train_step = tf.train.get_or_create_global_step(self.graph)

    def start(self):
        if not self.__loaded:
            self.sess.run(tf.global_variables_initializer())

    def train(self, file_list, batch_size=BATCH_SIZE,
              epoch_repeat_num=EPOCH_REPEAT_NUM,
              summary_interval=SUMMARY_INTERVAL):
        with self.graph.as_default():
            self.start()
            iterator, next_batch = generate_dataset(file_list, batch_size)
            for epoch in range(epoch_repeat_num):
                self.sess.run(iterator.initializer)
                self.logger.info('Start train epoch %d/%d' % (
                    epoch + 1, epoch_repeat_num))
                while True:
                    try:
                        img, label = self.sess.run(next_batch)
                        _, train_step = self.sess.run(
                            [self.trainer, self.train_step],
                            feed_dict={
                                self.img: img,
                                self.label: label
                            })

                        if train_step % summary_interval == 0:
                            summary, accuracy, loss = self.sess.run(
                                [self.summary, self.accuracy, self.loss],
                                feed_dict={
                                    self.img: img,
                                    self.label: label
                                })
                            self.writer.add_summary(summary, train_step)
                            self.logger.info('Save summary %d, accuracy: %f, loss %f' % (
                                train_step, accuracy, loss))
                    except tf.errors.OutOfRangeError:
                        break
            self.logger.info('Train end')

    def predict(self, img):
        return self.sess.run(
            self.prediction,
            feed_dict={
                self.img: img
            })

    def whole_predict(self, tfr_name, batch_size=BATCH_SIZE):
        predict_list, label_list = self.verify([tfr_name], batch_size)

    def verify(self, file_list, batch_size=VER_BATCH_SIZE):
        with self.graph.as_default():
            self.start()
            iterator, next_batch = generate_dataset(
                file_list, batch_size, True)
            self.sess.run(iterator.initializer)
            label_list = np.array([], dtype=np.float32)
            predict_list = np.array([], dtype=np.float32)
            acc_list = list()
            while True:
                try:
                    img, label = self.sess.run(next_batch)
                    accuracy, loss, predict, step = self.sess.run(
                        [
                            self.accuracy,
                            self.loss,
                            self.prediction,
                            self.train_step
                        ],
                        feed_dict={
                            self.img: img,
                            self.label: label
                        })
                    self.logger.info(
                        'accuracy: %f, loss: %f' % (accuracy, loss))
                    predict_list = np.concatenate(
                        [predict_list, predict.argmax(axis=1)])
                    label_list = np.concatenate(
                        [label_list, label.argmax(axis=1)])
                    acc_list.append(accuracy)
                except tf.errors.OutOfRangeError:
                    break
            predict_list[predict_list >= 0.5] = 1
            predict_list[predict_list < 0.5] = 0
            true_list = predict_list[label_list == 0]
            false_list = predict_list[label_list == 1]
            tn_rate = true_list.sum() / len(true_list)
            tp_rate = 1 - tn_rate
            fn_rate = false_list.sum() / len(false_list)
            fp_rate = 1 - fn_rate
            avg_accuracy = sum(acc_list) / len(acc_list)
            result = '%d: Average accuary %f, TP %f, TN %f, FP %f, FN %f' % (
                step, avg_accuracy, tp_rate, tn_rate, fp_rate, fn_rate)
            self.logger.info('Verify end')
            self.logger.info(result)

            with open(os.path.join(LOG_PATH, 'result'), 'a+') as file:
                file.write(result)

            return predict_list, label_list

    def save(self, model_name):
        model_path = os.path.join(MODEL_PATH, model_name)
        self.saver.save(self.sess, model_path)

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)
        self.__loaded = True

if __name__ == '__main__':
    pass
