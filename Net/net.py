import collections
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from config import (BATCH_SIZE, CLASS_NUM, REPEAT_NUM, MOD_NUM, LOG_PATH,
                    MODEL_PATH, SIZE, SUMMARY_INTERVAL, VAL_BATCH_SIZE, VAL_INTERVAL)
from utils.logger import Logger
from utils.tfrecord import generate_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'is_first', 'args'])):
    'A named tuple describing a ResNet block'


class Net(object):
    def __init__(self, class_num=CLASS_NUM):
        net_config = tf.ConfigProto()
        net_config.gpu_options.per_process_gpu_memory_fraction = 1.0
        net_config.gpu_options.allow_growth = True
        # net_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.__loaded = False
        self.logger = Logger('net')
        self.class_num = class_num
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=net_config)
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
            self.val_step = tf.Variable(0, dtype=tf.int32, name='v_step')
            self.val_data = {
                'avg_acc': tf.Variable(0.0, dtype=tf.float32, name='avg_acc'),
                'tp': tf.Variable(0.0, dtype=tf.float32, name='tp'),
                'tn': tf.Variable(0.0, dtype=tf.float32, name='tn'),
                'fp': tf.Variable(0.0, dtype=tf.float32, name='fp'),
                'fn': tf.Variable(0.0, dtype=tf.float32, name='fn')
            }
            with tf.variable_scope('validation_net'):
                for name, var in self.val_data.items():
                    tf.summary.scalar(name, var, ['validation_summary'])
                self.val_summary = tf.summary.merge(
                    self.graph.get_collection('validation_summary')
                )

    def start(self):
        if not self.__loaded:
            self.sess.run(tf.global_variables_initializer())
            self.__loaded = True

    def train(self, file_list, val_file_list,
              batch_size=BATCH_SIZE,
              val_batch_size=VAL_BATCH_SIZE,
              repeat_time=REPEAT_NUM,
              val_interval=VAL_INTERVAL,
              summary_interval=SUMMARY_INTERVAL):
        with self.graph.as_default():
            self.start()
            iterator, next_batch = generate_dataset(file_list, batch_size, repeat_time=repeat_time)
            self.sess.run(iterator.initializer)
            while True:
                try:
                    img, label = self.sess.run(next_batch)
                    [_, train_step] = self.sess.run(
                        fetches=[
                            self.trainer,
                            self.train_step
                        ],
                        feed_dict={
                            self.img: img,
                            self.label: label
                        })

                    if train_step % val_interval == 0:
                        self.validate(val_file_list, val_batch_size)
                    elif train_step % summary_interval == 0:
                        summary, accuracy, loss = self.sess.run(
                            fetches=[
                                self.summary,
                                self.accuracy,
                                self.loss
                            ],
                            feed_dict={
                                self.img: img,
                                self.label: label
                            })
                        self.writer.add_summary(summary, train_step)
                        self.logger.info('Training summary %d, accuracy: %f, loss %f' % (
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

    def whole_predict(self, tfr_name, batch_size=VAL_BATCH_SIZE):
        with self.graph.as_default():
            iterator, next_batch = generate_dataset(
                [tfr_name], batch_size, train=False, shuffle=False, batch=False)
            self.sess.run(iterator.initializer)
            predict_list = np.array([], dtype=np.float32)
            while True:
                try:
                    img, _ = self.sess.run(next_batch)
                    predict = self.predict(img)
                    predict_list = np.concatenate(
                        [predict_list, predict.argmax(axis=1)])
                except tf.errors.OutOfRangeError:
                    break
            self.logger.info('Prediction end')
            return predict_list

    def validate(self, file_list, batch_size=VAL_BATCH_SIZE, test=False, shuffle=True, batch=True):
        with self.graph.as_default():
            iterator, next_batch = generate_dataset(
                file_list, batch_size, train=False, shuffle=shuffle, batch=batch)
            val_step_initer = tf.variables_initializer([self.val_step])
            self.sess.run([iterator.initializer, val_step_initer])
            label_list = np.array([], dtype=np.float32)
            predict_list = np.array([], dtype=np.float32)
            acc_list = list()
            while True:
                try:
                    img, label = self.sess.run(next_batch)
                    accuracy, loss, predict, val_step = self.sess.run(
                        fetches=[
                            self.accuracy,
                            self.loss,
                            self.prediction,
                            self.val_step.assign_add(1)
                        ],
                        feed_dict={
                            self.img: img,
                            self.label: label
                        })
                    if val_step % 10 == 0:
                        self.logger.info('Validation summary %d, accuracy: %f, loss: %f' % (
                            val_step, accuracy, loss))
                    predict_list = np.concatenate(
                        [predict_list, predict.argmax(axis=1)])
                    label_list = np.concatenate(
                        [label_list, label.argmax(axis=1)])
                    acc_list.append(accuracy)
                except tf.errors.OutOfRangeError:
                    break
            predict_list[predict_list >= 0.5] = 1
            predict_list[predict_list < 0.5] = 0
            true_list = predict_list[label_list == 0]       # tumor result
            false_list = predict_list[label_list == 1]      # normal result
            if len(true_list) and len(false_list):
                fn_rate = true_list.sum() / len(true_list)      # t -> f
                tp_rate = 1 - fn_rate                           # t -> t
                tn_rate = false_list.sum() / len(false_list)    # f -> f
                fp_rate = 1 - tn_rate                           # f -> t
            elif len(true_list) is 0:
                tn_rate = false_list.sum() / len(false_list)    # f -> f
                fp_rate = 1 - tn_rate                           # f -> t
                fn_rate = 0.0
                tp_rate = 0.0
            elif len(false_list) is 0:
                fn_rate = true_list.sum() / len(true_list)      # t -> f
                tp_rate = 1 - fn_rate                           # t -> t
                fn_rate = 0.0
                tp_rate = 0.0
            avg_accuracy = sum(acc_list) / len(acc_list)
            train_step = self.sess.run(self.train_step)
            result = '(train step %d) Average accuary %f, TP %f, TN %f, FP %f, FN %f' % (
                train_step, avg_accuracy, tp_rate, tn_rate, fp_rate, fn_rate)
            self.logger.info('Validation end')
            self.logger.info(result)

            with open(os.path.join(LOG_PATH, 'result'), 'a+') as file:
                file.write(result + '\n')

            if not test:
                self.sess.run([
                    tf.assign(self.val_data['avg_acc'], avg_accuracy),
                    tf.assign(self.val_data['tp'], tp_rate),
                    tf.assign(self.val_data['tn'], tn_rate),
                    tf.assign(self.val_data['fp'], fp_rate),
                    tf.assign(self.val_data['fn'], fn_rate)
                ])
                self.writer.add_summary(self.sess.run(self.val_summary), train_step)

            return predict_list, label_list

    def save(self, model_name):
        model_path = os.path.join(MODEL_PATH, model_name)
        self.saver.save(self.sess, model_path)

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)
        self.__loaded = True

if __name__ == '__main__':
    pass
