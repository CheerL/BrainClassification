import collections
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from config import (BATCH_SIZE, CLASS_NUM, REPEAT_NUM, MOD_NUM, LOG_PATH, MIN_CONNECT_TUMOR_NUM,
                    MODEL_PATH, SIZE, SUMMARY_INTERVAL, VAL_INTERVAL, MIN_TUMOR_NUM, WHOLE_REPEAT_NUM)
from utils.logger import Logger
from utils.tfrecord import generate_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['TF_SYNC_ON_FINISH'] = '0'
# os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


class Net(object):
    def __init__(self, class_num=CLASS_NUM):
        net_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=1,
            gpu_options=tf.GPUOptions(
                allow_growth=True
                # force_gpu_compatible=True
            )
        )
        # net_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.__loaded = False
        self.logger = Logger('net')
        self.class_num = class_num
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=net_config)
        with self.graph.as_default(), tf.variable_scope('net_base'):
            self.img = tf.placeholder(
                tf.float32, [None, SIZE, SIZE, MOD_NUM], name='img')
            self.label = tf.placeholder(
                tf.float32, [None, CLASS_NUM], name='label')
            # self.inputs = tf.reshape(self.img, [-1, SIZE, SIZE, MOD_NUM])
            self.inputs = tf.identity(self.img, 'inputs')
            self.tower_loss = []
            self.tower_grads = []
            self.tower_preds = []
            self.tower_accuracy = []
            self.prediction = None
            self.accuracy = None
            self.loss = None
            self.trainer = None
            self.summary = None
            self.writer = None
            self.saver = None
            self.train_step = tf.train.get_or_create_global_step(self.graph)
            with tf.variable_scope('validation_net'):
                self.val_step = tf.Variable(0, dtype=tf.int32, name='v_step')
                self.val_data = {
                    'avg_acc': tf.Variable(0.0, dtype=tf.float32, name='avg_acc'),
                    'tp': tf.Variable(0.0, dtype=tf.float32, name='tp'),
                    'tn': tf.Variable(0.0, dtype=tf.float32, name='tn'),
                    'fp': tf.Variable(0.0, dtype=tf.float32, name='fp'),
                    'fn': tf.Variable(0.0, dtype=tf.float32, name='fn')
                }

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
              val_batch_size=BATCH_SIZE,
              repeat_time=REPEAT_NUM,
              val_interval=VAL_INTERVAL,
              summary_interval=SUMMARY_INTERVAL):
        with self.graph.as_default():
            self.start()
            self.logger.info('Train start')
            _, next_batch = generate_dataset(
                file_list, batch_size, repeat_time=repeat_time)
            while True:
                try:
                    img, label = self.sess.run(next_batch)
                    _, train_step, summary, accuracy, loss = self.sess.run(
                        fetches=[
                            self.trainer,
                            self.train_step,
                            self.summary,
                            self.accuracy,
                            self.loss
                        ],
                        feed_dict={
                            self.img: img,
                            self.label: label
                        })

                    if train_step % val_interval == 0:
                        self.validate(val_file_list, val_batch_size)
                    elif train_step % summary_interval == 0:
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

    def whole_predict(self, tfr_name, batch_size=BATCH_SIZE, with_label=True, final_result=False,
                      min_connect_tumor_num=MIN_CONNECT_TUMOR_NUM,
                      min_tumor_num=MIN_TUMOR_NUM, random=True):
        with self.graph.as_default():
            _, next_batch = generate_dataset([tfr_name], None, train=False, shuffle=False, batch=False)
            tfr_predict = []
            tfr_imgs, tfr_labels = self.sess.run(next_batch)
            tfr_size = len(tfr_labels)
            if random:
                rank = np.arange(tfr_size)
                for _ in range(WHOLE_REPEAT_NUM):
                    rank_copy = rank.copy()
                    np.random.shuffle(rank_copy)
                    predict = []
                    for i in range(0, tfr_size, batch_size):
                        imgs = tfr_imgs[rank_copy][i:i+batch_size]
                        if len(imgs) < batch_size:
                            imgs = np.concatenate(
                                [imgs] * np.ceil(batch_size / len(imgs)).astype(int)
                            )[:batch_size]
                        predict.append(self.predict(imgs))
                    predict = np.concatenate(predict)
                    pair = list(zip(predict, rank_copy))
                    pair.sort(key=lambda x: x[1])
                    tfr_predict.append(np.stack(np.array(pair).T[0]))
                tfr_predict = np.array(tfr_predict).mean(axis=0).argmax(axis=1)
            else:
                for i in range(0, tfr_size, batch_size):
                    imgs = tfr_imgs[i:i+batch_size]
                    if len(imgs) < batch_size:
                        imgs = np.concatenate(
                            [imgs] * np.ceil(batch_size / len(imgs)).astype(int)
                        )[:batch_size]
                    tfr_predict.append(self.predict(imgs))
                tfr_predict = np.concatenate(tfr_predict)

            tfr_zero_predict_pos = np.where(tfr_predict == 0)[0]
            for pos in tfr_zero_predict_pos:
                if 0 < pos < tfr_predict.size - 1:
                    if tfr_predict[pos - 1] == tfr_predict[pos + 1] == 1:
                        tfr_predict[pos] = 1

            tfr_one_predict_pos = np.where(tfr_predict == 1)[0]
            for pos in tfr_one_predict_pos:
                if 0 < pos < tfr_predict.size - 1:
                    if tfr_predict[pos - 1] == tfr_predict[pos + 1] == 0:
                        tfr_predict[pos] = 0

            if final_result:
                result = 0
                tfr_tumor_predict_pos = np.where(tfr_predict == 1)[0]
                if tfr_tumor_predict_pos.size >= min_tumor_num:
                    if min_connect_tumor_num in np.convolve(tfr_predict, np.ones(min_connect_tumor_num), mode='same'):
                        result = 1
                tfr_predict = result

            if with_label:
                return tfr_predict, tfr_labels.argmax(axis=1)
            return tfr_predict

    def validate_report(self, predict_list, label_list, test, tumor_as_1=True):
        tumor_num, normal_num = int(tumor_as_1), int(not tumor_as_1)
        tumor_list = predict_list[label_list == tumor_num]        # tumor result
        normal_list = predict_list[label_list == normal_num]      # normal result
        if tumor_list.size and normal_list.size:
            tp_rate = (tumor_list == tumor_num).sum() / len(tumor_list)     # t -> f
            fn_rate = 1 - tp_rate                                           # t -> t
            tn_rate = (normal_list == normal_num).sum() / len(normal_list)        # f -> f
            fp_rate = 1 - tn_rate                                           # f -> t
        elif not tumor_list.size:
            tn_rate = (normal_list == normal_num).sum() / len(normal_list)        # f -> f
            fp_rate = 1 - tn_rate                                           # f -> t
            fn_rate = 0.0
            tp_rate = 0.0
        elif not normal_list.size:
            tp_rate = (tumor_list == tumor_num).sum() / len(tumor_list)     # t -> f
            fn_rate = 1 - tp_rate                                           # t -> t
            fp_rate = 0.0
            tn_rate = 0.0
        avg_accuracy = np.equal(predict_list, label_list).mean()
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
            self.writer.add_summary(
                self.sess.run(self.val_summary), train_step)

    def validate(self, file_list, batch_size=BATCH_SIZE, test=False):
        with self.graph.as_default():
            _, next_batch = generate_dataset(
                file_list, batch_size, train=False, shuffle=True, batch=True)
            val_step_initer = tf.variables_initializer([self.val_step])
            self.sess.run(val_step_initer)
            label_list = []
            predict_list = []
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
                    predict_list.append(predict.argmax(axis=1))
                    label_list.append(label.argmax(axis=1))

                except tf.errors.OutOfRangeError:
                    break

            predict_list = np.concatenate(predict_list)
            label_list = np.concatenate(label_list)
            self.validate_report(predict_list, label_list,
                                 test, tumor_as_1=True)
        return predict_list, label_list

    def whole_validate(self, file_list, batch_size=BATCH_SIZE, test=False,
                       min_connect_tumor_num=MIN_CONNECT_TUMOR_NUM,
                       min_tumor_num=MIN_TUMOR_NUM):
        with self.graph.as_default():
            file_num = len(file_list)
            label_list = np.zeros((file_num), dtype=np.int64)
            predict_list = np.zeros((file_num), dtype=np.int64)
            for num, tfr_name in enumerate(file_list):
                tfr_predict, tfr_labels = self.whole_predict(
                    tfr_name, batch_size, with_label=True, final_result=True,
                    min_connect_tumor_num=min_connect_tumor_num,
                    min_tumor_num=min_tumor_num)
                predict_list[num] = tfr_predict
                if 1 in tfr_labels:
                    label_list[num] = 1

            self.validate_report(predict_list, label_list, test, tumor_as_1=True)
        return predict_list, label_list

    def save(self, model_name):
        model_path = os.path.join(MODEL_PATH, model_name)
        self.saver.save(self.sess, model_path)

    def load(self, model_path):
        self.saver.restore(self.sess, model_path)
        self.__loaded = True


if __name__ == '__main__':
    pass
