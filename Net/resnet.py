import collections
import os

import numpy as np
import tensorflow as tf

from config import (BATCH_SIZE, CLASS_NUM, EPOCH_REPEAT_NUM, SUMMARY_PATH, MOD_NUM,
                    MODEL_PATH, SIZE, SUMMARY_INTERVAL, VER_BATCH_SIZE)
from tensorflow.contrib import slim
from utils.logger import Logger
from utils.tfrecord import generate_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 定义Block


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'is_first', 'args'])):
    'A named tuple describing a ResNet block'


class ResNet(object):
    def __init__(self, class_num=CLASS_NUM, res_type=None, struct=None):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1.0
        config.gpu_options.allow_growth = True

        self.__loaded = False
        self.logger = Logger('resnet')
        self.class_num = class_num
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.img = tf.placeholder(
                tf.float32, [None, SIZE, SIZE, MOD_NUM], name='img')
            self.label = tf.placeholder(tf.float32, [None, CLASS_NUM], name='label')
            # self.inputs = tf.reshape(self.img, [-1, SIZE, SIZE, MOD_NUM])
            self.inputs = self.img
            self.prediction = None
            self.accuracy = None
            self.loss = None
            self.trainer = None
            self.summary = None
            self.writer = None
            self.saver = None

            self.train_step = tf.get_variable(
                'train_step', initializer=0, dtype=tf.int32, trainable=False)

            if res_type == 'DIY' and struct:
                self.resnet_v2_struct(self.class_num, struct)
            elif res_type is 50:
                self.resnet_v2_50(self.class_num)
            elif res_type is 101:
                self.resnet_v2_101(self.class_num)
            elif res_type is 152:
                self.resnet_v2_152(self.class_num)
            elif res_type is 200:
                self.resnet_v2_200(self.class_num)
            else:
                self.resnet_v2_50(self.class_num)

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
                self.logger.info('Start train epoch {}/{}'.format(
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
                            self.logger.info('Save summary {}, accuracy: {}, loss {}'.format(
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

    def verify(self, file_list, batch_size=VER_BATCH_SIZE):
        with self.graph.as_default():
            self.start()
            iterator, next_batch = generate_dataset(
                file_list, batch_size, True)
            self.sess.run(iterator.initializer)
            acc_list = list()
            while True:
                try:
                    img, label = self.sess.run(next_batch)
                    accuracy, loss = self.sess.run(
                        [self.accuracy, self.loss],
                        feed_dict={
                            self.img: img,
                            self.label: label
                        })
                    self.logger.info(
                        'accuracy: {}, loss: {}'.format(accuracy, loss))
                    acc_list.append(accuracy)
                except tf.errors.OutOfRangeError:
                    break
            self.logger.info('Verify end')
            self.logger.info('Average accuary %f' % (sum(acc_list) / len(acc_list)))


    def save(self, model_name):
        model_path = os.path.join(MODEL_PATH, model_name)
        self.saver.save(self.sess, model_path)

    def load(self, model_name):
        model_path = os.path.join(MODEL_PATH, model_name)
        self.saver.restore(self.sess, model_path)
        self.__loaded = True

    def subsample(self, inputs, factor, scope=None):
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, 1, stride=factor, scope=scope)

    def resnet_arg_scope(self, is_training=True,
                         weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True):
        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(
                                weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

    @slim.add_arg_scope
    def stack_blocks_dense(self, net, blocks, outputs_collections=None):
        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as sc:
                unit_depth, unit_depth_bottleneck, unit_num = block.args
                for i in range(unit_num):
                    with tf.variable_scope('unit_%d' % i, values=[net]):
                        unit_stride = 2 if i is 0 and not block.is_first else 1
                        net = block.unit_fn(net, depth=unit_depth, stride=unit_stride,
                                            depth_bottleneck=unit_depth_bottleneck)
                net = slim.utils.collect_named_outputs(
                    outputs_collections, sc.name, net)
        return net

    @slim.add_arg_scope
    # 定义核心bottleneck残差学习单元
    def bottleneck(self, inputs, depth, depth_bottleneck, stride,
                   outputs_collections=None, scope=None):
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(
                inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(
                inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = self.subsample(preact, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, 1, stride=stride,
                                       activation_fn=None, normalizer_fn=None, scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck,
                                   1, stride=1, scope='conv1')
            residual = slim.conv2d(residual, depth_bottleneck,
                                   3, stride=stride, padding='SAME', scope='conv2')
            residual = slim.conv2d(residual, depth, 1, stride=1,
                                   activation_fn=None, normalizer_fn=None, scope='conv3')

            output = tf.add(shortcut, residual)

            return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

    # 定义生成ResNet V2的主函数

    def resnet_v2(self, blocks, class_num, reuse=None, scope=None):
        with tf.variable_scope(scope, 'resnet_v2', [self.inputs], reuse=reuse) as sc:
            with slim.arg_scope([slim.conv2d, self.bottleneck, self.stack_blocks_dense]):
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = slim.conv2d(self.inputs, 64, 7, stride=2,
                                      padding='SAME', scope='preconv')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='prepool')
                net = self.stack_blocks_dense(net, blocks)
                net = slim.batch_norm(
                    net, activation_fn=tf.nn.relu, scope='postbn')
                net = tf.reduce_mean(
                    net, [1, 2], name='postpool', keepdims=True)
                net = slim.flatten(net, scope='flatten')
                net = slim.fully_connected(net, class_num, activation_fn=None, normalizer_fn=None, scope='fc')
                self.prediction = slim.softmax(net, scope='prediction')
                correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                # self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.label)
                self.loss = -tf.reduce_mean(self.label * tf.log(self.prediction))

                tf.summary.scalar('accuracy', self.accuracy)
                tf.summary.scalar('loss', self.loss)

                self.summary = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))
                self.writer = tf.summary.FileWriter(SUMMARY_PATH, self.graph)
                self.trainer = tf.train.AdadeltaOptimizer().minimize(self.loss, global_step=self.train_step)
                self.logger.info('Build Net OK')
                self.saver = tf.train.Saver()
                return net

    def resnet_v2_struct(self, class_num, struct=[], reuse=None, scope='resnet_v2'):
        blocks = [
            Block(
                'block_%d' % num,
                self.bottleneck,
                num is 0,
                block_struct
            ) for num, block_struct in enumerate(struct)
        ]
        return self.resnet_v2(blocks, class_num, reuse=reuse, scope=scope)

    def resnet_v2_50(self, class_num, reuse=None, scope='resnet_v2_50'):
        struct = [
            (256, 64, 3),
            (512, 128, 4),
            (1024, 256, 6),
            (2048, 512, 3)
        ]
        return self.resnet_v2_struct(class_num, struct, reuse, scope)

    def resnet_v2_101(self, class_num, reuse=None, scope='resnet_v2_101'):
        struct = [
            (256, 64, 3),
            (512, 128, 4),
            (1024, 256, 23),
            (2048, 512, 3)
        ]
        return self.resnet_v2_struct(class_num, struct, reuse, scope)

    def resnet_v2_152(self, class_num, reuse=None, scope='resnet_v2_152'):
        struct = [
            (256, 64, 3),
            (512, 128, 8),
            (1024, 256, 36),
            (2048, 512, 3)
        ]
        return self.resnet_v2_struct(class_num, struct, reuse, scope)

    def resnet_v2_200(self, class_num, reuse=None, scope='resnet_v2_200'):
        struct = [
            (256, 64, 3),
            (512, 128, 24),
            (1024, 256, 36),
            (2048, 512, 3)
        ]
        return self.resnet_v2_struct(class_num, struct, reuse, scope)


if __name__ == '__main__':
    pass
