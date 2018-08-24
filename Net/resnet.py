import collections
import itertools
import os

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import device_setter

from config import (BATCH_NORM_DECAY, BATCH_NORM_EPSILON, BATCH_NORM_SCALE,
                    CLASS_NUM, CONV_WEIGHT_DECAY, DEFAULT_VERSION, BATCH_SIZE,
                    LEARNING_RATE, LR_DECAY_RATE, LR_DECAY_STEP, MOMENTUM,
                    NUM_GPU, PS_TYPE, SUMMARY_PATH)
from Net.net import Net


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'is_first', 'args'])):
    'A named tuple describing a ResNet block'


class ResNet(Net):
    def __init__(self, class_num=CLASS_NUM, res_type=None, struct=None):
        super(ResNet, self).__init__(class_num)
        with self.graph.as_default():
            self.build(res_type, struct)

    def build(self, res_type=None, struct=None):
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

    def subsample(self, inputs, factor, scope=None):
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, 1, stride=factor, scope=scope)

    def resnet_arg_scope(self, is_training=True,
                         conv_weight_decay=CONV_WEIGHT_DECAY,
                         batch_norm_decay=BATCH_NORM_DECAY,
                         batch_norm_epsilon=BATCH_NORM_EPSILON,
                         batch_norm_scale=BATCH_NORM_SCALE):
        batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
        }

        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            weights_regularizer=slim.l2_regularizer(
                                conv_weight_decay),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.fully_connected], activation_fn=None, normalizer_fn=None):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                        return arg_sc

    @slim.add_arg_scope
    def stack_blocks_dense(self, net, blocks, outputs_collections=None):
        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as inter_scope:
                unit_depth, unit_depth_bottleneck, unit_num = block.args
                for i in range(unit_num):
                    with tf.variable_scope('unit_%d' % i, values=[net]):
                        unit_stride = 2 if i is 0 and not block.is_first else 1
                        net = block.unit_fn(net, depth=unit_depth, stride=unit_stride,
                                            depth_bottleneck=unit_depth_bottleneck)
                net = slim.utils.collect_named_outputs(
                    outputs_collections, inter_scope.name, net)
        return net

    @slim.add_arg_scope
    # 定义核心bottleneck残差学习单元
    def bottleneck(self, inputs, depth, depth_bottleneck, stride,
                   outputs_collections=None, scope=None):
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as inter_scope:
            depth_in = slim.utils.last_dimension(
                inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(
                inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = self.subsample(preact, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, 1, stride=stride,
                                       activation_fn=None, scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck,
                                   1, stride=1, scope='conv1')
            residual = slim.conv2d(residual, depth_bottleneck,
                                   3, stride=stride, scope='conv2')
            residual = slim.conv2d(residual, depth, 1, activation_fn=None,
                                   normalizer_fn=None, stride=1, scope='conv3')
            output = tf.add(shortcut, residual)
            return slim.utils.collect_named_outputs(outputs_collections, inter_scope.name, output)

    def resnet_v2(self, blocks, class_num, reuse=None, scope=None):
        with tf.variable_scope(scope, 'resnet_v2', [self.inputs], reuse=reuse):
            net = slim.conv2d(self.inputs, 64, 7, activation_fn=None,
                              normalizer_fn=None, stride=2, scope='preconv')
            net = slim.max_pool2d(net, [3, 3], stride=2, scope='prepool')
            net = self.stack_blocks_dense(net, blocks)
            net = slim.batch_norm(
                net, activation_fn=tf.nn.relu, scope='postbn')
            net = tf.reduce_mean(net, [1, 2], name='postpool', keepdims=True)
            # net = tf.nn.avg_pool(net, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME', name='postpool')
            net = slim.flatten(net, scope='flatten')
            logits = net = slim.fully_connected(net, class_num, scope='fc')
            cross_entropy = tf.losses.softmax_cross_entropy(
                self.label, logits, scope='loss')
            self.loss = tf.losses.get_total_loss(False)
            self.prediction = slim.softmax(logits, scope='prediction')
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(logits, 1), tf.argmax(self.label, 1)), tf.float32))

            learning_rate = tf.train.exponential_decay(
                LEARNING_RATE, self.train_step, LR_DECAY_STEP, LR_DECAY_RATE)
            # self.trainer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(
            #     self.loss, global_step=self.train_step)
            self.trainer = tf.train.AdamOptimizer(learning_rate).minimize(
                self.loss, global_step=self.train_step)
            self.saver = tf.train.Saver()
            self.logger.info('Build Net OK')

            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('lr', learning_rate)
            self.summary = tf.summary.merge(
                tf.get_collection(tf.GraphKeys.SUMMARIES))
            self.writer = tf.summary.FileWriter(SUMMARY_PATH, self.graph)

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
        with slim.arg_scope(self.resnet_arg_scope(is_training=True)):
            with slim.arg_scope([slim.conv2d, self.bottleneck, self.stack_blocks_dense]):
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


class ResNet_v2(Net):
    def __init__(self, resnet_size=32, data_format=None,
                 resnet_version=DEFAULT_VERSION,
                 bottleneck=True, class_num=CLASS_NUM,
                 num_gpu=NUM_GPU, ps_type=PS_TYPE):
        assert isinstance(
            num_gpu, int) and num_gpu >= 0, 'GPU count must be 0 or a positive integer.'
        assert ps_type in (
            'CPU', 'GPU'), 'Parameter server must be CPU or GPU'
        assert ps_type != 'GPU' or num_gpu is not 0, 'GPU num is 0, CPU must be used as parameter server.'
        assert resnet_size % 6 == 2, 'Resnet size must be 6n + 2'
        assert resnet_version in (1, 2), 'Resnet version should be 1 or 2.'

        self.num_gpu = num_gpu
        self.device_type = 'cpu' if num_gpu is 0 else 'gpu'
        self.ps_type = ps_type
        self.ps_device = '/gpu:0' if ps_type == 'GPU' else '/cpu:0'

        if data_format is None:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'

        if bottleneck:
            if resnet_version is 1:
                self.block_fn = self._bottleneck_block_v1
            else:
                self.block_fn = self._bottleneck_block_v2
        else:
            if resnet_version is 1:
                self.block_fn = self._building_block_v1
            else:
                self.block_fn = self._building_block_v2

        self.resnet_size = resnet_size
        self.resnet_version = resnet_version
        self.data_format = data_format
        self.bottleneck = bottleneck
        self.num_filters = 16
        self.kernel_size = 3
        self.conv_stride = 1
        self.first_pool_size = None
        self.first_pool_stride = None
        self.num_blocks = (resnet_size - 2) // 6
        self.block_sizes = [self.num_blocks] * 3
        self.block_strides = [1, 2, 2]
        self.final_size = 64
        self.pre_activation = resnet_version == 2
        self.update_ops = None

        super(ResNet_v2, self).__init__(class_num)
        with self.graph.as_default():
            self.build()

    def device_setter(self, worker_device, ps_ops=None,):
        if ps_ops is None:
            ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

        if self.ps_type == 'CPU':
            ps_strategy = device_setter._RoundRobinStrategy(
                self.num_gpu)
        else:
            ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(
                self.num_gpu, tf.contrib.training.byte_size_load_fn)

        def device_chooser(op):
            current_device = pydev.DeviceSpec.from_string(op.device or "")
            node_def = op if isinstance(
                op, node_def_pb2.NodeDef) else op.node_def
            if node_def.op in ps_ops:
                ps_device_spec = pydev.DeviceSpec.from_string(
                    '/{}:{}'.format(self.ps_type.lower(), ps_strategy(op)))

                ps_device_spec.merge_from(current_device)
                return ps_device_spec.to_string()
            else:
                worker_device_spec = pydev.DeviceSpec.from_string(
                    worker_device or "")
                worker_device_spec.merge_from(current_device)
                return worker_device_spec.to_string()
        return device_chooser

    def data_split(self, img, label, batch_size=BATCH_SIZE):
        with tf.device('/cpu:0'), tf.variable_scope('data_split'):
            if self.num_gpu <= 1:
                # No GPU available or only 1 GPU.
                return [img], [label]

            gap = batch_size // self.num_gpu
            with tf.variable_scope('img_split'):
                img_batch = tf.unstack(img, num=batch_size, axis=0)
                img_shards = [img_batch[i*gap:(i+1)*gap]
                              for i in range(self.num_gpu)]
                img_shards[self.num_gpu-1].extend(img_batch[self.num_gpu*gap:])
                img_shards = [tf.parallel_stack(x) for x in img_shards]
            with tf.variable_scope('label_split'):
                label_batch = tf.unstack(label, num=batch_size, axis=0)
                label_shards = [
                    label_batch[i*gap:(i+1)*gap] for i in range(self.num_gpu)]
                label_shards[self.num_gpu -
                             1].extend(label_batch[self.num_gpu*gap:])
                label_shards = [tf.parallel_stack(x) for x in label_shards]
            return img_shards, label_shards

    def build(self):
        with tf.variable_scope('resnet_main'):
            num_device = self.num_gpu if self.num_gpu is not 0 else 1
            img_shards, label_shards = self.data_split(self.img, self.label)
            for i in range(num_device):
                self.build_tower(img_shards[i], label_shards[i], i)
            self.build_trainer()

    def build_tower(self, inputs, label, tower_id, training=True):
        worker_device = '/{}:{}'.format(self.device_type, tower_id)
        device_setting = self.device_setter(worker_device)
        with tf.variable_scope('resnet', reuse=bool(tower_id != 0)):
            with tf.device(device_setting):
                with tf.name_scope('tower_%d' % tower_id) as name_scope:
                    with tf.variable_scope('pre_process'):
                        if self.data_format == 'channels_first':
                            net = tf.transpose(inputs, [0, 3, 1, 2])
                        else:
                            net = inputs

                        net = self.conv2d_fixed_padding(
                            net, self.num_filters, self.kernel_size, self.conv_stride)
                        net = tf.identity(net, 'pre_conv')

                        # We do not include batch normalization or activation functions in V2
                        # for the initial conv1 because the first ResNet unit will perform these
                        # for both the shortcut and non-shortcut paths as part of the first
                        # block's projection. Cf. Appendix of [2].
                        if self.resnet_version == 1:
                            net = self.batch_norm(net, training)
                            net = tf.nn.relu(net)

                        if self.first_pool_size:
                            net = tf.layers.max_pooling2d(
                                inputs=net, pool_size=self.first_pool_size,
                                strides=self.first_pool_stride, padding='SAME',
                                data_format=self.data_format)
                            net = tf.identity(net, 'initial_max_pool')

                    for i, num_blocks in enumerate(self.block_sizes):
                        num_filters = self.num_filters * (2 ** i)
                        net = self.block_layer(
                            inputs=net, filters=num_filters, blocks=num_blocks,
                            strides=self.block_strides[i], training=training,
                            name='block_layer{}'.format(i + 1))

                    # Only apply the BN and ReLU for model that does pre_activation in each
                    # building/bottleneck block, eg resnet V2.
                    with tf.variable_scope('post_process'):
                        if self.pre_activation:
                            net = self.batch_norm(net, training)
                            net = tf.nn.relu(net)

                        # The current top layer has shape
                        # `batch_size x pool_size x pool_size x final_size`.
                        # ResNet does an Average Pooling layer over pool_size,
                        # but that is the same as doing a reduce_mean. We do a reduce_mean
                        # here because it performs better than AveragePooling2D.
                        axes = [2, 3] if self.data_format == 'channels_first' else [
                            1, 2]
                        net = tf.reduce_mean(net, axes, keepdims=True)
                        net = tf.identity(net, 'final_reduce_mean')
                        net = tf.layers.flatten(net, 'flatten')
                        net = tf.layers.dense(inputs=net, units=self.class_num)
                        net = tf.identity(net, 'final_dense')

                    with tf.variable_scope('result'):
                        logits = tf.cast(net, tf.float32, 'logits')

                        prediction = tf.nn.softmax(logits, name='prediction')
                        accuracy = tf.reduce_mean(tf.cast(
                            tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1)), tf.float32))

                        # Calculate loss, which includes softmax cross entropy and L2 regularization.
                        cross_entropy = tf.losses.softmax_cross_entropy(
                            label, logits)
                        tf.identity(cross_entropy, name='cross_entropy')
                        # Add weight decay to the loss.
                        l2_loss = CONV_WEIGHT_DECAY * tf.add_n(
                            [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
                             if self.exclude_batch_norm(v.name)])
                        loss = {
                            'l2_loss': l2_loss,
                            'cross_entropy': cross_entropy,
                            'loss': l2_loss + cross_entropy
                        }
                        model_params = tf.trainable_variables()
                        grad = tf.gradients(loss['loss'], model_params)
                        self.tower_loss.append(loss)
                        self.tower_grads.append(zip(grad, model_params))
                        self.tower_preds.append(prediction)
                        self.tower_accuracy.append(accuracy)

                        if tower_id is 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            self.update_ops = tf.get_collection(
                                tf.GraphKeys.UPDATE_OPS, name_scope)

    def build_trainer(self):
        with tf.variable_scope('train_net'):
            gradvars = []
            with tf.name_scope('gradient_averaging'):
                all_grads = {}
                for grad, var in itertools.chain(*self.tower_grads):
                    if grad is not None:
                        all_grads.setdefault(var, []).append(grad)
                for var, grads in all_grads.items():
                    # Average gradients on the same device as the variables
                    # to which they apply.
                    with tf.device(var.device):
                        if len(grads) == 1:
                            avg_grad = grads[0]
                        else:
                            avg_grad = tf.multiply(
                                tf.add_n(grads), 1. / len(grads))
                    gradvars.append((avg_grad, var))

            # Device that runs the ops to apply global gradient updates.
            with tf.device(self.ps_device):
                learning_rate = tf.train.exponential_decay(
                    LEARNING_RATE, self.train_step, LR_DECAY_STEP, LR_DECAY_RATE)
                optimizer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
                # optimizer = tf.train.AdamOptimizer(learning_rate)

                train_op = [
                    optimizer.apply_gradients(
                        gradvars, global_step=self.train_step)
                ]
                train_op.extend(self.update_ops)
                self.trainer = tf.group(*train_op)

                cross_entropy = tf.reduce_mean(
                    [loss['cross_entropy'] for loss in self.tower_loss])
                l2_loss = tf.reduce_mean([loss['l2_loss']
                                          for loss in self.tower_loss])
                self.loss = tf.reduce_mean(
                    [loss['loss'] for loss in self.tower_loss])
                self.accuracy = tf.reduce_mean(self.tower_accuracy)
                self.prediction = tf.concat(self.tower_preds, axis=0)
                # self.trainer = tf.train.MomentumOptimizer(learning_rate, MOMENTUM).minimize(
                #     self.loss, global_step=self.train_step)
                # self.trainer = tf.train.AdamOptimizer(learning_rate).minimize(
                #     self.loss, global_step=self.train_step)
                self.saver = tf.train.Saver()
                self.logger.info('Build Net OK')

                tf.summary.scalar('cross_entropy', cross_entropy)
                tf.summary.scalar('l2_loss', l2_loss)
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('accuracy', self.accuracy)
                tf.summary.scalar('lr', learning_rate)
                self.summary = tf.summary.merge(
                    self.graph.get_collection(tf.GraphKeys.SUMMARIES))
                self.writer = tf.summary.FileWriter(SUMMARY_PATH, self.graph)

    def exclude_batch_norm(self, name):
        return 'batch_normalization' not in name

    def batch_norm(self, inputs, training):
        """Performs a batch normalization using a standard set of parameters."""
        # We set fused=True for a significant performance boost. See
        # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
        return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if self.data_format == 'channels_first' else 3,
            momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON, center=True,
            scale=BATCH_NORM_SCALE, training=training, fused=True)

    def fixed_padding(self, inputs, kernel_size):
        """Pads the input along the spatial dimensions independently of input size.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                        Should be a positive integer.

        Returns:
            A tensor with the same format as the input with the data either intact
            (if kernel_size == 1) or padded (if kernel_size > 1).
        """
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if self.data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, name=None):
        """Strided 2-D convolution with explicit padding."""
        # The padding is consistent and is based only on `kernel_size`, not on the
        # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'), use_bias=False, data_format=self.data_format,
            kernel_initializer=tf.variance_scaling_initializer(), name=name)

    def projection_shortcut(self, inputs, filters, strides):
                # Bottleneck blocks end with 4x the number of filters as they start with
        filters_out = filters * 4 if self.bottleneck else filters
        return self.conv2d_fixed_padding(inputs, filters_out, 1, strides)

    def _building_block_v1(self, inputs, filters, training, strides, is_pro_shortcut=False):
        """A single block for ResNet v1, without a bottleneck.

        Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.

        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if is_pro_shortcut is not None:
            shortcut = self.projection_shortcut(inputs, filters, strides)
            shortcut = self.batch_norm(shortcut, training)

        inputs = self.conv2d_fixed_padding(inputs, filters, 3, strides)
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv2d_fixed_padding(inputs, filters, 3, 1)
        inputs = self.batch_norm(inputs, training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs

    def _building_block_v2(self, inputs, filters, training, strides, is_pro_shortcut=False):
        """A single block for ResNet v2, without a bottleneck.

        Batch normalization then ReLu then convolution as described by:
            Identity Mappings in Deep Residual Networks
            https://arxiv.org/pdf/1603.05027.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if is_pro_shortcut is not None:
            shortcut = self.projection_shortcut(inputs, filters, strides)

        inputs = self.conv2d_fixed_padding(inputs, filters, 3, strides)

        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(inputs, filters, 3, 1)

        return inputs + shortcut

    def _bottleneck_block_v1(self, inputs, filters, training, strides, is_pro_shortcut=False):
        """A single block for ResNet v1, with a bottleneck.

        Similar to _building_block_v1(), except using the "bottleneck" blocks
        described in:
            Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs

        if is_pro_shortcut is not None:
            shortcut = self.projection_shortcut(inputs, filters, strides)
            shortcut = self.batch_norm(shortcut, training)

        inputs = self.conv2d_fixed_padding(inputs, filters, 1, 1)
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv2d_fixed_padding(inputs, filters, 3, strides)
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv2d_fixed_padding(inputs, 4 * filters, 1, 1)
        inputs = self.batch_norm(inputs, training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs

    def _bottleneck_block_v2(self, inputs, filters, training, strides, is_pro_shortcut=False):
        """A single block for ResNet v2, without a bottleneck.

        Similar to _building_block_v2(), except using the "bottleneck" blocks
        described in:
            Convolution then batch normalization then ReLU as described by:
            Deep Residual Learning for Image Recognition
            https://arxiv.org/pdf/1512.03385.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

        Adapted to the ordering conventions of:
            Batch normalization then ReLu then convolution as described by:
            Identity Mappings in Deep Residual Networks
            https://arxiv.org/pdf/1603.05027.pdf
            by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the convolutions.
            training: A Boolean for whether the model is in training or inference
            mode. Needed for batch normalization.
            projection_shortcut: The function to use for projection shortcuts
            (typically a 1x1 convolution when downsampling the input).
            strides: The block's stride. If greater than 1, this block will ultimately
            downsample the input.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block; shape should match inputs.
        """
        shortcut = inputs
        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if is_pro_shortcut is not None:
            shortcut = self.projection_shortcut(inputs, filters, strides)

        inputs = self.conv2d_fixed_padding(inputs, filters, 1, 1)

        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(inputs, filters, 3, strides)

        inputs = self.batch_norm(inputs, training)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(inputs, 4 * filters, 1, 1)

        return inputs + shortcut

    def block_layer(self, inputs, filters, blocks, strides,
                    training, name):
        """Creates one layer of blocks for the ResNet model.

        Args:
            inputs: A tensor of size [batch, channels, height_in, width_in] or
            [batch, height_in, width_in, channels] depending on data_format.
            filters: The number of filters for the first convolution of the layer.
            bottleneck: Is the block created a bottleneck block.
            block_fn: The block to use within the model, either `building_block` or
            `bottleneck_block`.
            blocks: The number of blocks contained in the layer.
            strides: The stride to use for the first convolution of the layer. If
            greater than 1, this layer will ultimately downsample the input.
            training: Either True or False, whether we are currently training the
            model. Needed for batch norm.
            name: A string name for the tensor output of the block layer.
            data_format: The input format ('channels_last' or 'channels_first').

        Returns:
            The output tensor of the block layer.
        """
        # Only the first block per block_layer uses projection_shortcut and strides
        with tf.variable_scope(name):
            with tf.variable_scope('block_conv_1'):
                inputs = self.block_fn(
                    inputs, filters, training, strides, True)

            for i in range(1, blocks):
                with tf.variable_scope('block_conv_%d' % (i + 1)):
                    inputs = self.block_fn(inputs, filters, training, 1, False)

            return inputs


if __name__ == '__main__':
    pass
