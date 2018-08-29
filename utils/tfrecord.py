import random

import tensorflow as tf

from config import CLASS_NUM, MOD_NUM, REPEAT_NUM, SIZE


def tfrecord_parse(example):
    parsed_example = tf.parse_single_example(
        serialized=example,
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.cast(
        tf.reshape(tf.decode_raw(
            parsed_example['img'], tf.float32), (SIZE, SIZE, MOD_NUM)),
        tf.float32
    )
    label = tf.cast(
        tf.decode_raw(parsed_example['label'], tf.float32),
        tf.float32
    )
    return img, label


def generate_example(img, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
            }
        )
    )


def generate_writer(path):
    return tf.python_io.TFRecordWriter(path)


def generate_dataset(files_list, batch_size, repeat_time=REPEAT_NUM,
                     train=True, shuffle=True, batch=True):
    if shuffle:
        random.shuffle(files_list)
    dataset = tf.data.TFRecordDataset(files_list)
    dataset = dataset.map(tfrecord_parse)
    if shuffle:
        dataset = dataset.shuffle(len(files_list) * 500)
    if train:
        dataset = dataset.repeat(repeat_time)
    if batch:
        dataset = dataset.batch(batch_size, True)
    else:
        dataset = dataset.batch(500)

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    return iterator, next_batch
