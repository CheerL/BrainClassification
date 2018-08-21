import tensorflow as tf

from config import MOD_NUM, SIZE, CLASS_NUM, REPEAT_NUM


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


def generate_dataset(files_list, batch_size,
                     train=True, repeat_time=REPEAT_NUM,
                     shuffle=True, batch=True):
    dataset = tf.data.TFRecordDataset(files_list)
    if shuffle:
        dataset = dataset.shuffle(len(files_list) * 500)
    if train:
        dataset = dataset.repeat(repeat_time)
    if batch:
        dataset = dataset.batch(batch_size, True)
    else:
        dataset = dataset.batch(1000)

    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    example = tf.parse_example(
        next_batch,
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.cast(
        tf.reshape(tf.decode_raw(
            example['img'], tf.float32), (-1, SIZE, SIZE, MOD_NUM)),
        tf.float32
    )
    label = tf.cast(
        tf.reshape(tf.decode_raw(
            example['label'], tf.float32), (-1, CLASS_NUM)),
        tf.float32
    )
    return iterator, [img, label]
