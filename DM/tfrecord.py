import tensorflow as tf
from config import SIZE, MOD_NUM


def generate_example(img, label):
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tostring()])),
                'label': tf.train.Feature(int64_list=tf.train.FloatList(value=[label]))
            }
        )
    )


def generate_writer(path):
    return tf.python_io.TFRecordWriter(path)


def generate_dataset(files_list, batch_size, verificate=False):
    dataset = tf.contrib.data.TFRecordDataset(files_list)
    dataset = dataset.shuffle(100 * SIZE * SIZE)
    if not verificate:
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.batch(10 * SIZE * SIZE)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    example = tf.parse_example(
        next_batch,
        features={
            'img': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        }
    )
    img = tf.cast(
        tf.reshape(tf.decode_raw(example['img'], tf.float32), (-1, SIZE, SIZE, MOD_NUM)),
        tf.float32
    )
    label = tf.cast(
        tf.reshape(example['label'], [-1, 1]),
        tf.float32
    )
    return iterator, [img, label]
