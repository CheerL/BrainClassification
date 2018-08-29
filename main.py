import os
import random
import shutil

from config import TEST_TFR_PATH, TRAIN_TFR_PATH, VAL_TFR_PATH, TFR_PATH, TEST_RATE, VAL_RATE
from DM.DataManagerNii import DataManagerNii as DMN
from Net.resnet import ResNet, ResNet_v2


def transfer_to_tfr(dm):
    dm.create_file_list()
    file_list = dm.file_list
    for i in range(0, len(file_list), 150):
        dm.file_list = file_list[i:i+150]
        dm.load_image()
        dm.load_label()
        dm.numpy_data = dm.get_numpy_data()
        dm.write_tfrecord(clear=False)
        dm.clear_data()


def train(net, train_tfr_list, val_tfr_list):
    net.train(train_tfr_list, val_tfr_list)
    net.save('model')


def test(net, test_tfr_list):
    net.validate(test_tfr_list, test=True)


def generate_list():
    if not (os.listdir(TRAIN_TFR_PATH) and os.listdir(TEST_TFR_PATH) and os.listdir(VAL_TFR_PATH)):
        tfr_list = os.listdir(TFR_PATH)
        total_num = len(tfr_list)
        train_num_list = set(range(total_num))
        test_num = int(total_num * TEST_RATE)
        test_num_list = random.sample(train_num_list, test_num)
        train_num_list = train_num_list.difference(set(test_num_list))
        val_num = int(total_num * (1 - TEST_RATE) * VAL_RATE)
        val_num_list = random.sample(train_num_list, val_num)
        train_num_list = list(train_num_list.difference(set(val_num_list)))
        train_tfr_list = [path for i, path in enumerate(
            tfr_list) if i in train_num_list]
        test_tfr_list = [path for i, path in enumerate(
            tfr_list) if i in test_num_list]
        val_tfr_list = [path for i, path in enumerate(
            tfr_list) if i in val_num_list]

        for path in train_tfr_list:
            shutil.copyfile(os.path.join(TFR_PATH, path),
                            os.path.join(TRAIN_TFR_PATH, path))
        for path in test_tfr_list:
            shutil.copyfile(os.path.join(TFR_PATH, path),
                            os.path.join(TEST_TFR_PATH, path))
        for path in val_tfr_list:
            shutil.copyfile(os.path.join(TFR_PATH, path),
                            os.path.join(VAL_TFR_PATH, path))

    train_tfr_list = [os.path.join(TRAIN_TFR_PATH, path) for path in os.listdir(
        TRAIN_TFR_PATH) if 'tfrecord' in path]
    test_tfr_list = [os.path.join(TEST_TFR_PATH, path) for path in os.listdir(
        TEST_TFR_PATH) if 'tfrecord' in path]
    val_tfr_list = [os.path.join(VAL_TFR_PATH, path) for path in os.listdir(
        VAL_TFR_PATH) if 'tfrecord' in path]
    return train_tfr_list, val_tfr_list, test_tfr_list


def main():
    # dm = DMN()
    # transfer_to_tfr(dm)
    train_tfr_list, val_tfr_list, test_tfr_list = generate_list()
    net = ResNet_v2()
    net.load('log/summary_Aug_16_18_02_09_2018/model/model')
    train(net, train_tfr_list, val_tfr_list)
    test(net, test_tfr_list)


if __name__ == '__main__':
    main()
