import os
import random

from config import TEST_TFR_PATH, TFR_PATH, TEST_RATE
from DM.DataManagerNii import DataManagerNii as DMN
from Net.resnet import ResNet, ResNet_v2


def transfer_to_tfr(dm=None):
    if dm is None:
        dm = DMN()

    dm.create_file_list()
    dm.load_image()
    dm.load_label()
    dm.numpy_data = dm.get_numpy_data()
    dm.write_tfrecord()


def train(net=None, train_tfr_list=None, val_tfr_list=None, dm=None):
    if train_tfr_list is None:
        if dm is None:
            dm = DMN()

        dm.clear_data()
        if not dm.train_list:
            dm.split_data()
        train_tfr_list = dm.get_tfrecord_path(dm.train_list)
        val_tfr_list = dm.get_tfrecord_path(dm.val_list)

    if net is None:
        net = ResNet()

    net.train(train_tfr_list, val_tfr_list)
    net.save('model')


def test(net, test_tfr_list=None):
    if test_tfr_list is None:
        test_tfr_list = generate_list()[2]

    net.validate(test_tfr_list, test=True)


def generate_list():
    tfr_list = os.listdir(TFR_PATH)
    total_num = len(tfr_list)
    val_num = int(total_num * TEST_RATE)
    val_num_list = random.sample(range(total_num), val_num)
    train_tfr_list = [os.path.join(TFR_PATH, path)
                     for i, path in enumerate(tfr_list) if i not in val_num_list]
    val_tfr_list = [os.path.join(TFR_PATH, path)
                    for i, path in enumerate(tfr_list) if i in val_num_list]
    test_tfr_list = [os.path.join(TEST_TFR_PATH, path)
                     for path in os.listdir(TEST_TFR_PATH)]
    return train_tfr_list, val_tfr_list, test_tfr_list


def main():
    # dm = DMN()
    # transfer_to_tfr(dm)
    train_tfr_list, val_tfr_list, test_tfr_list = generate_list()
    net = ResNet_v2()
    train(net, train_tfr_list, val_tfr_list)
    test(net, test_tfr_list)


if __name__ == '__main__':
    main()
