from DM.DataManagerNii import DataManagerNii as DMN
from Net.resnet import ResNet
from config import TEST_PATH
import os


def transfer_to_tfr(dm=None):
    if dm is None:
        dm = DMN()

    dm.createFileList()
    dm.loadImage()
    dm.loadGT()
    dm.numpyData = dm.getNumpyData()
    dm.write_tfrecord()


def train(dm=None, net=None):
    if dm is None:
        dm = DMN()

    if net is None:
        net = ResNet()

    dm.clear_data()
    if not dm.trainList:
        dm.splitData()

    train_tfr_list = dm.get_tfrecord_path(dm.trainList)
    net.train(train_tfr_list)
    net.save('model')


def verify(dm, net):
    dm.clear_data()
    ver_tfr_list = dm.get_tfrecord_path(dm.testList)
    net.verify(ver_tfr_list)

def test(dm, net):
    test_files = [os.path.join(TEST_PATH, each) for each in os.listdir(TEST_PATH)]

def main():
    dm = DMN()
    # transfer_to_tfr(dm)
    net = ResNet(res_type=50)
    train(dm, net)
    verify(dm, net)


if __name__ == '__main__':
    main()
