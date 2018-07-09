from DM.DataManagerNii import DataManagerNii as DMN
from Net.resnet import ResNet


def transfer_to_tfr(dm=None):
    if dm is None:
        dm = DMN()

    dm.createFileList()
    dm.loadImage()
    dm.loadGT()
    dm.getNumpyData()
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
    net.save('model_1')


def verify(dm, net):
    dm.clear_data()
    if not dm.testList:
        dm.splitData()

    ver_tfr_list = dm.get_tfrecord_path(dm.testList)
    acc, loss = net.verify(ver_tfr_list)


def main():
    dm = DMN()
    net = ResNet(res_type=50)
    transfer_to_tfr(dm)
    train(dm, net)


if __name__ == '__main__':
    main()
