from DM.DataManagerNii import DataManagerNii as DMN
import Net.resnet as resnet

dm = DMN('data', 'res', None)
dm.createFileList(100)
dm.getNumpyData()
