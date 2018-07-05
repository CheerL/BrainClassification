import os

#import cv2
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from DM.DataManager import DataManager

# from pathos.multiprocessing import ProcessingPool as Pool


class DataManagerNii(DataManager):
    def loadImage(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod in self.mods:
                    if name not in self.sitkImage.keys():
                        self.sitkImage[name] = dict()
                    self.sitkImage[name][mod] = sitk.Cast(sitk.ReadImage(
                        os.path.join(img_dir, img)), sitk.sitkFloat32)

    def loadGT(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            name = os.path.split(img_dir)[-1]
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod == 'seg':
                    self.sitkGT[name] = sitk.Cast(sitk.ReadImage(
                        os.path.join(img_dir, img)), sitk.sitkFloat32)
                    break
            else:
                size = self.sitkImage[name][self.mods[0]].GetSize()
                self.sitkGT[name] = sitk.Image(size, sitk.sitkFloat32)


    def getNumpyData(self, file_list, method):
        numpy_data = dict()
        for img_dir in file_list:
            name = os.path.split(img_dir)[-1]

            img_dict = self.sitkImage[name]
            if name not in numpy_data.keys():
                numpy_data[name] = dict()
            for mod, img in img_dict.items():
                numpy_data[name][mod] = sitk.GetArrayFromImage(
                    img).astype(dtype=np.float32)
            numpy_data[name]['label'] = self.getLabel(self.sitkGT[name])
        return numpy_data

    def getLabel(self, data):
        numpy_data = sitk.GetArrayFromImage(data)
        label = numpy_data.sum(axis=self.left_dim).astype(np.bool).astype(np.float32)
        return label

    def writeResultsFromNumpyLabel(self, result, key, original_image=False):
        # if self.probabilityMap:
        #     result = result * 255
        # else:
        #     pass
        #     # result = result>0.5
        #     # result = result.astype(np.uint8)
        # result = np.transpose(result, [2, 1, 0])
        # toWrite = sitk.GetImageFromArray(result)

        # if original_image:
        #     toWrite = sitk.Cast(toWrite, sitk.sitkFloat32)
        # else:
        #     toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        # writer = sitk.ImageFileWriter()
        # filename, ext = splitext(key)
        # # print join(self.resultsDir, filename + '_result' + ext)
        # writer.SetFileName(join(self.resultsDir, filename + '_result.nii.gz'))
        # writer.Execute(toWrite)
        pass

if __name__ == '__main__':
    dm = DataManagerNii('MICCAI_BraTS17_Data_Training/', '', '')
    dm.createFileList(20)
    dm.getTrainNumpyData()
