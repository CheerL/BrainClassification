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

            numpy_data[name]['label'] = self.get_label(self.sitkGT[name])

            trans_dim = self.left_dim + tuple([self.dim])
            for mod, img in img_dict.items():
                numpy_data[name][mod] = self.get_resample_numpy_data(
                    sitk.GetArrayFromImage(img).astype(dtype=np.float32).transpose(trans_dim)
                )

        return numpy_data

    def get_resample_numpy_data(self, data):
        width, height, _ = data.shape
        if width == self.pic_size and height == self.pic_size:
            resample = data
        elif width <= self.pic_size and height <= self.pic_size:
            pad_wl = (self.pic_size - width) // 2
            pad_hu = (self.pic_size - height) // 2
            pad_wr = self.pic_size - width - pad_wl
            pad_hd = self.pic_size - height - pad_hu
            resample = np.pad(data, ((pad_wl, pad_wr), (pad_hu, pad_hd), (0,0)), 'constant', constant_values=0)
        elif width <= self.pic_size and height > self.pic_size:
            pad_wl = (self.pic_size - width) // 2
            pad_wr = self.pic_size - width - pad_wl

            cut_hu = (height - self.pic_size) // 2
            cut_hd = height - self.pic_size - cut_hu

            resample = np.pad(data, ((pad_wl, pad_wr), (0, 0), (0,0)), 'constant', constant_values=0)
            resample = resample[:, cut_hu:-cut_hd, :]
        elif width > self.pic_size and height <= self.pic_size:
            pad_hu = (self.pic_size - height) // 2
            pad_hd = self.pic_size - height - pad_hu

            cut_wl = (width - self.pic_size) // 2
            cut_wr = width - self.pic_size - cut_wl

            resample = np.pad(data, ((0, 0), (pad_hu, pad_hd), (0,0)), 'constant', constant_values=0)
            resample = resample[cut_wl:-cut_wr, :, :]
        else:
            cut_wl = (width - self.pic_size) // 2
            cut_hu = (height - self.pic_size) // 2
            cut_wr = width - self.pic_size - cut_wl
            cut_hd = height - self.pic_size - cut_hu

            resample = data[cut_wl:-cut_wr, cut_hu:-cut_hd, :]
        return resample

    def get_label(self, data):
        numpy_data = sitk.GetArrayFromImage(data)
        self.dim = numpy_data.shape.index(min(numpy_data.shape))
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
    dm = DataManagerNii('data', 'result', None)
