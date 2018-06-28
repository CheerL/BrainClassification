import copy
import math
import os
from os import listdir
from os.path import isfile, join, splitext

#import cv2
import numpy as np
import SimpleITK as sitk
import skimage.transform
from DM.DataManager import DataManager
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool



class DataManagerNii(DataManager):
    def loadImage(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod in self.mods:
                    name = '_'.join(img.split('_')[:-1])
                    if name not in self.sitkImage.keys():
                        self.sitkImage[name] = dict()
                    self.sitkImage[name][mod] = sitk.Cast(sitk.ReadImage(
                        os.path.join(img_dir, img)), sitk.sitkFloat32)

    def loadGT(self, fileList=None):
        if fileList is None:
            fileList = self.fileList

        for img_dir in tqdm(fileList):
            for img in os.listdir(img_dir):
                mod = img.split('_')[-1].split('.')[0]
                if mod is 'seg':
                    name = '_'.join(img.split('_')[:-1])
                    self.sitkGT[name] = sitk.Cast(sitk.ReadImage(
                        os.path.join(img_dir, img)), sitk.sitkFloat32)

    def getNumpyData(self, dat, method):
        ret = dict()
        self.originalSizes = dict()
        for key in tqdm(dat):
            img = dat[key]
            ret[key] = sitk.GetArrayFromImage(img).astype(dtype=np.float32)
            self.originalSizes[key]=ret[key].shape

        return ret

    def writeResultsFromNumpyLabel(self, result, key,original_image=False):
        if self.probabilityMap:
            result = result * 255
        else:
            pass
            # result = result>0.5
            # result = result.astype(np.uint8)
        result = np.transpose(result, [2, 1, 0])
        toWrite = sitk.GetImageFromArray(result)

        if original_image:
            toWrite = sitk.Cast(toWrite, sitk.sitkFloat32)
        else:
            toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = splitext(key)
        # print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(join(self.resultsDir, filename + '_result.nii.gz'))
        writer.Execute(toWrite)
