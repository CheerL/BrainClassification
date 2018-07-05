import os
#import cv2
import random
from collections import deque

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def lazy_property(func):
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property

class DataManager(object):
    def __init__(self, srcFolder, resultsDir, parameters, probabilityMap=False, mods=['t1']):
        self.mods = mods
        self.params = parameters
        self.srcFolder = srcFolder.strip('/').strip('\\').strip()
        self.resultsDir = resultsDir.strip('/').strip('\\').strip()
        self.probabilityMap = probabilityMap
        self.fileList = list()
        self.trainList = list()
        self.testList = list()
        self.sitkImage = dict()
        self.sitkGT = dict()
        self.test_rate = 0.1
        self.numpyData = None
        self.meanIntensityTrain = None
        self.originalSizes = dict()

        self.createFileList()

        self._dim = 0

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, val):
        self._dim = val
        if hasattr(self, '_lazy_left_dim'):
            delattr(self, '_lazy_left_dim')

    @lazy_property
    def left_dim(self):
        left_dim = set(range(3))
        left_dim.remove(self.dim)
        return tuple(left_dim)

    def createFileList(self, limit=None):
        ''' find all directories containing images and put there name in the fileList'''
        if self.fileList:
            self.fileList.clear()

        stack = deque([self.srcFolder])

        while stack:
            if limit is not None and len(self.fileList) >= limit:
                break

            now_path = stack.pop()
            if os.path.isdir(now_path):
                stack.extend([os.path.join(now_path, sub) for sub in os.listdir(now_path)])
            else:
                if now_path.endswith('nii.gz'):
                    now_dir = os.path.dirname(now_path)
                    if now_dir not in self.fileList:
                        self.fileList.append(now_dir)

        self.checkFileList()

    def checkFileList(self):
        for img_dir in self.fileList:
            img_str = ''.join(os.listdir(img_dir))
            for mod in self.mods:
                if mod not in img_str:
                    self.fileList.remove(img_dir)
                    break

    def loadImage(self, fileList=None):
        ''' load images from image.nii'''
        raise NotImplementedError()


    def loadGT(self, fileList=None):
        ''' load labels from label.nii'''
        raise NotImplementedError()

    def loadTrainData(self):
        ''' load training data'''
        if not (self.trainList and self.testList):
            self.splitData()

        self.loadImage(self.trainList)
        self.loadGT(self.trainList)

    def loadTestData(self):
        ''' load testing data or validation data'''
        if not (self.trainList and self.testList):
            self.splitData()

        self.loadImage(self.testList)
        self.loadGT(self.testList)

    def splitData(self):
        total_num = len(self.fileList)
        test_num = int(total_num * self.test_rate)
        test_num_list = random.sample(range(total_num), test_num)
        self.testList = [path for i, path in enumerate(self.fileList) if i in test_num_list]
        self.trainList = [path for i, path in enumerate(self.fileList) if i not in test_num_list]

    def getTrainNumpyData(self, file_list=None):
        if file_list is None:
            file_list = self.trainList

        self.loadTrainData()
        self.numpyData = self.getNumpyData(file_list, sitk.sitkLinear)

    def getTestNumpyData(self, file_list=None):
        if file_list is None:
            file_list = self.testList

        self.loadTestData()
        self.numpyData = self.getNumpyData(file_list, sitk.sitkLinear)

    def getNumpyData(self, data, method):
        ''' load numpy data from sitk data'''
        raise NotImplementedError()

    def writeResultsFromNumpyLabel(self, result, key):
        ''' save the segmentation results to the result directory'''
        result = np.transpose(result, [2, 1, 0])

        if self.probabilityMap:
            result = result * 255
        else:
            result = result>0.5
            result = result.astype(np.uint8)
        toWrite = sitk.GetImageFromArray(result)

        toWrite = sitk.Cast(toWrite, sitk.sitkUInt8)

        writer = sitk.ImageFileWriter()
        filename, ext = os.path.splitext(key)
        # #print join(self.resultsDir, filename + '_result' + ext)
        writer.SetFileName(os.path.join(self.resultsDir, filename + '_result.nii'))
        writer.Execute(toWrite)
