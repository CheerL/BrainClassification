import os
#import cv2
import random
from collections import deque

from config import DATA_PATH, MODS, RESULT_PATH
from utils.lazy_property import lazy_property


class DataManager(object):
    def __init__(self, srcFolder=DATA_PATH, resultsDir=RESULT_PATH, parameters=None, mods=MODS):
        self.mods = mods
        self.params = parameters
        self.srcFolder = srcFolder
        self.resultsDir = resultsDir
        self.fileList = list()
        self.trainList = list()
        self.testList = list()
        self.sitkImage = dict()
        self.sitkGT = dict()
        self.numpyData = dict()
        self.test_rate = 0.1

    @property
    def dim(self):
        if not hasattr(self, '_dim'):
            setattr(self, '_dim', 0)
        return getattr(self, '_dim', 0)

    @dim.setter
    def dim(self, val):
        if getattr(self, '_dim', 0) is not val:
            setattr(self, '_dim', val)
            if hasattr(self, '_lazy_left_dim'):
                delattr(self, '_lazy_left_dim')

    @lazy_property
    def left_dim(self):
        left_dim = set(range(3))
        left_dim.remove(self.dim)
        return tuple(left_dim)

    def createFileList(self, limit=0):
        ''' find all directories containing images and put there name in the fileList'''
        if self.fileList:
            self.fileList.clear()

        stack = deque([self.srcFolder])

        while stack:
            now_path = stack.pop()
            if os.path.isdir(now_path):
                stack.extend([os.path.join(now_path, sub)
                              for sub in os.listdir(now_path)])
            else:
                if now_path.endswith('nii.gz'):
                    now_dir = os.path.dirname(now_path)
                    if now_dir not in self.fileList:
                        self.fileList.append(now_dir)

        self.checkFileList()
        if limit > 0:
            self.fileList = random.sample(self.fileList, k=limit)

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

    def loadTrainData(self, file_list=None):
        ''' load training data'''
        if file_list is None:
            if not (self.trainList and self.trainList):
                self.splitData()
            file_list = self.trainList

        self.loadImage(file_list)
        self.loadGT(file_list)

    def loadTestData(self, file_list=None):
        ''' load testing data or validation data'''
        if file_list is None:
            if not (self.trainList and self.testList):
                self.splitData()
            file_list = self.testList

        self.loadImage(file_list)
        self.loadGT(file_list)

    def splitData(self, test_rate=None):
        if not self.fileList:
            self.createFileList()

        if test_rate is None:
            test_rate = self.test_rate

        total_num = len(self.fileList)
        test_num = int(total_num * test_rate)
        test_num_list = random.sample(range(total_num), test_num)
        self.testList = [path for i, path in enumerate(
            self.fileList) if i in test_num_list]
        self.trainList = [path for i, path in enumerate(
            self.fileList) if i not in test_num_list]

    def getTrainNumpyData(self, file_list=None):
        self.loadTrainData(file_list)
        if file_list is None:
            file_list = self.trainList
        self.numpyData = self.getNumpyData(file_list)

    def getTestNumpyData(self, file_list=None):
        self.loadTestData(file_list)
        if file_list is None:
            file_list = self.testList

        self.numpyData = self.getNumpyData(file_list)

    def getNumpyData(self, file_list):
        ''' load numpy data from sitk data'''
        raise NotImplementedError()

    def clear_data(self):
        self.sitkGT.clear()
        self.sitkImage.clear()
        self.numpyData.clear()
